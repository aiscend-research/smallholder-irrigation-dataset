"""
Contains tests for helper functions for download_sentinel2_mosaics
"""

import unittest
from datetime import datetime, timedelta, date
from download_sentinel2_mosaics import download_sentinel2_mosaic, get_dense_time_windows, calculate_indices, NO_DATA, retrieve_images, retrieve_time_series_stack
import numpy as np
import rasterio
import os
from unittest.mock import patch, MagicMock

class TestDownloadMosaics(unittest.TestCase):
    def test_get_dense_time_windows(self):
        '''
        Tests that the get_dense_time_windows function returns the correct number of time windows, and that 
        the center window is in the middle of the time range.
        '''
        center_date = datetime(2022, 1, 1)
        windows = get_dense_time_windows(center_date)

        # Should return 37 windows
        self.assertEqual(len(windows), 37, 
                         msg=f"test_get_dense_time_windows fail: Expected 37 windows but got {len(windows)}")

        # Each window should be 10 days long
        for window in windows:
            self.assertEqual(window[1] - window[0], timedelta(days=10),
                             msg=f"test_get_dense_time_windows fail: Expected window to be 10 days long, but got window of length {window[1] - window[0]}")

        # Middle window should contain center_date
        self.assertEqual(center_date > windows[18][0] and center_date < windows[18][1], True,
                         msg=f"test_get_dense_time_windows fail: Expect center window to contain center_date")

        # Center date should be in the middle of the midle window
        self.assertEqual(center_date == windows[18][0] + timedelta(days=5) and center_date == windows[18][1] - timedelta(days=5), True,
                         msg=f"test_get_dense_time_windows fail: Expect center_date to be in middle of center_window")

        # Windows are symmetric about the center window. eg: First window is 180 days days away from center window, 
        # last window is 180 days away from center window
        for i in range(18):
            self.assertEqual(windows[18][0] - windows[i][0], windows[36 - i][1] - windows[18][1],
                             msg=f"test_get_dense_time_windows fail: Expect windows to be symmetric about center window")

            self.assertEqual(windows[18][0] - windows[i][0] == timedelta(days=180 - i * 10), True,
                             msg=f"test_get_dense_time_windows fail: Expect windows {18 - i} windows away from center window to be {timedelta(days=180-i*10)} days away from center window")
        
    def test_calculate_indices(self):
        '''
        Tests that NDVI, NDWI, and EVI are calculated correctly
        '''
        # Synthetic 2x2 image
        H, W = 2, 2
        img = np.zeros((10, H, W), dtype=np.int16)

        # Fill bands with per-pixel test values (scaled by 10000, like Sentinel-2 data)
        # Set B8 (NIR), B4 (Red), B2 (Blue), B11 (SWIR)
        
        # Pixel [0,0]
        img[8, 0, 0] = 8000   # B11
        img[7, 0, 0] = 8500   # B8A
        img[6, 0, 0] = 8000   # B8
        img[2, 0, 0] = 2000   # B4
        img[0, 0, 0] = 1000   # B2

        # Pixel [1,1] – Test divide by 0
        img[0, 1, 1] = 0   # B2
        img[2, 1, 1] = 0   # B4
        img[6, 1, 1] = 0  # B8
        img[8, 1, 1] = 0  # B11

        ndvi, evi, ndwi = calculate_indices(img)

        # Pixel[0, 0] – Standard case
        ndvi_00 = 1000 * (0.8 - 0.2) / (0.8 + 0.2)
        evi_00 = 1000 * (2.5 * (0.8 - 0.2) / (0.8 + 6*0.2 - 7.5*0.1 + 1))
        ndwi_00 = 1000 * ((0.8 - 0.8) / (0.8 + 0.8))

        self.assertAlmostEqual(ndvi[0,0], ndvi_00, places=4, 
                               msg=f"test_calculate_indices fail: Expect ndvi to be {ndvi_00} but got {ndvi[0, 0]}")
        self.assertAlmostEqual(evi[0,0], evi_00, places=4, 
                               msg=f"test_calculate_indices fail: Expect evi to be {evi_00} but got {evi[0, 0]}")
        self.assertAlmostEqual(ndwi[0,0], ndwi_00, places=4,
                               msg=f"test_calculate_indices fail: Expect ndwi to be {ndwi_00} but got {ndwi[0, 0]}")

        # Pixel [1, 1] – Test divide by 0
        self.assertEqual(ndvi[1,1], NO_DATA,
                                msg=f"test_calculate_indices fail: Expect ndvi to be {NO_DATA} but got {ndvi[1, 1]}")
        self.assertEqual(evi[1,1], 0,
                                msg=f"test_calculate_indices fail: Expect evi to be {NO_DATA} but got {evi[1, 1]}")
        self.assertEqual(ndwi[1,1], NO_DATA,
                                msg=f"test_calculate_indices fail: Expect ndwi to be {NO_DATA} but got {ndwi[1, 1]}")

        # Test that NDVI, NDWI, and EVI are all in range [-1000, 1000]
        img = np.random.randint(0, 1000, (10, 100, 100), dtype=np.int16)
        ndvi, evi, ndwi = calculate_indices(img)
        self.assertTrue(np.all(ndvi >= -1000) and np.all(ndvi <= 1000),
                        msg=f"test_calculate_indices fail: Expect all NDVI values to be in range [-1, 1] but got min {ndvi.min()} and max {ndvi.max()}")
        self.assertTrue(np.all(evi >= -1000) and np.all(evi <= 1000),
                        msg=f"test_calculate_indices fail: Expect all EVI values to be in range [-1, 1] but got min {evi.min()} and max {evi.max()}")
        self.assertTrue(np.all(ndwi >= -1000) and np.all(ndwi <= 1000),
                        msg=f"test_calculate_indices fail: Expect all NDWI values to be in range [-1, 1] but got min {ndwi.min()} and max {ndwi.max()}")

    @patch("download_sentinel2_mosaics.fs")
    @patch("download_sentinel2_mosaics.download_sentinel2_mosaic")
    @patch("download_sentinel2_mosaics.wait_for_task")
    @patch("download_sentinel2_mosaics.rasterio.open")
    @patch("download_sentinel2_mosaics.calculate_indices")
    @patch("download_sentinel2_mosaics.cloud_detector.get_cloud_masks")
    def test_retrieve_time_series_stack(
        self,
        mock_cloud_masks,
        mock_calculate_indices,
        mock_rasterio_open,
        mock_wait_for_task,
        mock_download,
        mock_fs,
    ):
        '''
        Test that retrieve_time_series_stack returns correctly shaped
        arrays when each time window has an image returned from Google 
        Earth Engine.
        '''
        # Setup mocks
        mock_fs.exists.return_value = False

        # Simulate successful EE task
        mock_task = MagicMock()
        mock_download.return_value = (mock_task, "dummy_prefix")
        mock_wait_for_task.return_value = {"state": "COMPLETED"}

        # Simulate image read
        dummy_img = np.ones((10, 100, 100), dtype=np.int16)
        mock_src = MagicMock()
        mock_src.read.return_value = dummy_img
        mock_rasterio_open.return_value.__enter__.return_value = mock_src

        # Indices
        mock_calculate_indices.return_value = (
            np.ones((100, 100)),  # NDVI
            np.ones((100, 100)),  # EVI
            np.ones((100, 100))   # NDWI
        )

        # Cloud mask: all clear
        mock_cloud_masks.return_value = np.zeros((100, 100), dtype=bool)

        # Call function
        stack, meta = retrieve_time_series_stack("site123", 30.0, 70.0, date(2021, 1, 5))

        # Ensure correct sizes
        self.assertEqual(len(stack), 37,
                         msg=f"test_retrieve_time_series_stack fail: Expect stack_list to be size 37, but got {len(stack)}")
        self.assertEqual(stack[0].shape, (13, 100, 100),
                         msg=f"test_retrieve_time_series_stack fail: Expect first image in stack_list to be size (13, 100, 100) but got {stack[0].shape}")
        self.assertEqual(len(meta), 37,
                         msg=f"test_retrieve_time_series_stack fail: Expect meta_list to be size 37, but got {len(meta)}")

        # Ensure mean_ndvi, mean_evi, mean_ndwi are all not NO_DATA, that
        # cloud_fraction is 0.0, and that the date ranges are correct
        windows = get_dense_time_windows(date(2021, 1, 5))
        for i in range(len(meta)):
            item = meta[i]
            wdws  = [d.strftime('%Y-%m-%d') for d in windows[i]]
            self.assertEqual(item['date_range'], wdws,
                             msg=f"test_retrieve_time_series_stack fail: Expect date range {wdws} but got {item['date_range']}")
            self.assertEqual(item['cloud_fraction'], 0.0,
                            msg=f"test_retrieve_time_series_stack fail: Expect cloud fraction to be 0.0 but got {item['cloud_fraction']}")
            self.assertEqual(item['mean_ndvi'], 1.0,
                            msg=f"test_retrieve_time_series_stack fail: Expect mean_ndvi to be 1.0 but got {item['mean_ndvi']}")
            self.assertEqual(item['mean_evi'], 1.0,
                            msg=f"test_retrieve_time_series_stack fail: Expect mean_evi to be 1.0 but got {item['mean_evi']}")
            self.assertEqual(item['mean_ndwi'], 1.0,
                            msg=f"test_retrieve_time_series_stack fail: Expect mean_ndwi to be 1.0, but got {item['mean_ndwi']}")

        # Ensure all values in stack are NOT NO_DATA
        for img in stack:
            self.assertTrue(np.all(img != NO_DATA),
                            msg=f"test_retrieve_time_series_stack fail: Expect image to not contain {NO_DATA}")

    @patch("download_sentinel2_mosaics.fs")
    @patch("download_sentinel2_mosaics.download_sentinel2_mosaic")
    @patch("download_sentinel2_mosaics.calculate_indices")
    def test_retrieve_time_series_stack_no_images(s
        self,
        mock_calculate_indices,
        mock_download,
        mock_fs,
    ):
        '''
        Test that retrieve_time_series_stack returns correctly shaped
        arrays when each time window does not have any images returned 
        from Google Earth Engine.
        '''
        # Setup mocks
        mock_fs.exists.return_value = False

        # Simulate successful EE task
        mock_task = MagicMock()
        mock_download.return_value = (None, "dummy_prefix")

        # Indices
        mock_calculate_indices.return_value = (
            np.ones((100, 100)),  # NDVI
            np.ones((100, 100)),  # EVI
            np.ones((100, 100))   # NDWI
        )

        stack, meta = retrieve_time_series_stack("site123", 30.0, 70.0, date(2021, 1, 5))

        # Each image in stack_list should be -9999
        for s in stack:
             self.assertTrue(np.all(s == NO_DATA),
                             msg=f"test_retrieve_time_series_stack_no_images fail: Expect every pixel in stack_list to be {NO_DATA}")

        # Each object in meta should have cloud_fraction 1.0, mean_ndvi NO_DATA, and mean_ndwi NO_DATA
        for m in meta:
            self.assertEqual(m['cloud_fraction'], 1.0,
                             msg=f"test_retrieve_time_series_stack_no_images fail: Expect cloud_fraction 1.0, but got cloud_fraction {m['cloud_fraction']}")
            self.assertEqual(m['mean_ndvi'], NO_DATA, 
                             msg=f"test_retrieve_time_series_stack_no_images fail: Expect mean_ndvi {NO_DATA}, but got {m['mean_ndvi']}")
            self.assertEqual(m['mean_evi'], NO_DATA, 
                             msg=f"test_retrieve_time_series_stack_no_images fail: Expect mean_evi {NO_DATA}, but got {m['mean_evi']}")
            self.assertEqual(m['mean_ndwi'], NO_DATA, 
                             msg=f"test_retrieve_time_series_stack_no_images fail: Expect mean_ndwi {NO_DATA}, but got {m['mean_ndwi']}")