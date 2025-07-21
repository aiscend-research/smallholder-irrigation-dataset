"""
Contains tests for helper functions for creating label band
"""

import unittest
from datetime import datetime, timedelta
from create_label_band import create_irrigation_table, get_survey_data, retrieve_polygons, rasterize_polygons, save_label_raster, create_labels, create_bounding_box
from utils.utils import get_data_root
import rasterio
import os

# Tests for helper functions in create_label_band.py: get_survey_data, 
# retrieve_polygons, rasterize_polygons, save_label_raster
class TestLabelBandFunctions(unittest.TestCase):
    def setUp(self):
        self.IRRIGATION_TABLE = create_irrigation_table()

    # test for get_survey_data()
    def test_get_survey_data(self):
        """
        Test the get_survey_data function to ensure it correctly extracts latitude, longitude,
        and survey date from the filename with the correct format.
        """

        img_path = "s2_-10.4035_29.1319_2023-05-20_2023-05-30_off-15.tif"
        (lat, lon, survey_date) = get_survey_data(img_path)

        self.assertEqual(
            "-10.4035", 
            lat,
            msg=f"test_get_survey_data fail: Expecteed lat -10.4035 but got {lat}"
        )
        self.assertEqual(
            "29.1319", 
            lon,
            msg=f"test_get_survey_data fail: Expecteed lon 29.1319 but got {lon}"
        )
        self.assertEqual(
            datetime.strptime("2023-05-25", "%Y-%m-%d").date() + timedelta(15), 
            survey_date,
            msg=f"test_get_survey_data fail: Expected survey date 2023-05-25 but got {survey_date}"
        )
        img_path = "s2_-12.0468_26.3789_2020-07-31_2020-08-10_off-30.tif"
        (lat, lon, survey_date) = get_survey_data(img_path)

        self.assertEqual(
            "-12.0468",
            lat,
            msg=f"test_get_survey_data fail: Expecteed lat -10.4035 but got {lat}"
        )
        self.assertEqual(
            "26.3789",
            lon,
            msg=f"test_get_survey_data fail: Expecteed lon 29.1319 but got {lon}"
        )
        self.assertEqual(
            datetime.strptime("2020-08-05", "%Y-%m-%d").date() + timedelta(30),
            survey_date,
            msg=f"test_get_survey_data fail: Expected survey date 2020-08-05 but got {survey_date}"
        )

    # tests for retrieve_polygons()
    def test_retrieve_polygons_invalid_file(self):
        """
        Test the retrieve_polygons function to ensure it raises an error when the irrigation
        geojson file does not exist.
        """

        irrigation_geojson = "non_existent_file.geojson"
        survey_id = 5043172
        internal_id = 23
        timestamp = datetime.strptime("2023-05-25", "%Y-%m-%d").date() + timedelta(15)
        image_meta = {
            'crs': 'EPSG:4326'
        }

        with self.assertRaises(RuntimeError):
            retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)

    def test_retrieve_polygons_no_polygons(self):
        """
        Test the retrieve_polygons function to ensure it returns an empty GeoDataFrame
        when no polygons match the criteria.
        """

        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/JL_KL_v2_101-125.geojson"
        survey_id = 5062566
        internal_id = 21
        timestamp = datetime.strptime("2019-06-11", "%Y-%m-%d").date()
        image_meta = {
            'crs': 'EPSG:4326'
        }
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        self.assertTrue(gdf.empty, msg="test_retrieve_polygons_no_polygons fail: Expected empty GeoDataFrame but got non-empty")
    
    def test_retrieve_polygons_single_polygon(self):
        """
        Test the retrieve_polygons function to ensure it correctly retrieves polygons
        from the source file, when there is only one corresponding polygon.
        """

        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/MV_950-974.geojson"
        survey_id = 3504043
        internal_id = 21
        timestamp = datetime.strptime("2020-8-27", "%Y-%m-%d").date()
        image_meta = {
            'crs': 'EPSG:4326'
        }

        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)

        self.assertEqual(
            1,
            len(gdf),
            msg=f"test_retrieve_polygons fail: Expected 1 polygon but got {len(gdf)}"
        )
        self.assertEqual(
            2020, gdf.iloc[0].year, msg=f"test_retrieve_polygons fail: Expected year 2020 but got {gdf['year'].values[0]}"
        )
        self.assertEqual(
            5, gdf.iloc[0].certainty, msg=f"test_retrieve_polygons fail: Expected certainty 5 but got {gdf['certainty'].values[0]}"
        )
        self.assertEqual(
            8, gdf.iloc[0].month, msg=f"test_retrieve_polygons fail: Expected month 8 but got {gdf['month'].values[0]}"
        )
        self.assertEqual(
            27, gdf.iloc[0].day, msg=f"test_retrieve_polygons fail: Expected day 27 but got {gdf['day'].values[0]}"
        )

    def test_retrieve_polygons_multiple_polygons(self):
        """
        Test the retrieve_polygons function to ensure it correctly retrieves polygons
        from the source file, when there are multiple corresponding polygons.
        """

        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/JL_KL_v2_101-125.geojson"
        survey_id = 5062566
        internal_id = 21
        timestamp = datetime.strptime("2019-06-14", "%Y-%m-%d").date()
        image_meta = {
            'crs': 'EPSG:4326'
        }

        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)

        self.assertEqual(
            4,
            len(gdf),
            msg=f"test_retrieve_polygons fail: Expected 4 polygons but got {len(gdf)}"
        )

        for i in range(3):
            # All polygons correct date
            self.assertEqual(
                2019, gdf.iloc[0].year, msg=f"test_retrieve_polygons fail: Expected year 2023 but got {gdf['year'].values[0]}"
            )
            self.assertEqual(
                6, gdf.iloc[i].month, msg=f"test_retrieve_polygons fail: Expected month 6 but got {gdf['month'].values[0]}"
            )
            self.assertEqual(
                14, gdf.iloc[i].day, msg=f"test_retrieve_polygons fail: Expected day 9 but got {gdf['day'].values[0]}"
            )

    # tests for rasterize_polygons()
    def test_rasterize_polygons_with_polygons(self):
        """
        Test the rasterize_polygons function to ensure it correctly rasterizes polygons
        into a band, when there are polygons within the bounding box.
        """
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/JL_KL_v2_101-125.geojson"
        survey_id = 5062566
        internal_id = 21
        timestamp = datetime.strptime("2019-06-14", "%Y-%m-%d").date()

        center_lon = 28.86255649573207
        center_lat = -11.15545729747778

        image_meta = create_bounding_box(center_lat, center_lon)

        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        band = rasterize_polygons(gdf, image_meta, certainty_thresh=3)

        self.assertEqual(
            (8, image_meta['height'], image_meta['width']),
            band.shape,
            msg=f"test_rasterize_polygons fail: Expected band shape {(image_meta['height'], image_meta['width'])} but got {band.shape}"
        )

        # Check that polygons show up in the band (first and second bands)
        self.assertTrue(
            (band[0] != 0).any(),
            msg="test_rasterize_polygons fail: Expected band to have at least one pixel with value 1"
        )
        self.assertTrue(
            (band[1] == 1).any(),
            msg="test_rasterize_polygons fail: Expected band to have at least one pixel with value 1"
        )

        # Test another case
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/MV_950-974.geojson"
        survey_id = 5107007
        internal_id = 16
        timestamp = datetime.strptime("2020-9-3", "%Y-%m-%d").date()

        center_lon = 28.479152896241143
        center_lat = -13.02898847831871

        image_meta = create_bounding_box(center_lat, center_lon)
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        band = rasterize_polygons(gdf, image_meta)
        self.assertEqual(
            (8, image_meta['height'], image_meta['width']),
            band.shape,
            msg=f"test_rasterize_polygons fail: Expected band shape {(8, image_meta['height'], image_meta['width'])} but got {band.shape}"
        )

        self.assertIn(
            1,
            band[0],
            msg="test_rasterize_polygons fail: Expected categorical irrigation band to have at least one pixel with value 1"
        )
        self.assertIn(
            4, # lawn
            band[0],
            msg="test_rasterize_polygons fail: Expected categorical irrigation band to have at least one pixel with value 4"
        )
        self.assertIn(
            0,
            band[0],
            msg="test_rasterize_polygons fail: Expected categorical irrigation band to have at least one pixel with value 0"
        )
        self.assertIn(
            1,
            band[1],
            msg="test_rasterize_polygons fail: Expected binary irrigation band to have at least one pixel with value 1"
        )
        self.assertIn(
            0, 
            band[1],
            msg="test_rasterize_polygons fail: Expected binary irrigation band to have at least one pixel with value 0"
        )

    def test_rasterize_polygons_no_polygons(self):
        """
        Test the rasterize_polygons function to ensure it returns an irrigation band of zeros
        when no polygons are present.
        """
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/MV_950-974.geojson"
        survey_id = 5107007
        internal_id = 16
        timestamp = datetime.strptime("2016-9-9", "%Y-%m-%d").date()

        center_lon = 28.479152896241143
        center_lat = -13.02898847831871

        image_meta = create_bounding_box(center_lat, center_lon)
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        band = rasterize_polygons(gdf, image_meta)
        self.assertEqual(
            (8, image_meta['height'], image_meta['width']),
            band.shape,
            msg=f"test_rasterize_polygons fail: Expected band shape {(8, image_meta['height'], image_meta['width'])} but got {band.shape}"
        )

        self.assertTrue(
            (band[0] == 0).all(),
            msg="test_rasterize_polygons fail: Expected categorical irrigation to be all zeros when no polygons are present"
        )
        self.assertTrue(
            (band[1] == 0).all(),
            msg="test_rasterize_polygons fail: Expected binary irrigation to be all zeros when no polygons are present"
        )
    # test uncertainty bands are being created correctly
    def test_rasterize_polygons_correct_uncertainty_bands_polygon_with_no_uncertainty(self):
        """
        Test rasterize_polygons to ensure that it correctly creates uncertainty bands
        when there are no uncertainties in the polygon.
        """
        # survey has one polygon of certainty 5
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/MV_950-974.geojson"
        survey_id = 3504043
        internal_id = 21
        timestamp = datetime.strptime("2020-8-27", "%Y-%m-%d").date()

        center_lon = 31.434356858051235
        center_lat = -11.826157726704926
        image_meta = create_bounding_box(center_lat, center_lon)
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        band = rasterize_polygons(gdf, image_meta, certainty_thresh=3)

        self.assertEqual(
            (8, image_meta['height'], image_meta['width']),
            band.shape,
            msg=f"test_rasterize_polygons fail: Expected band shape {(image_meta['height'], image_meta['width'])} but got {band.shape}"
        )

        for i in range(2, 7):
            self.assertTrue(
                (band[i] == 0).all(),
                msg=f"test_rasterize_polygons fail: Expected uncertainty band {i+1} to be all zeros"
            )

    def test_rasterize_polygons_correct_uncertainty_bands_polygon_with_single_uncertainty(self):
        """
        Test rasterize_polygons to ensure that it correctly creates uncertainty bands
        when there is one type of uncertainty in the set of polygons.
        """
        # survey has one polygon of certainty 5
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/MV_950-974.geojson"
        survey_id = 5107007
        internal_id = 16
        timestamp = datetime.strptime("2020-8-21", "%Y-%m-%d").date()

        center_lon = 28.479152896241143
        center_lat = -13.02898847831871

        image_meta = create_bounding_box(center_lat, center_lon)
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        band = rasterize_polygons(gdf, image_meta, certainty_thresh=2)

        self.assertEqual(
            (8, image_meta['height'], image_meta['width']),
            band.shape,
            msg=f"test_rasterize_polygons fail: Expected band shape {(image_meta['height'], image_meta['width'])} but got {band.shape}"
        )

        # check that only fourth uncertainty band has non-zero vals
        #  (polygon has uncertainty flag "uneven")
        for i in range(2, 7):
            if i == 4:
                self.assertTrue(
                    (band[i] != 0).any(),
                    msg=f"test_rasterize_polygons fail: Expected uncertainty band {i} to have at least one pixel with value 1"
                )
            else:
                self.assertTrue(
                    (band[i] == 0).all(),
                    msg=f"test_rasterize_polygons fail: Expected uncertainty band {i} to be all zeros"
                )

        # check that fourth uncertainty band non-zero vals line up with
        # the irrigated pixels indicated by the first and second bands
        self.assertTrue(
            (band[0][band[3] == 1] != 0).all(),
            msg="test_rasterize_polygons fail: Expected non-zero uncertainty descriptor pixels to line up with non-zero values in categorical irrigation band"
        )
        self.assertTrue(
            (band[1][band[3] == 1] == 1).all(),
            msg="test_rasterize_polygons fail: Expected uncertainty descriptor pixels to line up with pixels with value 1 in binary irrigation band"
        ) 

    def test_rasterize_polygons_correct_uncertainty_bands_polygon_with_multiple_uncertainties(self):
        """
        Test rasterize_polygons to ensure that it correctly creates uncertainty bands
        when there are multiple types of uncertainty in the set of polygons.
        """
        # survey has one polygon of certainty 5
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/KL_DSB_v2_101-125.geojson"
        survey_id = 1045543
        internal_id = 24
        timestamp = datetime.strptime("2018-7-25", "%Y-%m-%d").date()

        center_lon = 31.611147224629907
        center_lat = -9.201089565713376

        image_meta = create_bounding_box(center_lat, center_lon)
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        band = rasterize_polygons(gdf, image_meta, certainty_thresh=1)

        self.assertEqual(
            (8, image_meta['height'], image_meta['width']),
            band.shape,
            msg=f"test_rasterize_polygons fail: Expected band shape {(image_meta['height'], image_meta['width'])} but got {band.shape}"
        )

        # check that 2nd, 4th, and 5th uncertainty bands have non-zero vals
        # (only slightly green, may naturally be green, may be a fishpond)
        for i in range(2, 7):
            if i in [2, 4, 5]:
                self.assertTrue(
                    (band[i] != 0).any(),
                    msg=f"test_rasterize_polygons fail: Expected uncertainty band {i} to have at least one pixel with value 1"
                )
            else:
                self.assertTrue(
                    (band[i] == 0).all(),
                    msg=f"test_rasterize_polygons fail: Expected uncertainty band {i} to be all zeros"
                )

        # check that 2nd, 4th, and 5th uncertainty bands non-zero vals line up with
        # the non-zero vals in first band
        self.assertTrue(
            (band[0][band[2] != 0] == 1).all(),
            msg="test_rasterize_polygons fail: Expected uncertainty band 1 to have same non-zero vals as irrigation band"
        )
        self.assertTrue(
            (band[0][band[4] != 0] == 1).all(),
            msg="test_rasterize_polygons fail: Expected uncertainty band 3 to have same non-zero vals as irrigation band"
        )
        self.assertTrue(
            (band[0][band[5] != 0] == 1).all(),
            msg="test_rasterize_polygons fail: Expected uncertainty band 4 to have same non-zero vals as irrigation band"
        )
        # This timestamp/location only has one polygon with 3 uncertainty categories, so
        # make sure that the reverse is true as well.
        self.assertTrue(
            (band[2][band[0] != 0] == 1).all(),
            msg="test_rasterize_polygons fail: Expected uncertainty band 1 to have same non-zero vals as irrigation band"
        )
        self.assertTrue(
            (band[4][band[0] != 0] == 1).all(),
            msg="test_rasterize_polygons fail: Expected uncertainty band 3 to have same non-zero vals as irrigation band"
        )
        self.assertTrue(
            (band[5][band[0] != 0] == 1).all(),
            msg="test_rasterize_polygons fail: Expected uncertainty band 4 to have same non-zero vals as irrigation band"
        )

    # test certainty score band is being created correctly
    def test_rasterize_polygons_certainty_score_band(self):
        """
        Test rasterize_polygons to ensure that it correctly creates a certainty band
        when there are polygons with varying certainty scores.
        """
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/MV_950-974.geojson"
        survey_id = 5107007
        internal_id = 16
        timestamp = datetime.strptime("2020-8-21", "%Y-%m-%d").date()

        center_lon = 28.479152896241143
        center_lat = -13.02898847831871

        image_meta = create_bounding_box(center_lat, center_lon)
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        band = rasterize_polygons(gdf, image_meta, certainty_thresh=1)

        self.assertEqual(
            (8, image_meta['height'], image_meta['width']),
            band.shape,
            msg=f"test_rasterize_polygons fail: Expected band shape {(image_meta['height'], image_meta['width'])} but got {band.shape}"
        )

        # Check that the certainty band is correct
        self.assertTrue(
            (band[7] == 4).any(),
            msg="test_rasterize_polygons fail: Expected certainty band to have at least one pixel with value 4"
        )
        self.assertTrue(
            (band[7] == 2).any(),
            msg="test_rasterize_polygons fail: Expected certainty band to have at least one pixel with value 2"
        )
        # check certainty band pixels match the certainty of the polygons
        self.assertTrue(
            (band[1][band[6] == 4] != 0).all() and (band[0][band[6] == 4] != 0).all(),
            msg="test_rasterize_polygons fail: Expected irrigation band to have non-zero value where certainty score is 4"
        )
        self.assertTrue(
            (band[1][band[6] == 2] != 0).all() and (band[0][band[6] == 2] != 0).all(), 
            msg="test_rasterize_polygons fail: Expected irrigation band to have non-zero value where certainty score is 2"
        )

        # check that where there are no polygons, certainty band is zero
        self.assertTrue(
            (band[7][band[0] == 0] == 0).all() and (band[7][band[0] == 0] == 0).all(),
            msg="test_rasterize_polygons fail: Expect certainty score to be 0 where there are no polygons"
        )

    def test_rasterize_polygons_certainty_band_no_polygons(self):
        """
        Test rasterize_polygons to ensure that it correctly creates a certainty band
        when there are no polygons.
        """
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/MV_950-974.geojson"
        survey_id = 5107007
        internal_id = 16
        timestamp = datetime.strptime("2016-9-9", "%Y-%m-%d").date()

        center_lon = 28.479152896241143
        center_lat = -13.02898847831871

        image_meta = create_bounding_box(center_lat, center_lon)
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        band = rasterize_polygons(gdf, image_meta)

        self.assertEqual(
            (8, image_meta['height'], image_meta['width']),
            band.shape,
            msg=f"test_rasterize_polygons fail: Expected band shape {(image_meta['height'], image_meta['width'])} but got {band.shape}"
        )

        # Check that the certainty band is all zeros
        self.assertTrue(
            (band[7] == 0).all(),
            msg="test_rasterize_polygons fail: Expected certainty band to be all zeros when no polygons are present"
        )

    # tests for save_label_raster() to ensure data is saved correctly
    def test_save_label_raster_with_irrigation(self):
        """
        Test save_label_raster to ensure that it correctly saves the label raster
        when there is irrigation in it.
        """
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/JL_KL_v2_101-125.geojson"
        survey_id = 5062566
        internal_id = 21
        timestamp = datetime.strptime("2019-06-14", "%Y-%m-%d").date()

        center_lon = 28.86255649573207
        center_lat = -11.15545729747778

        image_meta = create_bounding_box(center_lat, center_lon)
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        band = rasterize_polygons(gdf, image_meta, certainty_thresh=3)

        output_path = "test_label_band.tif"
        save_label_raster(band, image_meta, output_path)

        full_output_path = get_data_root() + "/" + output_path
        full_metadata_path = get_data_root() + "/test_label_band_metadata.json"

        with rasterio.open(full_output_path) as src:
            self.assertEqual(
                src.meta['count'],
                8,
                msg=f"test_save_label_raster fail: Expected 6 bands but got {src.count}"
            )
            self.assertEqual(
                src.meta['width'],
                image_meta['width'],
                msg=f"test_save_label_raster fail: Expected width {image_meta['width']} but got {src.width}"
            )
            self.assertEqual(
                src.meta['height'],
                image_meta['height'],
                msg=f"test_save_label_raster fail: Expected height {image_meta['height']} but got {src.height}"
            )
            self.assertEqual(
                src.meta['crs'],
                image_meta['crs'],
                msg=f"test_save_label_raster fail: Expected CRS {image_meta['crs']} but got {src.crs}"
            )
            self.assertEqual(
                band.tolist(), 
                src.read().tolist(),
                msg="test_save_label_raster fail: Expected band to match saved raster band"
            )
        
        # Clean up
        if os.path.exists(full_output_path):
            os.remove(full_output_path)

        if os.path.exists(full_metadata_path):
            os.remove(full_metadata_path)

    def test_save_label_raster_no_irrigation(self):
        """
        Test save_label_raster to ensure that it correctly saves the label raster
        when there is no irrigation in it.
        """
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/JL_KL_v2_101-125.geojson"
        survey_id = 5062566
        internal_id = 21
        timestamp = datetime.strptime("2019-06-19", "%Y-%m-%d").date()

        center_lon = 28.86255649573207
        center_lat = -11.15545729747778

        image_meta = create_bounding_box(center_lat, center_lon)
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        band = rasterize_polygons(gdf, image_meta, certainty_thresh=3)

        output_path = "test_label_band.tif"
        save_label_raster(band, image_meta, output_path)

        full_output_path = get_data_root() + "/" + output_path
        full_metadata_path = get_data_root() + "/test_label_band_metadata.json"

        with rasterio.open(full_output_path) as src:
            self.assertEqual(
                src.meta['count'],
                8,
                msg=f"test_save_label_raster fail: Expected 6 bands but got {src.count}"
            )
            self.assertEqual(
                src.meta['width'],
                image_meta['width'],
                msg=f"test_save_label_raster fail: Expected width {image_meta['width']} but got {src.width}"
            )
            self.assertEqual(
                src.meta['height'],
                image_meta['height'],
                msg=f"test_save_label_raster fail: Expected height {image_meta['height']} but got {src.height}"
            )
            self.assertEqual(
                src.meta['crs'],
                image_meta['crs'],
                msg=f"test_save_label_raster fail: Expected CRS {image_meta['crs']} but got {src.crs}"
            )
            self.assertEqual(
                band.tolist(), 
                src.read().tolist(),
                msg="test_save_label_raster fail: Expected band to match saved raster band"
            )
        
        # Clean up
        if os.path.exists(full_output_path):
            os.remove(full_output_path)

        if os.path.exists(full_metadata_path):
            os.remove(full_metadata_path)