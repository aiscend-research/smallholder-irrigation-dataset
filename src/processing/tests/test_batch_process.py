"""
Tests for batch_process.py, particularly the generate_latest_irrigation_data function.

This test file demonstrates how to test complex data processing logic using mock data.
"""

import unittest
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import patch
from batch_process import generate_latest_irrigation_data


class TestGenerateLatestIrrigationData(unittest.TestCase):
    """
    Tests for the generate_latest_irrigation_data function.

    This function identifies the most recent version of each survey based on
    a priority system for file naming conventions.
    """

    def setUp(self):
        """
        Runs BEFORE each test.

        We'll create a temporary directory with mock CSV files to test against.
        This way our tests don't depend on real data that might change.
        """
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.group_name = "test_sample"
        self.merged_folder = os.path.join(self.test_dir, f"labels/labeled_surveys/{self.group_name}/merged")
        os.makedirs(self.merged_folder, exist_ok=True)

        print(f"\n📁 Created test directory: {self.test_dir}")

    def tearDown(self):
        """
        Runs AFTER each test.

        Clean up the temporary directory so we don't leave files around.
        """
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"🧹 Cleaned up test directory")

    @patch('batch_process.get_data_root')
    def test_special_case_manual_inclusion(self, mock_get_data_root):
        """
        Test that AB_JL_101-125 and PS_101-125 are ALWAYS included.

        These don't follow the standard naming pattern but should be manually
        included as most_recent=1 regardless of other versions.

        This is a regression test - if someone removes the special case logic,
        this test will fail.
        """
        mock_get_data_root.return_value = self.test_dir

        # ARRANGE: Create surveys including the special cases
        test_data = pd.DataFrame({
            'plot_file': ['Zambia_101-125.csv', 'Zambia_101-125.csv', 'Zambia_200-225.csv'],
            'source_file': ['AB_JL_101-125', 'JL_AB_101-125', 'PS_101-125'],
            'site_id': ['id_101', 'id_101', 'id_200'],
            'irrigation': [1, 1, 1]
        })

        test_csv_path = os.path.join(self.merged_folder, 'special_cases.csv')
        test_data.to_csv(test_csv_path, index=False)

        # ACT
        result = generate_latest_irrigation_data(group_name=self.group_name)

        # ASSERT: Both special cases should be included
        # AB_JL_101-125 should be included even though JL_AB_101-125 exists (higher priority normally)
        self.assertIn(
            'AB_JL_101-125',
            result['source_file'].values,
            msg="AB_JL_101-125 should be manually included as special case"
        )

        self.assertIn(
            'PS_101-125',
            result['source_file'].values,
            msg="PS_101-125 should be manually included as special case"
        )

        # Both should have most_recent=1
        ab_jl_row = result[result['source_file'] == 'AB_JL_101-125']
        ps_row = result[result['source_file'] == 'PS_101-125']

        self.assertEqual(ab_jl_row.iloc[0]['most_recent'], 1)
        self.assertEqual(ps_row.iloc[0]['most_recent'], 1)

        print(f"✅ Special cases correctly included: {result['source_file'].tolist()}")

    @patch('batch_process.get_data_root')
    def test_special_case_exclusion_mv_76_100(self, mock_get_data_root):
        """
        Test that MV_76-100 is ALWAYS excluded.

        This survey is corrupted and should never appear in results,
        even if it's the only version available.
        """
        mock_get_data_root.return_value = self.test_dir

        # ARRANGE: Create surveys including the excluded one
        test_data = pd.DataFrame({
            'plot_file': ['Zambia_76-100.csv', 'Zambia_76-100.csv', 'Zambia_1-25.csv'],
            'source_file': ['MV_76-100', 'JL_MV_76-100', 'AB_1-25'],
            'site_id': ['id_76', 'id_76', 'id_1'],
            'irrigation': [1, 1, 1]
        })

        test_csv_path = os.path.join(self.merged_folder, 'exclusion_test.csv')
        test_data.to_csv(test_csv_path, index=False)

        # ACT
        result = generate_latest_irrigation_data(group_name=self.group_name)

        # ASSERT: No MV_76-100 should appear in results
        mv_surveys = result[result['source_file'].str.contains('MV_76-100', na=False)]

        self.assertEqual(
            len(mv_surveys),
            0,
            msg="MV_76-100 should be excluded (corrupted survey)"
        )

        # AB_1-25 should still be there
        self.assertIn(
            'AB_1-25',
            result['source_file'].values,
            msg="Other surveys should not be affected by MV_76-100 exclusion"
        )

        print(f"✅ MV_76-100 correctly excluded. Remaining: {result['source_file'].tolist()}")

    @patch('batch_process.get_data_root')
    def test_priority_corrected_v2_beats_all(self, mock_get_data_root):
        """
        Test priority level 1: corrected_v2 beats everything else.

        Scenario: Same plot_file has 4 versions:
        - AB_1-25 (original) - Priority 4 (lowest)
        - AB_v2_1-25 (uncorrected v2) - Priority 3
        - JL_AB_1-25 (corrected) - Priority 2
        - JL_AB_v2_1-25 (corrected v2) - Priority 1 (highest) ← Should WIN

        This documents the expected priority order.
        """
        mock_get_data_root.return_value = self.test_dir

        # ARRANGE: Create all 4 priority levels for the same plot_file
        test_data = pd.DataFrame({
            'plot_file': ['Zambia_1-25.csv'] * 4,
            'source_file': ['AB_1-25', 'AB_v2_1-25', 'JL_AB_1-25', 'JL_AB_v2_1-25'],
            'site_id': ['id_1'] * 4,
            'irrigation': [1, 1, 1, 1]
        })

        test_csv_path = os.path.join(self.merged_folder, 'priority_test.csv')
        test_data.to_csv(test_csv_path, index=False)

        # ACT
        result = generate_latest_irrigation_data(group_name=self.group_name)

        # ASSERT: Only 1 row should be returned (the highest priority)
        self.assertEqual(
            len(result),
            1,
            msg="Should return only 1 row when multiple versions exist"
        )

        # The corrected_v2 version should win
        self.assertEqual(
            result.iloc[0]['source_file'],
            'JL_AB_v2_1-25',
            msg="Corrected v2 (JL_AB_v2_1-25) should beat all other versions"
        )

        print(f"✅ Priority test passed! Winner: {result.iloc[0]['source_file']}")

    @patch('batch_process.get_data_root')
    def test_priority_corrected_beats_uncorrected_v2_and_original(self, mock_get_data_root):
        """
        Test priority level 2: corrected beats uncorrected_v2 and original.

        Scenario: Same plot_file has 3 versions (no corrected_v2):
        - AB_1-25 (original) - Priority 4
        - AB_v2_1-25 (uncorrected v2) - Priority 3
        - JL_AB_1-25 (corrected) - Priority 2 ← Should WIN
        """
        mock_get_data_root.return_value = self.test_dir

        test_data = pd.DataFrame({
            'plot_file': ['Zambia_26-50.csv'] * 3,
            'source_file': ['AB_26-50', 'AB_v2_26-50', 'JL_AB_26-50'],
            'site_id': ['id_26'] * 3,
            'irrigation': [1, 1, 1]
        })

        test_csv_path = os.path.join(self.merged_folder, 'corrected_test.csv')
        test_data.to_csv(test_csv_path, index=False)

        # ACT
        result = generate_latest_irrigation_data(group_name=self.group_name)

        # ASSERT
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result.iloc[0]['source_file'],
            'JL_AB_26-50',
            msg="Corrected version should beat uncorrected_v2 and original"
        )

        print(f"✅ Corrected version won: {result.iloc[0]['source_file']}")

    @patch('batch_process.get_data_root')
    def test_priority_uncorrected_v2_beats_original(self, mock_get_data_root):
        """
        Test priority level 3: uncorrected_v2 beats original.

        Scenario: Same plot_file has 2 versions (no corrections):
        - AB_51-75 (original) - Priority 4
        - AB_v2_51-75 (uncorrected v2) - Priority 3 ← Should WIN
        """
        mock_get_data_root.return_value = self.test_dir

        test_data = pd.DataFrame({
            'plot_file': ['Zambia_51-75.csv'] * 2,
            'source_file': ['AB_51-75', 'AB_v2_51-75'],
            'site_id': ['id_51'] * 2,
            'irrigation': [1, 1]
        })

        test_csv_path = os.path.join(self.merged_folder, 'v2_test.csv')
        test_data.to_csv(test_csv_path, index=False)

        # ACT
        result = generate_latest_irrigation_data(group_name=self.group_name)

        # ASSERT
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result.iloc[0]['source_file'],
            'AB_v2_51-75',
            msg="Uncorrected v2 version should beat original"
        )

        print(f"✅ Uncorrected v2 won: {result.iloc[0]['source_file']}")

    @patch('batch_process.get_data_root')
    def test_multiple_plot_files_independent_selection(self, mock_get_data_root):
        """
        Test that each plot_file gets its own most recent version selected independently.

        Scenario: 3 different plot_files, each with different versions:
        - Plot 1: Has corrected_v2 → Should select corrected_v2
        - Plot 2: Has only corrected → Should select corrected
        - Plot 3: Has only original → Should select original

        This ensures the priority logic doesn't interfere across different surveys.
        """
        mock_get_data_root.return_value = self.test_dir

        test_data = pd.DataFrame({
            'plot_file': [
                'Zambia_1-25.csv', 'Zambia_1-25.csv',  # Plot 1: 2 versions
                'Zambia_26-50.csv', 'Zambia_26-50.csv',  # Plot 2: 2 versions
                'Zambia_51-75.csv'  # Plot 3: 1 version
            ],
            'source_file': [
                'AB_1-25', 'JL_AB_v2_1-25',  # Plot 1: original + corrected_v2
                'CD_26-50', 'EF_CD_26-50',  # Plot 2: original + corrected
                'GH_51-75'  # Plot 3: just original
            ],
            'site_id': ['id_1', 'id_1', 'id_26', 'id_26', 'id_51'],
            'irrigation': [1, 1, 1, 1, 1]
        })

        test_csv_path = os.path.join(self.merged_folder, 'multi_plot_test.csv')
        test_data.to_csv(test_csv_path, index=False)

        # ACT
        result = generate_latest_irrigation_data(group_name=self.group_name)

        # ASSERT: Should have 3 rows (one per plot_file)
        self.assertEqual(
            len(result),
            3,
            msg="Should have 1 selected version per plot_file"
        )

        # Check each plot_file got the correct version
        plot1 = result[result['plot_file'] == 'Zambia_1-25.csv']
        plot2 = result[result['plot_file'] == 'Zambia_26-50.csv']
        plot3 = result[result['plot_file'] == 'Zambia_51-75.csv']

        self.assertEqual(plot1.iloc[0]['source_file'], 'JL_AB_v2_1-25',
                        msg="Plot 1 should select corrected_v2")
        self.assertEqual(plot2.iloc[0]['source_file'], 'EF_CD_26-50',
                        msg="Plot 2 should select corrected")
        self.assertEqual(plot3.iloc[0]['source_file'], 'GH_51-75',
                        msg="Plot 3 should select original (only option)")

        print(f"✅ Independent selection works! Selected: {result['source_file'].tolist()}")

    @patch('batch_process.get_data_root')
    def test_edge_case_empty_merged_folder(self, mock_get_data_root):
        """
        Test edge case: What happens when merged folder is empty?

        Expected behavior: Should raise an error or return empty DataFrame.
        This tests defensive programming - handling missing data gracefully.
        """
        mock_get_data_root.return_value = self.test_dir

        # ARRANGE: merged folder exists but has no CSV files
        # (setUp already created the empty folder)

        # ACT & ASSERT: Should handle empty folder gracefully
        with self.assertRaises(ValueError) as context:
            result = generate_latest_irrigation_data(group_name=self.group_name)

        print(f"✅ Empty folder handled: {str(context.exception)}")

    @patch('batch_process.get_data_root')
    def test_edge_case_no_csv_files_only_other_files(self, mock_get_data_root):
        """
        Test edge case: merged folder has files but no CSVs.

        Scenario: Folder has .txt, .geojson files but no .csv files
        Expected: Should handle this gracefully (probably raise error)
        """
        mock_get_data_root.return_value = self.test_dir

        # ARRANGE: Create non-CSV files
        txt_path = os.path.join(self.merged_folder, 'notes.txt')
        with open(txt_path, 'w') as f:
            f.write("Some notes")

        geojson_path = os.path.join(self.merged_folder, 'polygons.geojson')
        with open(geojson_path, 'w') as f:
            f.write('{"type": "FeatureCollection", "features": []}')

        # ACT & ASSERT
        with self.assertRaises(ValueError):
            result = generate_latest_irrigation_data(group_name=self.group_name)

        print("✅ Non-CSV files ignored correctly")

    @patch('batch_process.get_data_root')
    def test_edge_case_malformed_source_file_names(self, mock_get_data_root):
        """
        Test edge case: source_file doesn't match any naming pattern.

        Scenario: Someone names a file incorrectly (lowercase, typo, etc.)
        - "ab_1-25" (lowercase)
        - "Survey_1-25" (weird name)
        - "123_456" (all numbers)

        Expected: These should still be processed, just with lowest priority
        """
        mock_get_data_root.return_value = self.test_dir

        test_data = pd.DataFrame({
            'plot_file': ['Zambia_1-25.csv'] * 4,
            'source_file': [
                'ab_1-25',  # lowercase - won't match regex
                'Survey_1-25',  # weird name
                '123_456',  # all numbers
                'AB_1-25'  # proper name (should win as fallback)
            ],
            'site_id': ['id_1'] * 4,
            'irrigation': [1, 1, 1, 1]
        })

        test_csv_path = os.path.join(self.merged_folder, 'malformed_test.csv')
        test_data.to_csv(test_csv_path, index=False)

        # ACT
        result = generate_latest_irrigation_data(group_name=self.group_name)

        # ASSERT: Should select AB_1-25 (the only one matching the pattern)
        # The malformed names fall through to the else clause (source_files)
        self.assertIn(
            result.iloc[0]['source_file'],
            ['AB_1-25', 'ab_1-25', 'Survey_1-25', '123_456'],
            msg="Should select one of the available versions (fallback to last else)"
        )

        print(f"✅ Handled malformed names. Selected: {result.iloc[0]['source_file']}")

    @patch('batch_process.get_data_root')
    def test_edge_case_three_letter_initials(self, mock_get_data_root):
        """
        Test edge case: Do 3-letter initials work with the regex?

        Scenario: DSB_JL_v2_26-50 (3 letters + 2 letters + v2)
        Expected: Should match corrected_v2 pattern

        This tests that r'^[A-Z]+_[A-Z]+_v2' handles variable lengths.
        """
        mock_get_data_root.return_value = self.test_dir

        test_data = pd.DataFrame({
            'plot_file': ['Zambia_26-50.csv'] * 2,
            'source_file': [
                'AB_26-50',  # 2 letter initials (original)
                'DSB_JL_v2_26-50'  # 3+2 letter initials (corrected_v2)
            ],
            'site_id': ['id_26'] * 2,
            'irrigation': [1, 1]
        })

        test_csv_path = os.path.join(self.merged_folder, 'three_letter_test.csv')
        test_data.to_csv(test_csv_path, index=False)

        # ACT
        result = generate_latest_irrigation_data(group_name=self.group_name)

        # ASSERT: corrected_v2 with 3-letter initials should win
        self.assertEqual(
            result.iloc[0]['source_file'],
            'DSB_JL_v2_26-50',
            msg="Regex should handle 3-letter initials (DSB_JL_v2)"
        )

        print(f"✅ 3-letter initials work: {result.iloc[0]['source_file']}")

    @patch('batch_process.get_data_root')
    def test_edge_case_multiple_csv_files(self, mock_get_data_root):
        """
        Test that function correctly merges multiple CSV files.

        Scenario: merged folder has 3 separate CSV files, each with different surveys.
        Expected: Should read and concatenate all CSV files, then select most recent.
        """
        mock_get_data_root.return_value = self.test_dir

        # ARRANGE: Create 3 separate CSV files
        csv1 = pd.DataFrame({
            'plot_file': ['Zambia_1-25.csv'],
            'source_file': ['AB_1-25'],
            'site_id': ['id_1'],
            'irrigation': [1]
        })
        csv2 = pd.DataFrame({
            'plot_file': ['Zambia_26-50.csv'],
            'source_file': ['CD_26-50'],
            'site_id': ['id_26'],
            'irrigation': [1]
        })
        csv3 = pd.DataFrame({
            'plot_file': ['Zambia_51-75.csv'],
            'source_file': ['EF_51-75'],
            'site_id': ['id_51'],
            'irrigation': [1]
        })

        csv1.to_csv(os.path.join(self.merged_folder, 'AB_1-25.csv'), index=False)
        csv2.to_csv(os.path.join(self.merged_folder, 'CD_26-50.csv'), index=False)
        csv3.to_csv(os.path.join(self.merged_folder, 'EF_51-75.csv'), index=False)

        # ACT
        result = generate_latest_irrigation_data(group_name=self.group_name)

        # ASSERT: Should have all 3 surveys
        self.assertEqual(
            len(result),
            3,
            msg="Should merge all 3 CSV files"
        )

        self.assertEqual(
            set(result['source_file'].values),
            {'AB_1-25', 'CD_26-50', 'EF_51-75'},
            msg="Should include surveys from all CSV files"
        )

        print(f"✅ Multiple CSV files merged correctly: {result['source_file'].tolist()}")

    @patch('batch_process.get_data_root')
    def test_edge_case_special_case_with_competing_higher_priority(self, mock_get_data_root):
        """
        Test edge case: Special case manual inclusion even when higher priority exists.

        Scenario: AB_JL_101-125 exists alongside JL_AB_v2_101-125
        Normally JL_AB_v2 would win (corrected_v2 > no pattern match)
        But AB_JL_101-125 is a special case and MUST be included.

        This tests that the manual override happens AFTER priority selection.
        """
        mock_get_data_root.return_value = self.test_dir

        test_data = pd.DataFrame({
            'plot_file': ['Zambia_101-125.csv'] * 2,
            'source_file': ['AB_JL_101-125', 'JL_AB_v2_101-125'],
            'site_id': ['id_101'] * 2,
            'irrigation': [1, 1]
        })

        test_csv_path = os.path.join(self.merged_folder, 'special_override_test.csv')
        test_data.to_csv(test_csv_path, index=False)

        # ACT
        result = generate_latest_irrigation_data(group_name=self.group_name)

        # ASSERT: BOTH should be included (AB_JL_101-125 is manually set to most_recent=1)
        # This is different from normal priority where only 1 would be selected
        ab_jl_included = 'AB_JL_101-125' in result['source_file'].values

        self.assertTrue(
            ab_jl_included,
            msg="AB_JL_101-125 should be included (special case override)"
        )

        # Check that AB_JL_101-125 has most_recent=1
        if ab_jl_included:
            ab_jl_row = result[result['source_file'] == 'AB_JL_101-125']
            self.assertEqual(
                ab_jl_row.iloc[0]['most_recent'],
                1,
                msg="AB_JL_101-125 should have most_recent=1 from manual override"
            )

        print(f"✅ Special case override works. Included: {result['source_file'].tolist()}")


if __name__ == '__main__':
    unittest.main()
