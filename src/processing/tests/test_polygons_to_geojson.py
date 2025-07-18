"""
Contains tests for match_uncertainty_categories, get_special_categories
to ensure robustness within both methods.
"""

import unittest
from polygons_to_geojson import match_uncertainty_categories, get_special_categories, parse_description

"""
Tests for uncertainty category matching (match_uncertainty_categories)
"""
class TestUncertaintyCategoryMatch(unittest.TestCase):

    def setUp(self):
        self.UNCERTAINTY_CATEGORIES = [
            "unclear signs of agriculture",
            "only slightly green",
            "uneven",
            "may naturally be green",
            "may be a fishpond"
        ]

    def test_single_exact_match(self):
        text = "4 may naturally be green"
        result = match_uncertainty_categories(text)
        self.assertIn(
            "may naturally be green",
            result,
            msg=f"Single exact match fail: Expected 'may naturally be green' in result, but got: {result}"
          )
        self.assertEqual(
            1,
            len(result),
          )

    def test_single_partial_match(self):
        text = "2 unclear signs of ag"
        result = match_uncertainty_categories(text)
        self.assertIn(
            "unclear signs of agriculture",
            result,
            msg=f"Single partial match fail: Expected 'unclear signs of agriculture' in result, but got: {result}"
        )
        self.assertEqual(
            1,
            len(result),
          )

        text = "3 uNCLEAr sings of ag"
        result = match_uncertainty_categories(text)
        self.assertIn(
            "unclear signs of agriculture",
            result,
            msg=f"Single partial match fail: Expected 'unclear signs of agriculture' in result, but got: {result}"
        )
        self.assertEqual(
            1,
            len(result),
          )

        text = "3 may be naturally green"
        result = match_uncertainty_categories(text)
        self.assertIn(
            "may naturally be green",
            result,
            msg=f"Single partial match fail: Expected 'may naturally be green' in result, but got: {result}"
        )
        self.assertEqual(
            1,
            len(result),
        )

    def test_multiple_exact_matches(self):
        text = "3 only slightly green\n; may be a fishpond\n"
        result = match_uncertainty_categories(text)
        self.assertIn(
            "only slightly green",
            result,
            msg=f"Multiple exact matches fail: Expected 'only slightly green' in result, but got: {result}"
        )
        self.assertIn(
            "may be a fishpond",
            result,
            msg=f"Multiple exact matches fail: Expected 'may be a fishpond' in result, but got: {result}"
        )
        self.assertEqual(
            2,
            len(result),
        )


    def test_multiple_partial_matches(self):
        text = "2 unevne\n only slgthly gren"
        result = match_uncertainty_categories(text)
        self.assertIn(
            "uneven",
            result,
            msg=f"Multiple partial matches fail: Expected 'uneven' in result, but got: {result}"
        )
        self.assertIn(
            "only slightly green",
            result,
            msg=f"Multiple partial matches fail: Expected 'only slightly green' in result, but got: {result}"
        )
        self.assertEqual(
            2,
            len(result),
        )

"""
Tests for special category matching (get_special_categories)
"""
class TestSpecialCategoriesMatch(unittest.TestCase):
    def setUp(self):
        self.FLAG_GROUPS = {
            "plantation": ["agroforestry", "plantation"],
            "industrial": ["industrial", "commercial"],
            "lawn": ["lawn"],
            "covered": ["covered"]
        }

    def test_single_exact_match(self):
        text = "3 commercial may be naturally green"
        result = get_special_categories(text, self.FLAG_GROUPS)
        self.assertIn(
            "industrial",
            result,
            msg=f"Exact match fail: Expected 'industrial' in result, but got: {result}"
        )
        self.assertEqual(
            1,
            len(result),
        )

    def test_single_partial_match(self):
        text = "3 commerical"
        result = get_special_categories(text, self.FLAG_GROUPS)
        self.assertIn(
            "industrial",
            result,
            msg=f"Exact match fail: Expected 'industrial' in result, but got: {result}"
        )
        self.assertEqual(
            1,
            len(result),
        )

    def test_multiple_exact_matches(self):
        text = "3 plantation lawn may be naturally green "
        result = get_special_categories(text, self.FLAG_GROUPS)
        self.assertIn(
            'plantation',
            result,
            msg=f"Multiple exact matches fail: Expected 'plantation' in result, but got: {result}"
        )
        self.assertIn(
            'lawn',
            result,
            msg=f"Multiple exact matches fail: Expected 'lawn' in result, but got: {result}"
        )
        self.assertEqual(
            2,
            len(result),
        )

    def test_multiple_partial_matches(self):
        text = "3 lwn coverde may be naturally green."
        result = get_special_categories(text, self.FLAG_GROUPS)
        self.assertIn(
            'lawn',
            result,
            msg=f"Multiple exact matches fail: Expected 'lawn' in result, but got: {result}"
        )
        self.assertIn(
            'covered',
            result,
            msg=f"Multiple exact matches fail: Expected 'covered' in result, but got: {result}"
        )
        self.assertEqual(
            2,
            len(result),
        )

"""
Tests that parse_description works correctly.
"""
class TestParseDescription(unittest.TestCase):
  def test_blank(self):
    text = ""
    result = parse_description(text)
    self.assertEqual(
        {
            "certainty": 5,
            "uncertainty_explanation": "",
            "special_category": ""
        },
        result,
        msg=f"Blank text fail: Expected certainty 5, uncertainty_explanation blank, special_category blank, but got: {result}"
    )
    text = "\n4\n\n\n"
    result = parse_description(text)
    self.assertEqual(
        {
            "certainty": 4,
            "uncertainty_explanation": "",
            "special_category": ""
        },
        result,
        msg=f"Blank text fail: Expected certainty 4, uncertainty_explanation blank, special_category blank, but got: {result}"
    )

  def test_no_description(self):
    text="4\n"
    result = parse_description(text)
    self.assertEqual(
        {
            "certainty": 4,
            "uncertainty_explanation": "",
            "special_category": ""
        },
        result,
        msg=f"No description fail: Expected certainty 4, uncertainty_explanation blank, special_category blank, but got: {result}"
    )


  def test_single_special_category(self):
    text="\n4; \nagroforstry \n\n\nonly slightly geen"
    result = parse_description(text)
    self.assertEqual(
        {
            "certainty": 4,
            "uncertainty_explanation": "only slightly green",
            "special_category": "plantation"
        },
        result,
        msg=f"Single special category: Expected certainty 4, uncertainty_explanation 'only slightly green', special_category 'plantation', but got: {result}"
    )

    text="3 \ncomerical"
    result = parse_description(text)
    self.assertEqual(
        {
            "certainty": 3,
            "uncertainty_explanation": "",
            "special_category": "industrial"
        },
        result,
        msg=f"Single special category: Expected certainty 3, uncertainty_explanation '', special_category 'industrial', but got: {result}"
    )

    text="3 \n may be fishpond\n comerical"
    result = parse_description(text)
    self.assertEqual(
        {
            "certainty": 3,
            "uncertainty_explanation": "may be a fishpond",
            "special_category": "industrial"
        },
        result,
        msg=f"Single special category: Expected certainty 3, uncertainty_explanation 'may be a fishpond', special_category 'industrial', but got: {result}"
    )

    text = "3 \n tree crops\n kinda looks like a tree?"
    result = parse_description(text)
    self.assertEqual(
        {
            "certainty": 3,
            "uncertainty_explanation": "",
            "special_category": "plantation"
        },
        result,
        msg=f"Single special category: Expected certainty 3, uncertainty_explanation '', special_category 'plantation', but got: {result}"
    )


  def test_no_special_category(self):
    text="2\n may b naturally grene"
    result = parse_description(text)
    self.assertEqual(
        {
            "certainty": 2,
            "uncertainty_explanation": "may naturally be green",
            "special_category": ""
        },
        result,
        msg=f"Single special category: Expected certainty 2, uncertainty_explanation 'may naturally be green', special_category '', but got: {result}"
    )

    text="4\n may b naturally grene"
    result = parse_description(text)
    self.assertEqual(
        {
            "certainty": 4,
            "uncertainty_explanation": "may naturally be green",
            "special_category": ""
        },
        result,
        msg=f"Single special category: Expected certainty 4, uncertainty_explanation 'may naturally be green', special_category '', but got: {result}"
    )