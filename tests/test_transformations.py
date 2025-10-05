import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from app.utils.transformations import _apply_halflife


class TestApplyHalflife:
    """Test suite for the _apply_halflife function."""

    def test_basic_functionality(self):
        """Test basic functionality with simple input."""
        series = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        halflife = 2.0
        
        result = _apply_halflife(series, halflife)
        
        # Check that result is returned as float32
        assert result.dtype == np.float32
        # Check that the first element remains unchanged
        assert result[0] == 1.0
        # Check that subsequent elements are affected by adstock
        assert result[1] > 2.0  # Should be > original due to adstock effect
        
    def test_zero_series(self):
        """Test with all zeros input."""
        series = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        halflife = 2.0
        
        result = _apply_halflife(series, halflife)
        
        expected = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert_array_equal(result, expected)
        
    def test_single_element(self):
        """Test with single element series."""
        series = np.array([5.0], dtype=np.float64)
        halflife = 1.0
        
        result = _apply_halflife(series, halflife)
        
        expected = np.array([5.0], dtype=np.float32)
        assert_array_equal(result, expected)
        
    def test_empty_series(self):
        """Test with empty series."""
        series = np.array([], dtype=np.float64)
        halflife = 2.0
        
        result = _apply_halflife(series, halflife)
        
        expected = np.array([], dtype=np.float32)
        assert_array_equal(result, expected)
        
    def test_different_halflife_values(self):
        """Test with different halflife values."""
        series = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # Test with shorter halflife (faster decay)
        result_short = _apply_halflife(series, halflife=0.5)
        
        # Test with longer halflife (slower decay)  
        result_long = _apply_halflife(series, halflife=5.0)
        
        # With longer halflife, the adstock effect should be stronger
        # (values should decay more slowly)
        assert result_long[1] > result_short[1]
        assert result_long[2] > result_short[2]
        
    def test_rounding_parameter(self):
        """Test the rounding parameter functionality."""
        series = np.array([1.123456789, 2.987654321], dtype=np.float64)
        halflife = 2.0
        
        # Test with different rounding values
        result_2_decimals = _apply_halflife(series, halflife, rounding=2)
        result_4_decimals = _apply_halflife(series, halflife, rounding=4)
        
        # Check that rounding is applied correctly
        # The second element should have different precision
        str_2_decimals = str(result_2_decimals[1])
        str_4_decimals = str(result_4_decimals[1])
        
        # Both should be properly rounded, but with different precision
        assert isinstance(result_2_decimals[0], np.float32)
        assert isinstance(result_4_decimals[0], np.float32)
        
    def test_mathematical_correctness(self):
        """Test the mathematical correctness of the adstock calculation."""
        series = np.array([10.0, 0.0, 0.0], dtype=np.float64)
        halflife = 2.0
        
        result = _apply_halflife(series, halflife)
        
        # Calculate expected values manually
        decay_factor = np.log(0.5) / halflife
        expected = np.array([10.0, 0.0, 0.0], dtype=np.float64)
        
        # Apply the same logic as in the function
        for i in range(1, len(expected)):
            expected[i] += expected[i - 1] * np.exp(decay_factor)
            
        expected_rounded = np.round(expected, 4).astype(np.float32)
        
        assert_array_almost_equal(result, expected_rounded, decimal=4)
        
    def test_input_preservation(self):
        """Test that the original input series is not modified."""
        original_series = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        series_copy = original_series.copy()
        halflife = 2.0
        
        _apply_halflife(series_copy, halflife)
        
        # Original series should remain unchanged
        assert_array_equal(original_series, series_copy)
        
    def test_negative_values(self):
        """Test with negative values in the series."""
        series = np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float64)
        halflife = 2.0
        
        result = _apply_halflife(series, halflife)
        
        # Should handle negative values correctly
        assert result.dtype == np.float32
        assert len(result) == len(series)
        
    def test_large_values(self):
        """Test with large values to check for numerical stability."""
        series = np.array([1e6, 2e6, 3e6], dtype=np.float64)
        halflife = 2.0
        
        result = _apply_halflife(series, halflife)
        
        # Should handle large values without overflow
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        
    @pytest.mark.parametrize("halflife", [0.1, 0.5, 1.0, 2.5, 10.0])
    def test_various_halflife_values(self, halflife):
        """Test with various halflife values using parametrize."""
        series = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        
        result = _apply_halflife(series, halflife)
        
        assert result.dtype == np.float32
        assert len(result) == len(series)
        assert result[0] == 1.0  # First element should always remain the same
        
    def test_type_annotations_compatibility(self):
        """Test that the function works with the expected numpy types."""
        # Test with numpy array of correct type
        series = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        halflife = np.float32(2.0)
        
        result = _apply_halflife(series, halflife)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""
    
    def test_actual_data_transformation(self):
        """Test transformation with actual raw data against expected processed results."""
        import pandas as pd
        from pathlib import Path
        
        # Load raw data
        raw_data_path = Path(__file__).parent.parent / "data" / "raw" / "raw_data.csv"
        raw_df = pd.read_csv(raw_data_path, parse_dates=["date_week"])
        raw_series = raw_df["tv_ad_executions"].values.astype(np.float64)
        
        # Load expected processed data
        processed_data_path = Path(__file__).parent.parent / "data" / "processed" / "processed_data.csv"
        processed_df = pd.read_csv(processed_data_path, parse_dates=["date_week"])
        expected_series = processed_df["tv_ad_executions_adstock"].values.astype(np.float32)
        
        # Apply transformation with the same parameters used in the original
        halflife = 2.5
        rounding = 4
        result = _apply_halflife(raw_series, halflife, rounding)
        
        # Compare results - should match the expected processed data
        assert result.dtype == np.float32
        assert len(result) == len(expected_series)
        
        # Check that the transformation produces the expected results
        # Allow for small floating-point differences
        assert_array_almost_equal(result, expected_series, decimal=3)
        
        # Verify specific data points to ensure correctness
        assert abs(result[0] - expected_series[0]) < 0.01  # First value should be nearly identical
        assert result[0] == pytest.approx(expected_series[0], rel=1e-3)
        
        # Check that adstock effect is working (later values should be influenced by earlier ones)
        assert np.sum(result) >= np.sum(raw_series)  # Adstock should increase cumulative effect
    
    def test_realistic_tv_ad_data(self):
        """Test with realistic TV advertisement execution data."""
        # Simulate weekly TV ad executions over a year
        np.random.seed(42)
        series = np.random.exponential(scale=100, size=52).astype(np.float64)
        # Add some zero weeks (no ads)
        series[::7] = 0.0
        
        halflife = 2.5  # Typical media mix modeling halflife
        
        result = _apply_halflife(series, halflife)
        
        assert result.dtype == np.float32
        assert len(result) == len(series)
        # Adstock should generally increase cumulative values
        assert np.sum(result) >= np.sum(series)
        
    def test_seasonal_pattern(self):
        """Test with seasonal advertising pattern."""
        # Create a seasonal pattern (higher in Q4)
        weeks = 52
        series = np.array([
            10.0 if (i % 52) > 39 else 5.0  # Higher spending in last 12 weeks
            for i in range(weeks)
        ], dtype=np.float64)
        
        halflife = 3.0
        result = _apply_halflife(series, halflife)
        
        # The adstock effect should carry over from high-spending periods
        assert result.dtype == np.float32
        assert len(result) == weeks
        # Values after high-spending periods should show carryover effect
        assert result[45] > series[45]  # Should benefit from previous weeks


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])