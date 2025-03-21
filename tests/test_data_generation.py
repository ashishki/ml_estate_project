import pytest
import pandas as pd
from src.data_generation import generate_synthetic_data

def test_generate_synthetic_data_default():
    """
    Test that the function returns a DataFrame of the expected size with default params.
    """
    df = generate_synthetic_data()
    assert isinstance(df, pd.DataFrame), "Expected a pandas DataFrame"
    assert len(df) == 1000, "Expected 1000 rows by default"
    expected_columns = {
        "price", "location", "last_sale_date", "demand",
        "demographics", "area_sqft", "num_rooms", "property_condition", "sold"
    }
    assert expected_columns.issubset(df.columns), "Missing expected columns"

def test_generate_synthetic_data_custom_samples():
    """
    Test that the function correctly handles a custom n_samples value.
    """
    n_samples = 500
    df = generate_synthetic_data(n_samples=n_samples)
    assert len(df) == n_samples, f"Expected {n_samples} rows"

def test_generate_synthetic_data_values_range():
    """
    Test that generated values are within expected ranges.
    """
    df = generate_synthetic_data()
    assert (df["price"].min() >= 50000) and (df["price"].max() <= 500000), "Price out of expected range"
    assert (df["demand"].min() >= 0) and (df["demand"].max() <= 1), "Demand out of expected range"
    assert (df["area_sqft"].min() >= 500) and (df["area_sqft"].max() <= 5000), "area_sqft out of expected range"



def test_generate_synthetic_data_zero_samples():
    """
    Test generating a dataset with zero samples.
    Expect an empty DataFrame with the correct columns.
    """
    df = generate_synthetic_data(n_samples=0)
    assert df.empty, "DataFrame should be empty when n_samples=0"
    expected_columns = {
        "price", "location", "last_sale_date", "demand",
        "demographics", "area_sqft", "num_rooms", "property_condition", "sold"
    }
    assert expected_columns.issubset(df.columns), "Missing expected columns for zero samples"

def test_generate_synthetic_data_negative_samples():
    """
    Test generating a dataset with a negative sample size.
    Expect a ValueError or similar.
    """
    with pytest.raises(ValueError):
        _ = generate_synthetic_data(n_samples=-10)

def test_generate_synthetic_data_large_samples():
    """
    Test generating a large dataset (e.g., 10_000 samples).
    This checks performance and ensures no unexpected errors.
    """
    n_samples = 10_000
    df = generate_synthetic_data(n_samples=n_samples)
    assert len(df) == n_samples, f"Expected {n_samples} rows for large dataset"
    # We won't do heavy checks here to keep test time reasonable.
