import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.preprocessing import preprocess_data

def test_preprocess_data_basic():
    """
    Test that preprocess_data creates 'days_since_last_sale' and one-hot encodes 'location'.
    """
    data = pd.DataFrame({
        "price": [100000, 150000],
        "location": ["Central", "Rural"],
        "last_sale_date": [
            datetime.now() - timedelta(days=100),
            datetime.now() - timedelta(days=200)
        ],
        "demand": [0.5, 0.7],
        "demographics": [50, 60],
        "area_sqft": [1000, 2000],
        "num_rooms": [3, 4],
        "property_condition": [0.8, 0.6],
        "sold": [1, 0]
    })
    
    processed = preprocess_data(data)
    
    # Check that 'last_sale_date' is dropped
    assert "last_sale_date" not in processed.columns, "last_sale_date should be dropped"
    # Check that 'days_since_last_sale' is created
    assert "days_since_last_sale" in processed.columns, "days_since_last_sale should be created"
    # Check one-hot encoding for 'location'
    assert "location_Rural" in processed.columns or "location_Central" in processed.columns, "One-hot encoding for location is missing"

def test_preprocess_data_no_location_column():
    """
    If 'location' is missing, function should log a warning but not fail.
    """
    data = pd.DataFrame({
        "price": [100000],
        "last_sale_date": [datetime.now() - timedelta(days=50)],
        "demand": [0.5],
        "demographics": [50],
        "area_sqft": [1000],
        "num_rooms": [3],
        "property_condition": [0.8],
        "sold": [1]
    })
    
    processed = preprocess_data(data)
    # Check that it didn't break
    assert "days_since_last_sale" in processed.columns, "days_since_last_sale should be created"
    # location was missing, so no one-hot columns
    for col in processed.columns:
        assert not col.startswith("location_"), "Should not have created location columns"

def test_preprocess_data_missing_values():
    """
    Check that the function logs a warning if missing values are found.
    """
    data = pd.DataFrame({
        "price": [100000, None],
        "location": ["Central", "Suburban"],
        "last_sale_date": [datetime.now(), None],
        "demand": [0.5, 0.7],
        "demographics": [50, None],
        "area_sqft": [1000, 2000],
        "num_rooms": [3, 4],
        "property_condition": [0.8, 0.6],
        "sold": [1, 0]
    })
    processed = preprocess_data(data)
    # The function doesn't fill or drop missing values by default, it only logs a warning.
    # Just check that it runs without error.
    assert len(processed) == 2, "Should not drop rows by default"



def test_preprocess_data_no_columns():
    """
    Test behavior when DataFrame has no columns at all.
    Expect function not to crash but simply return the same empty DataFrame.
    """
    data = pd.DataFrame()
    processed = preprocess_data(data)
    assert processed.empty, "Processed DataFrame should be empty if input is empty"
    assert len(processed.columns) == 0, "No columns should be created from an empty DataFrame"

def test_preprocess_data_non_datetime():
    """
    If 'last_sale_date' is present but not in datetime format,
    the function should coerce it. Invalid parsing leads to NaT, 
    and days_since_last_sale might become NaN.
    """
    data = pd.DataFrame({
        "last_sale_date": ["not a date", "2024-01-01"],
        "location": ["Central", "Suburban"],
        "price": [100000, 150000],
        "demand": [0.5, 0.7],
        "demographics": [50, 60],
        "area_sqft": [1000, 2000],
        "num_rooms": [3, 4],
        "property_condition": [0.8, 0.6],
        "sold": [1, 0]
    })
    processed = preprocess_data(data)
    # Check that last_sale_date is dropped and days_since_last_sale is created
    assert "last_sale_date" not in processed.columns
    assert "days_since_last_sale" in processed.columns
    # Check for possible NaN if date was invalid
    assert processed["days_since_last_sale"].isna().sum() == 1, "One invalid date should result in NaN"

def test_preprocess_data_multiple_locations():
    """
    Test one-hot encoding with multiple location categories.
    """
    data = pd.DataFrame({
        "location": ["Central", "Rural", "Suburban", "Rural"],
        "price": [100000, 120000, 140000, 160000],
        "last_sale_date": [datetime.now(), datetime.now(), datetime.now(), datetime.now()],
        "demand": [0.3, 0.5, 0.6, 0.8],
        "demographics": [40, 55, 65, 75],
        "area_sqft": [900, 1500, 2500, 3000],
        "num_rooms": [2, 3, 5, 4],
        "property_condition": [0.9, 0.7, 0.8, 0.6],
        "sold": [0, 1, 1, 0]
    })
    processed = preprocess_data(data)
    assert "location_Rural" in processed.columns, "Expected a location_Rural column"
    assert "location_Suburban" in processed.columns, "Expected a location_Suburban column"
    # Since drop_first=True, we won't see location_Central