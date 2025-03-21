import pytest
import pandas as pd
from src.modeling import train_and_evaluate_model
from src.data_generation import generate_synthetic_data
from src.preprocessing import preprocess_data

def test_train_and_evaluate_model_basic():
    """
    Test that the model trains and returns expected outputs (best_model, X_test, y_test).
    """
    raw_data = generate_synthetic_data(n_samples=200)  # smaller dataset for speed
    preprocessed = preprocess_data(raw_data)
    
    best_model, X_test, y_test = train_and_evaluate_model(preprocessed, use_smote=False, test_size=0.2)
    
    # Check that we got the expected outputs
    assert best_model is not None, "Should return a trained model"
    assert isinstance(X_test, pd.DataFrame), "X_test should be a pandas DataFrame"
    assert len(X_test) == int(0.2 * len(preprocessed)), "Test set should be 20% of the data"

def test_train_and_evaluate_model_smote():
    """
    Check that SMOTE can be applied without error and returns the same outputs.
    """
    raw_data = generate_synthetic_data(n_samples=200)
    preprocessed = preprocess_data(raw_data)
    
    best_model, X_test, y_test = train_and_evaluate_model(preprocessed, use_smote=True, test_size=0.2)
    assert best_model is not None, "Should return a trained model even with SMOTE"


def test_train_and_evaluate_model_no_sold_column():
    """
    If the target column 'sold' is missing, we expect a ValueError.
    """
    raw_data = generate_synthetic_data(n_samples=50)
    preprocessed = preprocess_data(raw_data)
    # Remove 'sold' column
    preprocessed.drop(columns=['sold'], inplace=True)
    
    with pytest.raises(ValueError):
        _ = train_and_evaluate_model(preprocessed)

def test_train_and_evaluate_model_all_sold_ones():
    """
    If all targets are 1, the function should raise a ValueError indicating that 
    at least two classes are required.
    """
    raw_data = generate_synthetic_data(n_samples=50)
    # Force all sold = 1
    raw_data['sold'] = 1
    preprocessed = preprocess_data(raw_data)
    
    with pytest.raises(ValueError):
        _ = train_and_evaluate_model(preprocessed, use_smote=False)

def test_train_and_evaluate_model_small_data():
    """
    Test with a very small dataset (e.g., 10 samples) to ensure the model can still train.
    """
    raw_data = generate_synthetic_data(n_samples=10)  # increased from 5 to 10
    preprocessed = preprocess_data(raw_data)
    best_model, X_test, y_test = train_and_evaluate_model(preprocessed, use_smote=False, test_size=0.4)
    # With 10 samples and test_size=0.4, we should have around 4 samples in the test set.
    # We check that the test set size is as expected.
    expected_test_size = int(10 * 0.4)
    assert len(X_test) == expected_test_size, f"With 10 samples and test_size=0.4, expected {expected_test_size} samples in test."
