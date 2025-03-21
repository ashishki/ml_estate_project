import pytest
import pandas as pd
from src.data_generation import generate_synthetic_data
from src.preprocessing import preprocess_data
from src.modeling import train_and_evaluate_model
from src.interpretation import interpret_model
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def test_interpret_model_no_crash(tmp_path):
    """
    Check that the interpret_model function runs without errors and can save plots.
    """
    raw_data = generate_synthetic_data(n_samples=100)
    preprocessed = preprocess_data(raw_data)
    best_model, X_test, y_test = train_and_evaluate_model(preprocessed, use_smote=False, test_size=0.2)
    
    # We set save_plots=True to ensure that plots are saved to a temporary directory
    output_dir = tmp_path / "plots"
    interpret_model(best_model, X_test, save_plots=True, output_dir=str(output_dir))
    
    # Check that files were created
    bar_plot = output_dir / "shap_bar_plot.png"
    summary_plot = output_dir / "shap_summary_plot.png"
    assert bar_plot.exists(), "SHAP bar plot was not saved"
    assert summary_plot.exists(), "SHAP summary plot was not saved"

def test_interpret_model_wrong_model(tmp_path):
    """
    Passing a non-XGBClassifier model should still work if shap.Explainer can handle it,
    but we check for any potential issues or warnings.
    """
    # Create dummy data
    raw_data = generate_synthetic_data(n_samples=50)
    preprocessed = preprocess_data(raw_data)
    # We'll pass a dummy classifier that isn't XGB
    from sklearn.ensemble import RandomForestClassifier
    dummy_model = RandomForestClassifier()
    # Fit dummy model quickly
    X = preprocessed.drop('sold', axis=1)
    y = preprocessed['sold']
    dummy_model.fit(X, y)
    
    output_dir = tmp_path / "plots"
    interpret_model(dummy_model, X, save_plots=True, output_dir=str(output_dir))
    # We just check that it doesn't crash and saves plots if shap can handle it
    bar_plot = output_dir / "shap_bar_plot.png"
    summary_plot = output_dir / "shap_summary_plot.png"
    # They might not be created if shap doesn't support this model well, 
    # but let's see if we need to handle that scenario explicitly.
    # Here we simply check that the function didn't raise an exception.

def test_interpret_model_empty_data(tmp_path):
    """
    If X is empty, interpret_model might fail or do nothing. We test that it doesn't crash.
    """
    from sklearn.ensemble import RandomForestClassifier
    dummy_model = RandomForestClassifier()
    # Fit on minimal data
    df = generate_synthetic_data(n_samples=2)
    preproc = preprocess_data(df)
    X = preproc.drop('sold', axis=1)
    y = preproc['sold']
    dummy_model.fit(X, y)
    
    # Now pass an empty DataFrame for interpretation
    empty_df = pd.DataFrame()
    output_dir = tmp_path / "plots"
    
    interpret_model(dummy_model, empty_df, save_plots=True, output_dir=str(output_dir))
    # We expect no crash. SHAP might throw warnings or produce empty plots.


def test_interpret_model_save_plots(tmp_path):
    """
    Test that interpret_model saves the SHAP plots in the specified output directory.
    """
    raw_data = generate_synthetic_data(n_samples=100)
    preprocessed = preprocess_data(raw_data)
    best_model, X_test, y_test = train_and_evaluate_model(preprocessed, use_smote=False)
    
    output_dir = tmp_path / "interpretation_plots"
    interpret_model(best_model, X_test, save_plots=True, output_dir=str(output_dir))
    
    expected_files = ["shap_bar_plot.png", "shap_summary_plot.png"]
    for fname in expected_files:
        file_path = output_dir / fname
        assert file_path.exists(), f"Expected file {fname} to be created in {output_dir}"

def test_interpret_model_empty_data(tmp_path):
    """
    Test that interpret_model handles an empty DataFrame gracefully by logging a warning and not creating any plots.
    """
    # Using a dummy model that can be fit on minimal data.
    dummy_model = RandomForestClassifier()
    df = generate_synthetic_data(n_samples=10)
    preprocessed = preprocess_data(df)
    X = preprocessed.drop('sold', axis=1)
    y = preprocessed['sold']
    dummy_model.fit(X, y)
    
    # Pass an empty DataFrame for interpretation.
    output_dir = tmp_path / "interpretation_empty_plots"
    interpret_model(dummy_model, pd.DataFrame(), save_plots=True, output_dir=str(output_dir))
    
    # Check that no plot files are created.
    # output_dir may be created by os.makedirs, but should be empty.
    assert not any(output_dir.iterdir()), "No plots should be created for empty input data."