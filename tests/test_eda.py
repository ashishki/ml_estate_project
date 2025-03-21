import os
import pytest
import pandas as pd
from src.eda import run_eda
from src.data_generation import generate_synthetic_data

def test_run_eda_save_plots(tmp_path):
    """
    Test the run_eda function with save_plots=True.
    Verifies that expected plot files are created in the output directory.
    """
    data = generate_synthetic_data(n_samples=50)
    output_dir = tmp_path / "eda_plots"
    # Run EDA with plots being saved
    run_eda(data, save_plots=True, output_dir=str(output_dir))
    
    # Expected plot filenames based on run_eda function implementation
    expected_files = [
        "sold_distribution.png",
        "price_distribution.png",
        "demand_distribution.png",
        "demographics_distribution.png",
        "area_sqft_distribution.png",
        "num_rooms_distribution.png",
        "property_condition_distribution.png",
        "last_sale_date_distribution.png",
        "correlation_heatmap.png",
    ]
    
    for fname in expected_files:
        file_path = output_dir / fname
        assert file_path.exists(), f"Expected file {fname} to be created in {output_dir}"
