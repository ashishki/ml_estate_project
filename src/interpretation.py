import os
import logging
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
from typing import Any
from xgboost import XGBClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def interpret_model(model: XGBClassifier, X: pd.DataFrame, save_plots: bool = False, output_dir: str = "output") -> None:
    """
    Interpret the model using SHAP values. This function generates SHAP summary and bar plots
    to explain the impact of each feature on the model's predictions.
    
    Args:
        model (XGBClassifier): The trained model.
        X (pd.DataFrame): The dataset for which to compute SHAP values.
        save_plots (bool): If True, plots will be saved to the specified output directory.
                           In non-interactive environments, consider setting this to True.
        output_dir (str): The directory where plots should be saved.
    """
    try:
        if X.empty:
            logger.warning("Empty input data provided for SHAP interpretation. Skipping plotting.")
            return
        
        # Create a SHAP explainer for the model
        explainer = shap.Explainer(model)
        # Compute SHAP values for the dataset
        shap_values = explainer(X)
        logger.info("SHAP values computed successfully.")
        
        # Create a reproducible random number generator instance
        rng_instance = np.random.default_rng(42)
        
        # Plot SHAP summary bar plot to show average impact of features
        plt.figure()
        # Pass explicit rng to silence FutureWarning regarding global RNG seeding
        shap.summary_plot(shap_values, X, plot_type="bar", show=not save_plots, rng=rng_instance)
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "shap_bar_plot.png"))
            logger.info("SHAP bar plot saved to output directory.")
            plt.close()
        else:
            # In some non-interactive environments, plt.show() might not work.
            # If plots do not appear, consider setting save_plots=True.
            plt.show()
        
        # Plot SHAP summary (beeswarm) plot to show distribution of feature impacts
        plt.figure()
        shap.summary_plot(shap_values, X, show=not save_plots, rng=rng_instance)
        if save_plots:
            plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
            logger.info("SHAP summary plot saved to output directory.")
            plt.close()
        else:
            plt.show()
        
    except Exception as e:
        logger.error(f"Error during model interpretation: {e}")
        raise

if __name__ == "__main__":
    try:
        from data_generation import generate_synthetic_data
        from preprocessing import preprocess_data
        from modeling import train_and_evaluate_model
        
        # Generate and preprocess data, then train the model
        raw_data = generate_synthetic_data()
        preprocessed_data = preprocess_data(raw_data)
        best_model, X_test, y_test = train_and_evaluate_model(preprocessed_data)
        
        # Interpret the model using SHAP plots.
        # If you are not in an interactive environment, set save_plots=True to save plots as files.
        interpret_model(best_model, X_test, save_plots=False)
    except Exception as e:
        logger.error("Failed during model interpretation module execution.")
