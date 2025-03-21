import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from data_generation import generate_synthetic_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_or_show_plot(save_path: Optional[str] = None) -> None:
    """
    Saves the plot to a file if a save path is provided,
    otherwise displays the plot.
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def run_eda(data: pd.DataFrame, save_plots: bool = False, output_dir: str = "output") -> None:
    """
    Perform exploratory data analysis on the provided dataset.
    If save_plots is True, plots will be saved to the output directory.
    """
    # Log dataset information and descriptive statistics
    logger.info("Dataset Info:")
    logger.info(data.info())
    logger.info("\nDescriptive Statistics:")
    logger.info(data.describe().to_string())
    
    # Plot the distribution of the target variable 'sold'
    plt.figure(figsize=(6, 4))
    sns.countplot(x='sold', data=data)
    plt.title("Distribution of 'sold' variable")
    plt.xlabel("Sold")
    plt.ylabel("Count")
    if save_plots:
        save_or_show_plot(os.path.join(output_dir, "sold_distribution.png"))
    else:
        save_or_show_plot()
    
    # Plot histograms for numerical features
    numeric_features = ['price', 'demand', 'demographics', 'area_sqft', 'num_rooms', 'property_condition']
    for feature in numeric_features:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[feature], bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        if save_plots:
            save_or_show_plot(os.path.join(output_dir, f"{feature}_distribution.png"))
        else:
            save_or_show_plot()
    
    # Process and plot the 'last_sale_date' column if present
    if 'last_sale_date' in data.columns:
        data['last_sale_date'] = pd.to_datetime(data['last_sale_date'])
        plt.figure(figsize=(10, 4))
        plt.plot(data['last_sale_date'].sort_values())
        plt.title("Last Sale Date Distribution")
        plt.xlabel("Index")
        plt.ylabel("Last Sale Date")
        if save_plots:
            save_or_show_plot(os.path.join(output_dir, "last_sale_date_distribution.png"))
        else:
            save_or_show_plot()
    
    # Plot a correlation heatmap for numerical features
    plt.figure(figsize=(8, 6))
    corr = data.select_dtypes(include=['float64', 'int64']).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    if save_plots:
        save_or_show_plot(os.path.join(output_dir, "correlation_heatmap.png"))
    else:
        save_or_show_plot()

if __name__ == "__main__":
    try:
        df = generate_synthetic_data()
        run_eda(df, save_plots=False)
    except Exception as e:
        logger.error(f"Error running EDA: {e}")
