import pandas as pd
import logging
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data by handling date columns, categorical variables, and basic cleaning.
    
    Steps:
    - Convert 'last_sale_date' to datetime and create a new feature 'days_since_last_sale'.
    - Drop the original 'last_sale_date' column.
    - Apply one-hot encoding to the 'location' column.
    - Optionally, check for missing values.
    
    Args:
        data (pd.DataFrame): The raw input DataFrame.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    try:
        # Process 'last_sale_date' if it exists
        if 'last_sale_date' in data.columns:
            data['last_sale_date'] = pd.to_datetime(data['last_sale_date'], errors='coerce')
            # Create a new feature: days since the last sale
            data['days_since_last_sale'] = (pd.Timestamp.now() - data['last_sale_date']).dt.days
            # Drop the original date column
            data.drop(columns=['last_sale_date'], inplace=True)
            logger.info("Converted 'last_sale_date' to 'days_since_last_sale'.")
        else:
            logger.warning("'last_sale_date' column not found in data.")
        
        # One-hot encode the 'location' column if present
        if 'location' in data.columns:
            data = pd.get_dummies(data, columns=['location'], drop_first=True)
            logger.info("Applied one-hot encoding to 'location' column.")
        else:
            logger.warning("'location' column not found in data.")
        
        # Check for missing values and log a warning if any found
        missing_values = data.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in the data. Consider handling them appropriately.")
        else:
            logger.info("No missing values found in the data.")
        
        return data
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

if __name__ == "__main__":
    # For testing purposes, generate a sample dataset and apply preprocessing
    from data_generation import generate_synthetic_data
    
    try:
        raw_data = generate_synthetic_data()
        logger.info("Raw data generated for preprocessing:")
        logger.info(raw_data.head().to_string())
        
        preprocessed_data = preprocess_data(raw_data)
        logger.info("Preprocessed data sample:")
        logger.info(preprocessed_data.head().to_string())
    except Exception as e:
        logger.error("Failed during preprocessing module execution.")
