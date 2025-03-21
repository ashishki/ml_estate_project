import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generates a more realistic synthetic real estate dataset with additional features and noise.
    
    Features:
    - price: Property price between 50,000 and 500,000
    - location: Categorical variable with values 'Central', 'Suburban', 'Rural'
    - last_sale_date: Date of the last sale within the past year
    - demand: Simulated demand in the area (0 to 1)
    - demographics: Simulated demographic information (average income in thousands)
    - area_sqft: Property area in square feet (between 500 and 5000)
    - num_rooms: Number of rooms (integer between 1 and 10)
    - property_condition: A value between 0 and 1 indicating property condition (1 is best)
    - sold: Binary target variable (1 if sold, 0 if not sold), based on a heuristic
    """
    # Validate input
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if n_samples == 0:
        # Return an empty DataFrame with the expected columns and types
        return pd.DataFrame({
            'price': pd.Series(dtype='float64'),
            'location': pd.Series(dtype='object'),
            'last_sale_date': pd.Series(dtype='datetime64[ns]'),
            'demand': pd.Series(dtype='float64'),
            'demographics': pd.Series(dtype='float64'),
            'area_sqft': pd.Series(dtype='float64'),
            'num_rooms': pd.Series(dtype='int64'),
            'property_condition': pd.Series(dtype='float64'),
            'sold': pd.Series(dtype='int64')
        })

    # Generate property prices between 50,000 and 500,000
    prices = np.random.uniform(50000, 500000, n_samples)
    
    # Randomly select property location
    locations = np.random.choice(['Central', 'Suburban', 'Rural'], n_samples)
    
    # Generate random last sale dates within the past year
    start_date = datetime.now() - timedelta(days=365)
    last_sale_dates = [start_date + timedelta(days=int(x)) for x in np.random.uniform(0, 365, n_samples)]
    
    # Generate demand as a random value between 0 and 1
    demand = np.random.uniform(0, 1, n_samples)
    
    # Generate demographic information (average income in thousands)
    demographics = np.random.uniform(20, 100, n_samples)
    
    # Generate additional features:
    # area_sqft: Property area in square feet (between 500 and 5000)
    area_sqft = np.random.uniform(500, 5000, n_samples)
    # num_rooms: Number of rooms (integer between 1 and 10)
    num_rooms = np.random.randint(1, 11, n_samples)
    # property_condition: A value between 0 and 1 indicating property condition (1 is best)
    property_condition = np.random.uniform(0, 1, n_samples)
    
    # Calculate probability of sale based on heuristic factors:
    price_factor = (prices.max() - prices) / (prices.max() - prices.min())
    demand_factor = demand
    demographics_factor = (demographics - demographics.min()) / (demographics.max() - demographics.min())
    rooms_factor = np.clip(num_rooms / 10, 0, 1)
    
    # Combine factors with empirically determined weights:
    # Price factor: 30%, Demand: 25%, Demographics: 25%, Property Condition: 10%, Rooms Factor: 5%
    prob_sold = (
        price_factor * 0.3 +
        demand_factor * 0.25 +
        demographics_factor * 0.25 +
        property_condition * 0.1 +
        rooms_factor * 0.05
    )
    
    # Add random noise to simulate real-world data variability
    noise = np.random.normal(0, 0.05, n_samples)
    prob_sold += noise
    
    # Clip probabilities to the valid range [0, 1]
    prob_sold = np.clip(prob_sold, 0, 1)
    
    # Generate the target variable 'sold' based on calculated probability
    sold = np.random.binomial(1, prob_sold)
    
    # Create a DataFrame with the generated data
    data = pd.DataFrame({
        'price': prices,
        'location': locations,
        'last_sale_date': last_sale_dates,
        'demand': demand,
        'demographics': demographics,
        'area_sqft': area_sqft,
        'num_rooms': num_rooms,
        'property_condition': property_condition,
        'sold': sold
    })
    
    logger.info("Synthetic data generation successful.")
    return data

if __name__ == "__main__":
    try:
        df = generate_synthetic_data()
        logger.info("Sample of the generated synthetic data:")
        logger.info(df.head().to_string())
    except Exception as e:
        logger.error("Failed to generate synthetic data.")
