import pandas as pd
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_and_evaluate_model(data: pd.DataFrame, use_smote: bool = True, 
                             test_size: float = 0.2, random_state: int = 42,
                             cv: int = 5) -> Tuple[XGBClassifier, pd.DataFrame, pd.Series]:
    """
    Train and evaluate an XGBoost model on the provided dataset.

    Steps:
    - Splits the data into features (X) and target (y).
    - Splits data into training and testing sets.
    - Optionally applies SMOTE to balance the training set.
    - Performs hyperparameter tuning using GridSearchCV with cv folds.
    - Trains the model and evaluates it using ROC-AUC and F1-score.
    
    Args:
        data (pd.DataFrame): Preprocessed DataFrame with target variable 'sold'.
        use_smote (bool): Whether to apply SMOTE for balancing the training set.
        test_size (float): Proportion of the data to use as test set.
        random_state (int): Random state for reproducibility.
        cv (int): Number of folds for cross-validation.
    
    Returns:
        Tuple containing:
            - best_model (XGBClassifier): The trained model with best hyperparameters.
            - X_test (pd.DataFrame): The test features.
            - y_test (pd.Series): The test target values.
    """
    try:
        # Check if target column exists
        if 'sold' not in data.columns:
            raise ValueError("Target variable 'sold' not found in data.")
        
        # Separate features and target
        X = data.drop('sold', axis=1)
        y = data['sold']
        
        # Check that the target has at least two unique classes
        if len(y.unique()) < 2:
            raise ValueError("Target variable 'sold' must contain at least two classes for training.")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Data split into train and test sets with test size = {test_size}.")
        
        # Apply SMOTE for balancing if required
        if use_smote:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info("Applied SMOTE to balance the training data.")
        
        # Initialize XGBoost classifier
        model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        
        # Define hyperparameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        }
        
        # Perform grid search with cv-fold cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best hyperparameters found: {grid_search.best_params_}")
        
        # Evaluate model on test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Test ROC-AUC: {roc_auc:.3f}")
        logger.info(f"Test F1-score: {f1:.3f}")
        
        return best_model, X_test, y_test
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
        raise

if __name__ == "__main__":
    try:
        from data_generation import generate_synthetic_data
        from preprocessing import preprocess_data
        
        # Generate and preprocess the data
        raw_data = generate_synthetic_data()
        preprocessed_data = preprocess_data(raw_data)
        
        # Train and evaluate the model
        best_model, X_test, y_test = train_and_evaluate_model(preprocessed_data)
    except Exception as e:
        logger.error("Failed during model training and evaluation module execution.")
