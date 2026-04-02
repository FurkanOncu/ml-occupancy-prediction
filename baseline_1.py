import logging
import pandas as pd
import numpy as np
import json
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


def main():
    """
    Main function to load data, train model, and generate submission file.
    """
    logging.info("--- Starting Model Training and Prediction ---")

    # --- 1. Load Data ---
    logging.info("Reading train.json and test.json files...")
    try:
        train_df = pd.read_json("train.json", orient='records')
        test_df = pd.read_json("test.json", orient='records')
    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Make sure train.json and test.json are in the same directory.")
        return
    except ValueError as e:
        logging.error(f"Error loading JSON: {e}. Check file format.")
        return

    logging.info(f"Training data loaded with {train_df.shape[0]} rows and {train_df.shape[1]} columns.")
    logging.info(f"Test data loaded with {test_df.shape[0]} rows and {test_df.shape[1]} columns.")

    # --- 2. Data Splitting ---
    logging.info("Splitting training data into train and validation sets...")
    label = 'occupancy'

    if label not in train_df.columns:
        logging.error(f"Error: Target column '{label}' not found in train.json.")
        return

    try:
        train_split, valid_split = train_test_split(train_df, test_size=0.2, random_state=42)

        y_train = train_split[label]
        X_train = train_split.drop(label, axis=1)

        y_valid = valid_split[label]
        X_valid = valid_split.drop(label, axis=1)

        X_test = test_df.copy()

    except Exception as e:
        logging.error(f"Error during data splitting: {e}")
        return

    # --- 3. Preprocessing Pipeline ---
    logging.info("Defining preprocessing pipelines...")

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    logging.info(f"Identified {len(numeric_features)} numeric features: {numeric_features}")
    logging.info(f"Identified {len(categorical_features)} categorical features: {categorical_features}")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # --- 4. Model Definition ---
    logging.info("Defining the model pipeline...")
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(
            random_state=42,
            loss='absolute_error',
            max_iter=500,
            learning_rate=0.07,
            max_depth=10,
            l2_regularization=1.0,
            scoring='neg_mean_absolute_error',
            early_stopping=True,
            validation_fraction=0.1
        ))
    ])

    # --- 5. Model Training & Evaluation ---
    logging.info("Fitting the model on the training data...")
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        logging.error(f"Error during model fitting: {e}")
        return

    logging.info("Evaluating model on the local validation set...")
    y_pred_valid = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred_valid)
    logging.info(f"--- Validation MAE: {mae:.4f} ---")
    logging.info("This MAE is an estimate of your leaderboard score. Lower is better.")

    # --- 6. Generate Test Predictions ---
    logging.info("Generating predictions on the test set...")
    try:
        pred_test = model.predict(X_test)
    except Exception as e:
        logging.error(f"Error during test prediction: {e}")
        return

    pred_test[pred_test < 0] = 0

    # --- 7. Create Submission File ---
    logging.info("Formatting predictions for submission...")
    submission_df = pd.DataFrame({'occupancy': pred_test})
    predicted_records = submission_df.to_dict(orient='records')

    output_zip_path = "submission.zip"
    output_json_name = "predicted.json"

    try:
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr(output_json_name, json.dumps(predicted_records, indent=2))
        logging.info(f"--- Successfully created submission file: {output_zip_path} ---")
    except Exception as e:
        logging.error(f"Error creating zip file: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
