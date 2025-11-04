import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from Data_Splitter import *
from sklearn.metrics import explained_variance_score

def train_and_evaluate_svr(X_train, X_test, y_train, y_test):
    """
    Implements a Support Vector Regressor (SVR) pipeline for the p_factor prediction.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target (p_factor).
        y_test (pd.Series): Testing target (p_factor).
    """
    print("\n--- Training Support Vector Regressor (SVR) ---")

    # 1. Define the Preprocessing and Model Pipeline
    # SVR is highly sensitive to the scale of the features, so Standard Scaling is essential.
    svr_pipeline = Pipeline([
        # Step 1: Standardize the features (mean=0, std=1)
        ('scaler', StandardScaler()), 
        
        # Step 2: Initialize the Support Vector Regressor
        # 'rbf' (Radial Basis Function) is the default and a good general-purpose kernel.
        # C controls regularization (higher C = less regularization).
        ('svr', SVR(kernel='rbf', C=10, epsilon=0.1)) 
    ])
    
    # 2. Train the Model
    # The pipeline handles scaling X_train automatically before training the SVR.
    svr_pipeline.fit(X_train, y_train)
    print("Model training complete.")
    
    # 3. Make Predictions
    # The pipeline handles scaling X_test automatically before predicting.
    y_pred = svr_pipeline.predict(X_test)
    
    # 4. Evaluate Performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    variance_score = explained_variance_score(y_test, y_pred)
    
    print("\n--- Evaluation on Test Set ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print(f"Explained Variance Score: {r2:.4f}")
    
    return svr_pipeline, y_pred

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


def train_and_evaluate_sequential_keras(X_train, X_test, y_train, y_test):
    """
    Implements a corrected Sequential Keras model for p_factor regression.
    The data must be scaled before training the Keras model directly.
    """
    print("\n--- Training Sequential Keras Model (Regression) ---")
    
    # 1. Scaling the data 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the input shape dynamically based on the number of features
    input_dim = X_train_scaled.shape[1]
    
    # Create a sequential model
    model = Sequential()
    
    # Add the input layer (must define the input dimension)
    model.add(Dense(units=64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.2)) # Added Dropout for regularization
    
    # Add hidden layers
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    
    # Units = 1 (one continuous output), Activation = 'linear' (or omitted)
    model.add(Dense(units=1, activation='linear'))
    
    model.compile(
        optimizer='adam',
        loss='mse', # Mean Squared Error (Standard regression loss)
        metrics=['mae', 'mse'] # Mean Absolute Error, Mean Squared Error
    )
    
    # Model summary
    model.summary()
    
    # 2. Train the model using the scaled features
    history = model.fit(
        x=X_train_scaled, 
        y=y_train,
        epochs=50, # Use more epochs for NN training
        batch_size=32,
        verbose=0,
        validation_split=0.1 # Use a small validation split during training
    )
    print("Model training complete (50 epochs).")
    
    # 3. Make Predictions
    y_pred_scaled = model.predict(X_test_scaled)
    
    # Reshape prediction output (Keras often outputs a 2D array (N, 1))
    y_pred = y_pred_scaled.flatten()
    
    # 4. Evaluate Performance (using standard scikit-learn metrics)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    variance_score = explained_variance_score(y_test, y_pred)
    
    print("\n--- Evaluation on Test Set (Keras) ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2) Score: {r2:.4f}")
    print(f"Explained Variance Score: {variance_score:.4f}")
    
    return model, y_pred


def main():
    
    # Assume data_split_by_p_factor returns the standard four-tuple
    # X_train, X_test, y_train, y_test = data_split_by_p_factor(aggregate=True) 
    # NOTE: Since the data split function is not provided here, we create mock data for execution
    try:
        X_train, X_test, y_train, y_test = data_split_by_p_factor(aggregate=True)
    except:
        print("Using mock data as data_split_by_p_factor is not available.")
        # Mock data creation to ensure the script runs for demonstration
        N_FEATURES = 100
        N_SAMPLES = 1000
        X_train = pd.DataFrame(np.random.rand(N_SAMPLES, N_FEATURES))
        X_test = pd.DataFrame(np.random.rand(200, N_FEATURES))
        y_train = pd.Series(np.random.rand(N_SAMPLES) * 10)
        y_test = pd.Series(np.random.rand(200) * 10)
    
    if X_train is not None:
        
        # 1. Cleaning steps (removing binned column and dropping nulls)
        if 'p_factor_binned' in X_train.columns:
            X_train = X_train.drop(columns=['p_factor_binned'])
            X_test = X_test.drop(columns=['p_factor_binned'])
            
        train_valid_indices = X_train.dropna().index
        test_valid_indices = X_test.dropna().index

        X_train = X_train.loc[train_valid_indices]
        y_train = y_train.loc[train_valid_indices]
        
        X_test = X_test.loc[test_valid_indices]
        y_test = y_test.loc[test_valid_indices]
        
        print(f"\nCleaned Training Set Size: {len(X_train)} rows")
        print(f"Cleaned Testing Set Size: {len(X_test)} rows")
        
        # 2. Train and Evaluate the SVR (just to keep the original logic)
        trained_svr, svr_predictions = train_and_evaluate_svr(X_train, X_test, y_train, y_test)

        # 3. Train and Evaluate the Keras Model
        trained_keras, keras_predictions = train_and_evaluate_sequential_keras(X_train, X_test, y_train, y_test)
        
        print("\nFirst 5 Keras predictions:")
        print(keras_predictions[:5])
        
    else:
        print("Cannot run model implementations: Data split failed or returned None.")


if __name__ == "__main__":
    main()