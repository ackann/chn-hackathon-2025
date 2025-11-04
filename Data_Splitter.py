import zarr
import xarray as xr
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
# Requires pip install xarray

def load_data_from_zarr(zarr_path='merged_data.zarr'):
    """
    Loads the merged data from the Zarr store and returns it as a pandas DataFrame.
    """
    try:
        # 1. Open the Zarr store as an xarray Dataset
        ds = xr.open_zarr(zarr_path)
        
        # 2. Convert the xarray Dataset back to a pandas DataFrame
        df = ds.to_dataframe()
        
        # 3. Reset the index if needed (often necessary after xarray conversion)
        df = df.reset_index(drop=True)
        
        print(f"Successfully loaded data from Zarr store: {zarr_path}")
        print(f"Data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: Zarr store not found at {zarr_path}. Did you run merge_data(aggregate=False) first?")
        return None
    except Exception as e:
        print(f"An error occurred during Zarr loading: {e}")
        return None


def data_split_by_p_factor(aggregate=True, target_column="p_factor", n_bins= 20):
    """
    Performs a Stratified Group Train/Test split on the input DataFrame.
    
    Args:
        input_df (pd.DataFrame): The combined demographic and brain scan data.
        target_column (str): The column containing the continuous target variable (e.g., p_factor).
        n_bins (int): Number of bins to use for stratifying the continuous target.
        
    Returns:
        tuple: (train_df, test_df) ready for modeling.
    """

    if aggregate:
        merged_data_df = pd.read_csv("posteda.csv")
    else:
        merged_data_df = load_data_from_zarr("merged_data.zarr")

    if merged_data_df is None:
        print("Error: Data could not be loaded. Cannot perform split.")
        return None, None

    if target_column + '_x' in merged_data_df.columns and target_column + '_y' in merged_data_df.columns:
            print(f"Detected duplicate merge columns: {target_column}_x and {target_column}_y. Selecting {target_column}_x.")
            
            # 1. Select the '_x' column and rename it back to the target_column name
            merged_data_df = merged_data_df.rename(
                columns={target_column + '_x': target_column}
            )
            
            # 2. Drop the redundant '_y' column
            merged_data_df = merged_data_df.drop(
                columns=[target_column + '_y']
            )
    
    print("\nStarting Train/Test Split...")

    merged_df_clean = merged_data_df.dropna(subset=['p_factor'])
    unique_participants_df = merged_df_clean.drop_duplicates(subset=['participant_id'])
    
    # Bin the Target Variable (p_factor)
    unique_participants_df['p_factor_binned'] = pd.qcut(
        unique_participants_df['p_factor'],
        q=n_bins,
        labels=False,
        # 'drop' handles cases where certain values are repeated, resulting in fewer than n_bins
        duplicates='drop'
    )
    
    # Merge the binned label back into the full dataset
    merged_df_clean = pd.merge(
        merged_df_clean,
        unique_participants_df[['participant_id', 'p_factor_binned']],
        on='participant_id',
        how='left'
    )

    # --- 3. Step: Perform the Stratified Group Split ---
    
    # Define the inputs for the split
    # X and y are technically placeholders, the split operates on indices
    X = merged_df_clean.drop(columns=['p_factor']) # Features
    y_stratify = merged_df_clean['p_factor_binned'] # Binned target for stratification
    groups = merged_df_clean['participant_id'] # Participant IDs for grouping
    
    # Initialize StratifiedGroupKFold (n_splits=5 gives 80/20 split)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get the indices for the single train/test split
    # We iterate once to get the first fold (the single split)
    for train_index, test_index in sgkf.split(X, y_stratify, groups):
        # Use the indices to slice the cleaned DataFrame
        train_df = merged_df_clean.iloc[train_index]
        test_df = merged_df_clean.iloc[test_index]
        break
    
    print(f"Total rows after dropping nulls: {len(merged_df_clean)}")
    print(f"Train Set Size (approx. 80%): {len(train_df)} rows")
    print(f"Test Set Size (approx. 20%): {len(test_df)} rows")
    print(f"Number of Unique Participants in Train Set: {train_df['participant_id'].nunique()}")
    print(f"Number of Unique Participants in Test Set: {test_df['participant_id'].nunique()}")
    
    # --- Verification of Stratification ---
    print("\n--- Target Distribution Check (should be similar) ---")
    print("Train Set Binned Target Distribution:")
    print(train_df['p_factor_binned'].value_counts(normalize=True).sort_index().round(4))
    print("\nTest Set Binned Target Distribution:")
    print(test_df['p_factor_binned'].value_counts(normalize=True).sort_index().round(4))

    # 1. Define columns to drop from the feature set (X)
    # This includes the actual target, the binned target, and the grouping ID.
    cols_to_drop = [target_column, 'p_factor_binned', 'participant_id'] 
    
    # 2. Create the final four arrays/DataFrames
    X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
    y_train = train_df[target_column]
    
    X_test = test_df.drop(columns=cols_to_drop, errors='ignore')
    y_test = test_df[target_column]
    
    print("\n--- Model Preparation Summary ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Return the standard four-tuple for machine learning
    return X_train, X_test, y_train, y_test

def main():

    X_train, X_test, y_train, y_test = data_split_by_p_factor(aggregate=True) 
    
    if X_train is not None:
        print("\nReady for Modeling:")
        print(f"Features (X_train) head:\n{X_train.head()}")
        print(f"Target (y_train) head:\n{y_train.head()}")

if __name__ == "__main__":
    main()