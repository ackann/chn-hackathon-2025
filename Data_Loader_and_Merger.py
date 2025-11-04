from rbclib import RBCPath
from pathlib import Path 
import pandas as pd
import numpy as np
import csv
import zarr
import gc # Import for garbage collection
# Note: You may need to install 'fastparquet' or 'pyarrow' for to_parquet/read_parquet: pip install fastparquet
# Note: xarray library is also needed for the Zarr conversion.

def load_fsdata(participant_id, local_cache_dir):
    """
    Loads and returns the dataframe of a PNC participant's FreeSurfer data 
    from a specific TSV file within the FreeSurfer directory.
    """
    pnc_freesurfer_path = RBCPath(
        'rbc://PNC_FreeSurfer/freesurfer',
        local_cache_dir=local_cache_dir
    )
    participant_path = pnc_freesurfer_path / f'sub-{participant_id}'
    tsv_path = participant_path / f'sub-{participant_id}_regionsurfacestats.tsv' 

    try:
        # Open and read the TSV file from the cloud/cache
        with tsv_path.open('r') as f:
            data = pd.read_csv(f, sep='\t')
            
        # Clean and convert the 'subject_id' string column to integer
        if 'subject_id' in data.columns:
            # 1. Remove the "sub-" prefix
            data['subject_id'] = data['subject_id'].str.replace('sub-', '')
            
            # 2. Convert the resulting string to an integer
            data['subject_id'] = data['subject_id'].astype('int64')
            
        return data
    except Exception as e:
        print(f"Error loading FreeSurfer data for sub-{participant_id}: {e}")
        return None

def merge_data(aggregate):
    
    # --- Configuration (Define paths and cache) ---
    rbcdata_path = Path('/home/jovyan/shared/data/RBC')
    train_filepath = rbcdata_path / 'train_participants.tsv'
    local_cache_dir = Path.home() / 'rbc_cache'
    local_cache_dir.mkdir(exist_ok=True) 
    
    # Gather Demographic Data
    try:
        with train_filepath.open('r') as f:
            demographic_data = pd.read_csv(f, sep='\t')
    except FileNotFoundError:
        print(f"Error: The file {train_filepath} was not found.")
        return None

    # Check if participant_id exists and convert to integer
    if 'participant_id' not in demographic_data.columns:
        print("Error: 'participant_id' column not found in demographic data.")
        return None
    try:
        demographic_data['participant_id'] = demographic_data['participant_id'].astype('int64')
    except ValueError as e:
        print(f"Error converting 'participant_id' to integer in demographic data: {e}")
        return None
        
    # Initialize a list to collect brain scan DataFrames (only used in AGGREGATE mode for loading)
    brain_scan_data = None 

    if aggregate:
        # Load pre-aggregated data
        try:
            with open('aggregated_brain_scan_data.csv', mode='r') as f:
                brain_scan_data = pd.read_csv(f)
                
            # Ensure aggregated ID column is integer
            if 'participant_id' in brain_scan_data.columns:
                brain_scan_data['participant_id'] = brain_scan_data['participant_id'].astype('int64')
                
        except FileNotFoundError:
            print("Error: 'aggregated_brain_scan_data.csv' not found. Set aggregate=False to build it.")
            return None
    else:
        
        # Create a temporary directory for Parquet files
        temp_parquet_dir = Path(local_cache_dir) / 'temp_parquet_data'
        temp_parquet_dir.mkdir(exist_ok=True)
        print(f"Using temporary Parquet directory: {temp_parquet_dir}")

        processed_files = []
        
        for participant_id in demographic_data['participant_id']:
            
            pid_str = str(participant_id)
            print(f"\nProcessing participant: {pid_str}")
            
            # Load the FreeSurfer data (subject_id is already int64 here)
            subject_df = load_fsdata(pid_str, local_cache_dir)
            
            if subject_df is not None:
                # Save subject data directly to disk as Parquet
                temp_file = temp_parquet_dir / f'sub_{pid_str}.parquet'
                subject_df.to_parquet(temp_file)
                processed_files.append(temp_file)
                
            # Manually trigger garbage collection after processing each participant
            del subject_df 
            gc.collect() 

        # Efficiently read all temporary files into one DataFrame
        if processed_files:
            print("\nReading temporary Parquet files and concatenating...")
            brain_scan_data = pd.concat(
                [pd.read_parquet(f) for f in processed_files], 
                ignore_index=True
            )
        else:
            print("No brain scan data was loaded. Cannot merge.")
            return None
            
        # Clean up temporary files
        for f in temp_parquet_dir.glob("*.parquet"):
            f.unlink()
        temp_parquet_dir.rmdir()
        print("Cleaned up temporary files.")

        # Align the demographic merge key with the loaded brain scan data key
        demographic_data.rename(columns={'participant_id': 'subject_id'}, inplace=True)
        
    # --- Final Merge ---
    
    if aggregate:
        # Merge on 'participant_id' (both int64)
        merged_df = pd.merge(demographic_data, brain_scan_data, on='participant_id')
    else:
        # Merge on 'subject_id' (both int64, after renaming)
        merged_df = pd.merge(demographic_data, brain_scan_data, on='subject_id')
    
    # --- Final Data Saving ---
    
    if aggregate:
        output_filename = 'merged_data_aggregated.csv'
        merged_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully merged and saved data to: {output_filename} (CSV)")
    else:
        zarr_path = 'merged_data.zarr'
        # Final, large memory step: conversion and Zarr save
        merged_df.to_xarray().to_zarr(zarr_path, mode='w')
        print(f"\nSuccessfully merged and saved data to: {zarr_path} (Zarr)")
        
    return merged_df

def main():
    # --- Execute the non-aggregated version (runs the memory-efficient path) ---
    final_df_built = merge_data(aggregate=False)
    if final_df_built is not None:
        print("Non-Aggregated Data Head:")
        print(final_df_built.head())
    
if __name__ == "__main__":
    main()