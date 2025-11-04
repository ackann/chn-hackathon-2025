from rbclib import RBCPath
from pathlib import Path 
import pandas as pd
import numpy as np # (Though not strictly used, keeping for completeness)
import csv # (Though not strictly used, keeping for completeness)
import zarr
# Note: The zarr library must be installed (pip install zarr)

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
        return data
    except Exception as e:
        print(f"Error loading FreeSurfer data for sub-{participant_id}: {e}")
        return None

def merge_data(aggregate=True):
    
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

    # Check if participant_id exists
    if 'participant_id' not in demographic_data.columns:
        print("Error: 'participant_id' column not found in demographic data.")
        return None
        
    # Initialize a list to collect brain scan DataFrames
    brain_scan_data_list = []

    if aggregate:
        # Load pre-aggregated data
        try:
            with open('aggregated_brain_scan_data.csv', mode='r') as f:
                brain_scan_data = pd.read_csv(f)
        except FileNotFoundError:
            print("Error: 'aggregated_brain_scan_data.csv' not found. Set aggregate=False to build it.")
            return None
    else:
        # --- BUILD DATA FROM SCRATCH (aggregate=False) ---
        for participant_id in demographic_data['participant_id']:
            
            participant_id = str(participant_id)
            print(f"\nProcessing participant: {participant_id}")
            
            # Load the FreeSurfer data for the current participant
            subject_df = load_fsdata(participant_id, local_cache_dir)
            
            if subject_df is not None:
                brain_scan_data_list.append(subject_df)
                
        # Concatenate all collected DataFrames into a single one
        if brain_scan_data_list:
            brain_scan_data = pd.concat(brain_scan_data_list, ignore_index=True)
        else:
            print("No brain scan data was loaded. Cannot merge.")
            return None
            
        # This aligns the demographics key ('subject_id') with the loaded data key ('subject_id')
        demographic_data.rename(columns={'participant_id': 'subject_id'}, inplace=True)
            
    # --- Final Merge ---
    
    if aggregate:
        merged_df = pd.merge(demographic_data, brain_scan_data, on='participant_id')
    else:
        merged_df = pd.merge(demographic_data, brain_scan_data, on='subject_id')
    
    # --- Final Data Saving ---
    
    if aggregate:
        output_filename = 'merged_data_aggregated.csv'
        merged_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully merged and saved data to: {output_filename} (CSV)")
    else:
        zarr_path = 'merged_data.zarr'
        # Assuming merged_df.to_xarray() works and necessary libraries (xarray) are installed
        merged_df.to_xarray().to_zarr(zarr_path, mode='w')
        print(f"\nSuccessfully merged and saved data to: {zarr_path} (Zarr)")
        
    return merged_df

def main():
    final_df = merge_data(aggregate=True)
    print(final_df.head())

if __name__ == "__main__":
    main()