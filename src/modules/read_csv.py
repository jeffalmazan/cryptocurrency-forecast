import pandas as pd
from pathlib import Path
import os

def read_data():
    # Get the current working directory as a Path object
    cwd = Path(os.getcwd())
    print("Current Path:", cwd)
    
    # Define the data directory relative to the current working directory
    data_dir = cwd / 'data/Datasets'
    print("Data Directory:", data_dir)

    # Check if the data directory exists
    if not data_dir.exists():
        print(f"The directory {data_dir} does not exist.")
        # Return an empty DataFrame if the directory doesn't exist
        return pd.DataFrame()

    # List to store dataframes for concatenation
    renamed_dfs = []
    # Iterate over each CSV file in the data directory
    for data_file in data_dir.glob('*.csv'):
        print(f"Reading file: {data_file}")
        # Read the CSV file, skipping the first row (often header row)
        df = pd.read_csv(data_file, skiprows=1)
        
        # Find columns that contain the substring 'Volume' and consider them as volume columns
        volume_columns = [col for col in df.columns if 'Volume' in col]
        # If there is at least one 'Volume' column, rename the first one found to 'Crypto Volume'
        if volume_columns:
            df.rename(columns={volume_columns[0]: 'Crypto Volume'}, inplace=True)
        
        # Append the modified dataframe to the list
        renamed_dfs.append(df)

    # Concatenate all dataframes in the list into a single dataframe
    concatenated_df = pd.concat(renamed_dfs, ignore_index=True)
    # Return the concatenated dataframe
    return concatenated_df
