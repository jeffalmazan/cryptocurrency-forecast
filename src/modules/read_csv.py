import pandas as pd
from pathlib import Path
import os

def read_data():
    cwd = Path(os.getcwd())
    print("Current Path:", cwd)
    
    data_dir = cwd / 'data/Datasets'
    print("Data Directory:", data_dir)

    if not data_dir.exists():
        print(f"The directory {data_dir} does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if the directory doesn't exist

    renamed_dfs = []
    for data_file in data_dir.glob('*.csv'):
        print(f"Reading file: {data_file}")
        df = pd.read_csv(data_file, skiprows=1)
        
        volume_columns = [col for col in df.columns if 'Volume' in col]
        if volume_columns:
            df.rename(columns={volume_columns[0]: 'Crypto Volume'}, inplace=True)
        
        renamed_dfs.append(df)

    concatenated_df = pd.concat(renamed_dfs, ignore_index=True)
    return concatenated_df

