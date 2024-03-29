import pandas as pd
# from read_csv import read_data
import time

# df = read_data()



def perform_cleanup(df):
    start_time = time.time()

    print("\nChecking for Null Values")
    null_counts = df.isnull().sum()

    # Print out the null value counts for each column
    if null_counts.sum() > 0:
        print("There are null values.")
        for column, count in null_counts.items():
            if count > 0:
                print(f"{column}: {count} null values")
    
    # Separate numerical and categorical data
    print("Separating Numerical and Categorical Data")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    print("Separation Complete")
    
    # Interpolate numerical columns
    print("Starting Interpolation for Numerical Columns")
    df[numerical_columns] = df[numerical_columns].interpolate(method='linear', limit_direction='forward', axis=0)
    print("Completed Interpolation for Numerical Columns")

    # Fill categorical columns
    if len(categorical_columns) > 0:
        print("Starting Fill for Categorical Columns")
        df[categorical_columns] = df[categorical_columns].fillna(method='ffill').fillna(method='bfill')
        print("Completed Fill for Categorical Columns")

    # Print out null value counts again to verify there are no more nulls
    print("\nNull values in each column after cleanup:")
    null_counts = df.isnull().sum()
    print(null_counts)

    # Checking for duplicate values
    print("\nChecking for Duplicate Values")
    num_records_before = len(df)
    duplicated_rows = df[df.duplicated(keep=False)]
    num_duplicated = len(duplicated_rows)

    # Dropping duplicate values
    if num_duplicated > 0:
        print(f"\nDuplicated rows: {num_duplicated}")
        print(duplicated_rows)
        df = df.drop_duplicates()
    else:
        print("\nNo duplicate rows found.")

    num_records_after = len(df)

    print(f"\nNumber of records before removing duplicates: {num_records_before}")
    print(f"Number of records after removing duplicates: {num_records_after}")

    print(f"Cleanup completed in {time.time() - start_time} seconds")

    return df

# perform_cleanup(df)

def handle_outliers(df):
    
    start_time = time.time()


    # Automatically select numerical columns in the DataFrame
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Initialize an empty DataFrame to hold the cleaned data
    df_no_outliers = pd.DataFrame()

    # Process each group separately
    for symbol, group in df.groupby('Symbol'):
        print(f"\nProcessing {symbol}...")
        cleaned_group = group.copy()

        for column in numerical_columns:
            print(f"\nColumn: {column}")

            # Display statistical summary before outlier removal
            print("Before outlier removal:")
            print(group[column].describe())

            # Calculate Q1, Q3, and IQR
            Q1 = group[column].quantile(0.25)
            Q3 = group[column].quantile(0.75)
            IQR = Q3 - Q1

            # Define the outlier boundaries
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Filter out outliers within the group
            filtered_group = cleaned_group[(cleaned_group[column] >= lower_bound) & (cleaned_group[column] <= upper_bound)]
            
            # Update the cleaned group
            cleaned_group = filtered_group

            # Display statistical summary after outlier removal
            print("After outlier removal:")
            print(filtered_group[column].describe())

        # Append the cleaned group to the df_no_outliers DataFrame
        df_no_outliers = pd.concat([df_no_outliers, cleaned_group], axis=0)

    # Print the number of records before and after removing outliers
    print("\nOverall results:")
    print("Number of records before removing outliers:", len(df))
    print("Number of records after removing outliers:", len(df_no_outliers))

    # Record the end time and calculate runtime
    end_time = time.time()
    runtime = end_time - start_time
    print("\nRuntime[handle_outliers]:", runtime, "seconds\n")

    return df_no_outliers


# df = handle_outliers(df)
