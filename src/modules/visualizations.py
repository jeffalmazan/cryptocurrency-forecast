import matplotlib.pyplot as plt
import seaborn as sns
# from read_csv import read_data
# from preprocessing import perform_cleanup

# df = read_data()
# df = perform_cleanup(df)


def crypto_visualization(df):
    symbols = df['Symbol'].unique()  # Get the unique symbols for their own graph

    for symbol in symbols:
        df_symbol = df[df['Symbol'] == symbol]  # Filter the DataFrame for the current symbol

        # Selecting only umerical columns (excluding 'Unix' if present)
        numerical_columns = df_symbol.select_dtypes(include=['float64', 'int64']).columns.drop('Unix', errors='ignore')

        
        num_cols = len(numerical_columns)
        # Calculate the number of rows needed for subplots based on the number of columns
        num_rows = num_cols // 2 if num_cols % 2 == 0 else (num_cols // 2) + 1

        # Create boxplots for each numerical column
        plt.figure(figsize=(20, num_rows * 5))
        plt.suptitle(f'Boxplots for {symbol}', fontsize=20)
        for i, col in enumerate(numerical_columns):
            plt.subplot(num_rows, 2, i + 1)
            sns.boxplot(y=col, data=df_symbol)
            plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

        # Creating Histogram
        plt.figure(figsize=(20, num_rows * 5))
        plt.suptitle(f'Histograms for {symbol}', fontsize=20)
        for i, col in enumerate(numerical_columns):
            plt.subplot(num_rows, 2, i + 1)
            plt.hist(df_symbol[col], bins=80, edgecolor='black')
            plt.title(f'Histogram of {col}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.yscale('log') 
            
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(df):
    # excluded_columns = ['Unix', 'Open', 'High', 'Low']  # Excluded columns
    excluded_columns = ['Unix']
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).drop(columns=excluded_columns, errors='ignore')
    
    # Calculate the correlation matrix
    corr = numerical_columns.corr()

    # Generate a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".6f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
    
    return df

# plot_correlation_heatmap(df)




