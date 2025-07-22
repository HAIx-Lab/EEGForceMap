import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import os
import numpy as np

# Function to add a trendline
def add_trendline(x, y, ax):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), color='r', linestyle='--', label='Trendline')
    ax.legend(fontsize=14)

# Function to handle missing values
def handle_missing_values(data, strategy='drop'):
    """
    Handle missing values in the dataset.
    :param data: Input DataFrame.
    :param strategy: 'drop' to drop rows with NaN, 'impute' to replace NaN with mean.
    :return: Processed DataFrame.
    """
    if strategy == 'drop':
        return data.dropna()
    elif strategy == 'impute':
        imputer = SimpleImputer(strategy='mean')
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    else:
        raise ValueError("Invalid strategy. Use 'drop' or 'impute'.")

# Function to make the first column absolute
def make_first_column_absolute(data):
    """
    Make the first column of the DataFrame absolute.
    :param data: Input DataFrame.
    :return: DataFrame with the first column values made absolute.
    """
    data.iloc[:, 0] = data.iloc[:, 0].abs()
    return data

# Function to plot PCA components with highest correlation (for first sheet)
def plot_pca_scatter(sheet_data, sheet_name, output_folder):
    # Handle missing values
    sheet_data = handle_missing_values(sheet_data, strategy='impute')  # Choose 'drop' or 'impute'
    
    # Make the first column absolute
    sheet_data = make_first_column_absolute(sheet_data)

    x = sheet_data.iloc[:, 0]
    x_label = sheet_data.columns[0]

    features = sheet_data.iloc[:, 1:]
    pca = PCA(n_components=min(10, features.shape[1]))
    pca_components = pca.fit_transform(features)

    correlations = [pearsonr(x, pca_components[:, i])[0] for i in range(pca_components.shape[1])]
    top_two_indices = np.argsort(np.abs(correlations))[-10:]

    for i in top_two_indices:
        y = pca_components[:, i]
        y_label = f'PCA Component {i+1} (Corr: {correlations[i]:.2f})'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, color='b')
        ax.set_xlabel(x_label, fontsize=18)
        ax.set_ylabel(y_label, fontsize=18)
        ax.set_title(f'Scatter Plot: {x_label} vs {y_label}', fontsize=18)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=14)
        add_trendline(x, y, ax)
        
        ax.text(0.05, 0.95, f"Pearson's r: {correlations[i]:.2f}", transform=ax.transAxes, 
                fontsize=14, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        output_path = os.path.join(output_folder, f'{sheet_name}_PCA_Component_{i+1}_scatter.png')
        plt.savefig(output_path)
        plt.show()

# Function to plot raw feature correlations (for second sheet)
def plot_feature_scatter(sheet_data, sheet_name, output_folder):
    # Handle missing values
    sheet_data = handle_missing_values(sheet_data, strategy='impute')  # Choose 'drop' or 'impute'
    
    # Make the first column absolute
    sheet_data = make_first_column_absolute(sheet_data)

    x = sheet_data.iloc[:, 0]
    x_label = sheet_data.columns[0]

    for col in sheet_data.columns[1:]:
        y = sheet_data[col]
        y_label = col
        corr, _ = pearsonr(x, y)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y, color='b')  # Different color for distinction
        ax.set_xlabel(x_label, fontsize=18)
        ax.set_ylabel(y_label, fontsize=18)
        ax.set_title(f'Scatter Plot: {x_label} vs {y_label}', fontsize=18)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=14)
        add_trendline(x, y, ax)
        
        ax.text(0.05, 0.95, f"Pearson's r: {corr:.2f}", transform=ax.transAxes, 
                fontsize=14, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        output_path = os.path.join(output_folder, f'{sheet_name}_{y_label}_scatter.png')
        plt.savefig(output_path)
        plt.show()

# Load and process Excel file
def process_excel_file(filename, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    xls = pd.ExcelFile(filename)
    
    # First sheet: PCA scatter plots
    sheet1_data = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    plot_pca_scatter(sheet1_data, xls.sheet_names[0], output_folder)
    
    # Second sheet: Feature scatter plots
    if len(xls.sheet_names) > 1:
        sheet2_data = pd.read_excel(xls, sheet_name=xls.sheet_names[1])
        plot_feature_scatter(sheet2_data, xls.sheet_names[1], output_folder)
    else:
        print("The Excel file does not contain a second sheet.")

# Example usage
excel_filename = r"C:\Users\parth\OneDrive\Desktop\Thesis\MLR Plots\Feature_Plots_1\All_Runs_Features_List_1.xlsx"
output_folder = r"C:\Users\parth\OneDrive\Desktop\Thesis\MLR Plots\Feature_Plots"
process_excel_file(excel_filename, output_folder)