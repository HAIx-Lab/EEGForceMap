import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Function to calculate Fraction of Variance Accounted For (FVAF)
def calculate_fvaf(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot)

# Function to calculate Covariance
def calculate_covariance(x, y):
    return np.cov(x, y)[0, 1]

# Updated function to sample data from specific ranges
def sample_from_groups(data, group_ranges, sample_fraction):
    sampled_indices = []
    for lower, upper in group_ranges:
        group_indices = data[(data > lower) & (data <= upper)].index
        sample_size = max(1, int(len(group_indices) * sample_fraction))  # Ensure at least one sample
        sampled_indices.extend(np.random.choice(group_indices, size=sample_size, replace=False))
    return sampled_indices

def calculate_r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot)

# Function to plot predicted vs actual values
def plot_predicted_vs_actual(sheet_data, sheet_name, output_folder, full_sampling_rate=500):
    # Define group ranges
    group_ranges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    sample_fraction = 0.05  # Adjust sample fraction as needed

    # Treat first column as actual values
    actual_values = sheet_data.iloc[:, 0]
    actual_label = sheet_data.columns[0]

    # Loop through each subsequent column as predicted values
    for col in sheet_data.columns[1:]:
        predicted_values = sheet_data[col]
        predicted_label = col

        # Sample indices based on defined groups
        sampled_indices = sample_from_groups(actual_values, group_ranges, sample_fraction)

        # Use the same indices for actual and predicted values
        sampled_actual = actual_values.loc[sampled_indices]
        sampled_predicted = predicted_values.loc[sampled_indices]

        # Set y-axis limits based on min and max values across both actual and predicted values
        y_min = min(sampled_actual.min(), sampled_predicted.min())
        y_max = max(sampled_actual.max(), sampled_predicted.max())

        # Calculate R² (Coefficient of Determination)
        r2_value = calculate_r2(sampled_actual, sampled_predicted)

        # Create figure with scatter plot
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.scatter(sampled_actual, sampled_predicted, color='royalblue', alpha=0.7, label='Sampled Data')
        ax.plot([y_min, y_max], [y_min, y_max], 'r--', label='y = x', linewidth=2)  # Reference line
        ax.legend(loc='lower right', fontsize=22)
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.set_xlabel('Actual Values', fontsize=26)
        ax.set_ylabel('Predicted Values', fontsize=26)
        ax.set_title(f'{predicted_label} vs {actual_label}', fontsize=28)
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)

        # Add R² value as a text box
        textstr = f'$R^2$ = {r2_value:.4f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=22,
                verticalalignment='bottom', bbox=props)

        # Save the figure to the output folder
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{sheet_name}_{predicted_label}_vs_{actual_label}.png")
        plt.savefig(output_path)
        plt.close()

# Function to process a single sheet from an Excel file
def process_single_sheet(excel_filename, output_folder):
    xls = pd.ExcelFile(excel_filename)
    for sheet_name in xls.sheet_names:
        sheet_data = pd.read_excel(xls, sheet_name=sheet_name)
        plot_predicted_vs_actual(sheet_data, sheet_name, output_folder)

# Main code
excel_filename = r"C:\Users\parth\OneDrive\Desktop\Assignments\Term 3\Thesis\MLR Plots\MLR_5\nn_summary_results_HS_P1_S1_V4.xlsx"
output_folder = r"C:\Users\parth\OneDrive\Desktop\Assignments\Term 3\Thesis\MLR Plots"
process_single_sheet(excel_filename, output_folder)
