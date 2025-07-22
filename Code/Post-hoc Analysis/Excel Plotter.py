import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_joint_bar_graph(sheet_data, sheet_name, output_folder):
    try:
        # Handle merged cells and convert labels to strings
        sheet_data.iloc[:, 0] = sheet_data.iloc[:, 0].fillna(method='ffill').astype(str)
        
        # Validate column structure (corrected to 5 columns)
        if sheet_data.shape[1] < 5:
            print(f"Error in {sheet_name}: Need 5 columns (Labels, Val1, Err1, Val2, Err2)")
            return

        # Extract data components with corrected indices
        labels = sheet_data.iloc[:, 0]
        
        # Correct column indices based on user specification
        vals_group1 = pd.to_numeric(sheet_data.iloc[:, 1], errors='coerce')  # First value column
        err_group1 = pd.to_numeric(sheet_data.iloc[:, 2], errors='coerce')   # First error column
        vals_group2 = pd.to_numeric(sheet_data.iloc[:, 3], errors='coerce')  # Third value column
        err_group2 = pd.to_numeric(sheet_data.iloc[:, 4], errors='coerce')   # Fourth error column

        # Drop rows with NaN values
        valid_rows = ~(
            vals_group1.isna() | 
            err_group1.isna() | 
            vals_group2.isna() | 
            err_group2.isna()
        )
        
        labels = labels[valid_rows]
        vals_group1 = vals_group1[valid_rows]
        err_group1 = err_group1[valid_rows]
        vals_group2 = vals_group2[valid_rows]
        err_group2 = err_group2[valid_rows]

        # Manual legend configuration
        legend_labels = ["Subject-Specific", "Subject-Independent"]
        colors = ['#808080', '#d3d3d3']  # Grey and Orange

        # Plot configuration
        bar_width = 0.35
        x = np.arange(len(labels))
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        
        # Plot first group (corrected color and label indexing)
        ax.bar(x - bar_width/2, vals_group1, bar_width,
               color=colors[0], yerr=err_group1,
               capsize=5, label=legend_labels[0],
               error_kw={'elinewidth': 1.5})
        
        # Plot second group (corrected color and label indexing)
        ax.bar(x + bar_width/2, vals_group2, bar_width,
               color=colors[1], yerr=err_group2,
               capsize=5, label=legend_labels[1],
               error_kw={'elinewidth': 1.5})

        # Axis formatting with increased font size
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha='center', 
                          fontsize=20)  # Increased by 4 points
        ax.set_ylabel("Coefficient of Determination", 
                     fontsize=22)  # Increased by 4 points
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis='y', labelsize=18)  # Increased by 4 points
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        # Legend configuration with increased font size
        ax.legend(fontsize=18, framealpha=0.9, 
                 loc='upper right', bbox_to_anchor=(1.18, 1),
                 edgecolor='black')

        # Save output
        plt.tight_layout()
        output_path = os.path.join(output_folder, f'{sheet_name}_comparison.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Successfully created plot: {output_path}")

    except Exception as e:
        print(f"Plotting error in {sheet_name}: {str(e)}")

def process_excel_file(filename, output_folder):
    # Validate file existence
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        return

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    try:
        xls = pd.ExcelFile(filename)
        
        # Process all sheets after the first
        for sheet_name in xls.sheet_names[1:]:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            plot_joint_bar_graph(df, sheet_name, output_folder)
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")

# Example usage with path validation
excel_file = r"C:\Users\parth\OneDrive\Desktop\Thesis\MLR Plots\Feature_Plots_1\AE86AE10.xlsx"
output_dir = r"C:\Users\parth\OneDrive\Desktop\Thesis\MLR Plots\Feature_Plots_2"

# Verify file existence before processing
if os.path.exists(excel_file):
    process_excel_file(excel_file, output_dir)
else:
    print(f"Critical Error: Input file not found at {excel_file}")
