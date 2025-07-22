import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import mne
import os
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import skew
import matplotlib.pyplot as plt

# --- Input Data Information --- #

file_paths = # [Sample for Decoder Model File]
# --- UTILITY FUNCTIONS --- #

# Function to load the EEG and kinetic data from the .mat file
def load_kin_eeg_data(file_path):
    mat_data = loadmat(file_path)
    
    # Check available keys in the .mat file
    print("Available keys in the .mat file:", mat_data.keys())
    
    if 'hs' in mat_data:
        hs_data = mat_data['hs'][0, 0]  # Access the structured array
        
        # Print available fields in 'hs'
        print("Available fields in 'hs':", hs_data.dtype.names)
        
        # Access 'eeg' field in 'hs' and inspect it
        if 'eeg' in hs_data.dtype.names:
            eeg_data = hs_data['eeg'][0, 0]
            print("Available fields in 'eeg':", eeg_data.dtype.names)  # Inspect 'eeg' structure

            # Access the signal ('sig') and sampling rate
            eeg_signal = eeg_data['sig']  # Use 'sig' for EEG data
            sampling_rate = eeg_data['samplingrate'][0][0]  # Access sampling rate
            
            # Assuming channel names are present in 'eeg' under 'names'
            if 'names' in eeg_data.dtype.names:
                eeg_names = [name[0] for name in eeg_data['names'][0]]
            else:
                eeg_names = [f"EEG-{i+1}" for i in range(eeg_signal.shape[1])]
        else:
            raise KeyError("'eeg' field not found in 'hs'.")
        
        # Load kinetic data (assuming it's stored in 'kin' field of 'hs')
        if 'kin' in hs_data.dtype.names:
            kin_data = hs_data['kin'][0, 0]
            if 'sig' in kin_data.dtype.names:
                kin_sig_data = kin_data['sig']  # Assuming kinetic data is under 'sig'
            else:
                raise KeyError("'sig' field not found in 'kin' structure.")
            
            # Assuming there are names for the kin channels
            if 'names' in kin_data.dtype.names:
                kin_names = [name[0] for name in kin_data['names'][0]]
            else:
                kin_names = [f"Kin-{i+1}" for i in range(kin_sig_data.shape[1])]

            # **Modification starts here**
            # Define desired channels
            desired_channels = [
                'FX1 - force x plate 1', 'FX2 - force x plate 2',
                'FY1 - force y plate 1', 'FY2 - force y plate 2',
                'FZ1 - force z plate 1', 'FZ2 - force z plate 2'
            ]
            # Find indices of desired channels
            indices = [i for i, name in enumerate(kin_names) if name in desired_channels]
            if not indices:
                raise ValueError("None of the desired channels found in kinetic data.")
            # Select only the desired channels
            kin_sig_data = kin_sig_data[:, indices]
            kin_names = [kin_names[i] for i in indices]

            # Calculate the total load force (sum of the 6 sensor readings) for each timestamp
            total_load_force = np.sum(kin_sig_data, axis=1)
            # **Modification ends here**

        else:
            raise KeyError("'kin' field not found in 'hs'.")
    else:
        raise KeyError("'hs' key not found in the .mat file.")
    
    return total_load_force, eeg_signal, float(sampling_rate), kin_names, eeg_names

# --- Main Methodology + Regressor Pseudo-Online Evaluation Loop --- #

# Loop through each file and process data
for file_path in file_paths:
    try:
        # Load kinetic and EEG data
        total_load_force, eeg_signal, sampling_rate, kin_names, eeg_names = load_kin_eeg_data(file_path)

        # Ensure EEG data is a 2D array
        if eeg_signal.ndim != 2:
            raise ValueError("EEG data is not properly formatted. Expected a 2D array.")

        print(f"Processing file: {file_path}")
        print(f"EEG data shape: {eeg_signal.shape}")
        print(f"Window size: {int(0.1 * sampling_rate)}")

        # Print the channel names from both kin and eeg
        print(f"Kinetic Channel Names: {kin_names}")
        print(f"EEG Channel Names: {eeg_names}")

        # Create Sliding Windows (100 ms with 50 ms increment)
        window_size = int(0.1 * sampling_rate)  # Convert 100 ms to samples
        step_size = int(0.05 * sampling_rate)  # Increment size

        # Calculate windows from EEG data
        windows = [eeg_signal[i:i + window_size, :] for i in range(0, eeg_signal.shape[0] - window_size + 1, step_size)]

        # Check if windows are generated
        if not windows:
            raise ValueError("EEG data is empty or too short to create windows.")

        # Resample kinetic data to match the number of EEG windows
        num_windows = len(windows)
        if total_load_force.shape[0] != num_windows:
            print(f"Resampling total load force from {total_load_force.shape[0]} to {num_windows} to match EEG windows.")
            total_load_resampled = resample(total_load_force, num_windows)
        else:
            total_load_resampled = total_load_force

        # Reshape total_load_resampled for regression
        total_load_resampled = total_load_resampled.reshape(-1, 1)
        print(f"Resampled Total Load Force shape: {total_load_resampled.shape}")

        # Helper function to extract statistical features
        def extract_statistical_features(window):
            mean_val = np.mean(window, axis=0)
            mean_abs = np.mean(np.abs(window), axis=0)
            auc = np.trapz(window, axis=0)  # Area under the curve
            skewness = skew(window, axis=0)

            # Combine statistical features into a single array
            stats_features = np.hstack([mean_val, mean_abs, auc, skewness])

            return stats_features

        # Extract Statistical Features for each window
        stat_features = []

        for window in windows:
            stats = extract_statistical_features(window)
            stat_features.append(stats)

        stat_features = np.array(stat_features)

        # Check the shape of the feature matrix
        print("Statistical features shape:", stat_features.shape)

        # Apply SimpleImputer to handle NaN values in 'stat_features'
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        # Create a pipeline with PCA and Linear Regression
        pipeline = Pipeline(steps=[
            ('imputer', imputer),
            ('pca', PCA(n_components=0.95)),  # Retain 95% of variance
            ('regressor', MultiOutputRegressor(LinearRegression()))
        ])

        # Train-test split (e.g., last 100 samples as test set)
        X_train, y_train = stat_features[:-100], total_load_resampled[:-100]  # Training data
        X_test, y_test = stat_features[-100:], total_load_resampled[-100:]  # Testing data

        # Fit the model
        pipeline.fit(X_train, y_train)

        # Predict kinetic data
        y_pred = pipeline.predict(X_test)

        r_squared = r2_score(y_test, y_pred)
        print(f"R-squared (R²) for file {file_path}: {r_squared}")

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error (MSE) for file {file_path}: {mse}")

        # **Save results in an Excel file**

        # Flatten y_pred and y_test for saving
        y_pred_flatten = y_pred.flatten()
        y_test_flatten = y_test.flatten()

        # Calculate errors and handle any division by zero in percentage error
        error = y_pred_flatten - y_test_flatten
        mean_absolute_error = np.abs(error)

        # Avoid division by zero by replacing zeros in y_test_flatten with small values
        y_test_flatten_safe = np.where(y_test_flatten == 0, 1e-5, y_test_flatten)
        mean_percentage_error = (mean_absolute_error / np.abs(y_test_flatten_safe)) * 100

        # Create a DataFrame to store results
        result_df = pd.DataFrame({
            'Predicted Value': y_pred_flatten,
            'Actual Value': y_test_flatten,
            'Error': error,
            'Mean Absolute Error': mean_absolute_error,
            'Mean Percentage Error': mean_percentage_error,
            'R-squared (R²)': [r_squared] * len(y_pred_flatten)
            })

        # Save to Excel
        output_excel_path = os.path.join(os.path.dirname(file_path), f"results_{os.path.basename(file_path).replace('.mat', '')}.xlsx")
        result_df.to_excel(output_excel_path, index=False)
        print(f"Results saved to: {output_excel_path}")

        # Plot Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual Total Load Force', color='blue')
        plt.plot(y_pred, label='Predicted Total Load Force', color='red', linestyle='--')
        plt.xlabel('Samples')
        plt.ylabel('Total Load Force')
        plt.title(f'Actual vs Predicted Total Load Force (R² = {r_squared:.4f})')
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {str(e)}")
