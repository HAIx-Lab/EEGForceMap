import numpy as np
import pandas as pd
from scipy.io import loadmat
import mne
from mne.time_frequency import psd_array_welch
from mne.preprocessing import ICA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.integrate import simps
from scipy.stats import skew
import os
import re
from mne_icalabel import label_components  # Import ICLabel
import matplotlib.pyplot as plt
from openpyxl import load_workbook

print('Script Started')

# --- Input Data Information --- #

file_paths = # [Sample for Decoder Model File]

output_folder = # Folder to Store the Results
output_excel = # Excel File to Store the Model Performance
r2_mse_excel = # Excel File to Store Evaluation Matrices

# Select specific EEG channels for processing
selected_eeg_channels = ['C3', 'CP1', 'CP2', 'Cz', 'FC2', 'FC6', 'Fp1', 'P3', 'C4'] # Channels Resembling the EEG activity

# --- UTILITY FUNCTIONS --- #

# Functions to load the EEG and kinetic data from the .mat file
def load_kin_eeg_data(file_path):
    try:
        mat_data = loadmat(file_path)
        
        if 'hs' in mat_data:
            hs_data = mat_data['hs'][0, 0]
            if 'eeg' in hs_data.dtype.names:
                eeg_data = hs_data['eeg'][0, 0]
                eeg_signal = eeg_data['sig']  
                sampling_rate = eeg_data['samplingrate'][0][0]  
                
                if 'names' in eeg_data.dtype.names:
                    eeg_names = [name[0] for name in eeg_data['names'][0]]
                else:
                    eeg_names = [f"EEG-{i+1}" for i in range(eeg_signal.shape[1])]

                if eeg_signal.ndim == 1:
                    eeg_signal = eeg_signal[:, np.newaxis]

            if 'kin' in hs_data.dtype.names:
                kin_data = hs_data['kin'][0, 0]
                if 'sig' in kin_data.dtype.names:
                    kin_sig_data = kin_data['sig']  

                    if kin_sig_data.ndim == 1:
                        kin_sig_data = kin_sig_data[:, np.newaxis]

                if 'names' in kin_data.dtype.names:
                    kin_names = [name[0] for name in kin_data['names'][0]]
                else:
                    kin_names = [f"Kin-{i+1}" for i in range(kin_sig_data.shape[1])]


        else:
            raise KeyError("'hs' key not found in the .mat file.")
        
        return kin_sig_data, eeg_signal, float(sampling_rate), kin_names, eeg_names
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise
def sanitize_filename(name):
    # Replace spaces and special characters with underscores
    return re.sub(r'[^A-Za-z0-9]+', '_', name)

# Function to compute total load force
def compute_total_load_force(kin_sig_data):
    force_indices = [i for i, name in enumerate(kin_names) if 'force' in name.lower()]
    force_data = kin_sig_data[:, force_indices] if force_indices else None

    if force_data is not None:
        total_load_force = np.sum(force_data, axis=1)
    else:
        total_load_force = None
        print("No 'force' channels found in kinetic data.")
    return total_load_force

# Function to save results
def save_results(y_test, y_pred, kinetic_channel_name, output_dir):
    try:
        r2 = r2_score(y_test, y_pred.flatten())
        mse = mean_squared_error(y_test, y_pred.flatten())

        sanitized_channel_name = sanitize_filename(kinetic_channel_name)
        results_df = pd.DataFrame({
            'Kinetic Channel': [kinetic_channel_name] * len(y_test),
            'Predicted Value': y_pred.flatten(),
            'Actual Value': y_test,
            'Error': np.abs(y_test - y_pred.flatten()),
            'Mean Absolute Error (Percentage)': 100 * np.abs(y_test - y_pred.flatten()) / np.maximum(np.abs(y_test), 1e-5)
        })

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_excel = os.path.join(output_dir, f"{sanitized_channel_name}_results.xlsx")
        r2_mse_excel = os.path.join(output_dir, "R2_MSE_results.xlsx")
        
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name=kinetic_channel_name, index=False)

        r2_mse_df = pd.DataFrame({
            'Kinetic Channel': [kinetic_channel_name],
            'R²': [r2],
            'MSE': [mse]
        })

        if not os.path.exists(r2_mse_excel):
            with pd.ExcelWriter(r2_mse_excel, engine='openpyxl') as writer:
                r2_mse_df.to_excel(writer, sheet_name='R2_MSE', index=False)
        else:
            with pd.ExcelWriter(r2_mse_excel, mode='a', engine='openpyxl') as writer:
                r2_mse_df.to_excel(writer, sheet_name='R2_MSE', index=False, header=None, startrow=writer.sheets['R2_MSE'].max_row)
        
        print(f"Results saved to {output_excel} for channel {kinetic_channel_name}")
    except Exception as e:
        print(f"Error saving results: {e}")
        raise

# --- Main Methodology + Regressor Pseudo-Online Evaluation Loop --- #

# Function to apply ICA and remove muscle, heart, and other components using ICLabel
def remove_artifacts(eeg_signal, sampling_rate, eeg_channel_names):
    try:
        if eeg_signal.ndim == 1:
            eeg_signal = eeg_signal[:, np.newaxis]
        
        info = mne.create_info(ch_names=eeg_channel_names, sfreq=sampling_rate, ch_types=['eeg'] * len(eeg_channel_names))
        raw = mne.io.RawArray(eeg_signal.T, info)
        
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)

        raw.set_eeg_reference('average', projection=True)
        raw.filter(1., 100.)

        ica = ICA(n_components=9, method='infomax', fit_params=dict(extended=True), random_state=97, max_iter=800)
        ica.fit(raw)

        iclabel_labels = label_components(raw, ica, 'iclabel')
        labels = iclabel_labels['labels']

        components_to_remove = np.where(np.isin(labels, ['muscle', 'heart', 'other']))[0]
        
        if len(components_to_remove) > 0:
            ica.exclude = components_to_remove
            raw_ica_clean = ica.apply(raw.copy())
            cleaned_eeg_signal = raw_ica_clean.get_data().T
        else:
            cleaned_eeg_signal = eeg_signal

        return cleaned_eeg_signal
    except Exception as e:
        print(f"Error during artifact removal: {e}")
        raise

# Function to compute the area under the curve (AUC) for each channel
def compute_auc(eeg_window, sampling_rate):
    auc_vals = []
    for channel_data in eeg_window.T:
        auc = simps(channel_data, dx=1/sampling_rate)
        auc_vals.append(auc)
    return np.array(auc_vals)

# Function to compute statistical features from an EEG window
def extract_statistical_features(eeg_window, sampling_rate):
    try:
        if eeg_window.ndim == 1:
            eeg_window = eeg_window[:, np.newaxis]

        mean_vals = np.mean(eeg_window, axis=0)
        var_vals = np.var(eeg_window, axis=0)
        mav_vals = np.mean(np.abs(eeg_window), axis=0)
        auc_vals = compute_auc(eeg_window, sampling_rate)
        skew_vals = skew(eeg_window, axis=0)

        stat_features = np.hstack([mean_vals, var_vals, mav_vals, auc_vals, skew_vals])
        return stat_features

    except Exception as e:
        print(f"Error during statistical feature extraction: {e}")
        raise






# Main loop
for file_path in file_list:
    try:
        output_dir = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0])
        
        kin_sig, eeg_signal, sampling_rate, kin_names, eeg_names = load_kin_eeg_data(file_path)

        if kin_sig is None or eeg_signal is None:
            print(f"Skipping {file_path} due to missing data")
            continue

        eeg_cleaned = remove_artifacts(eeg_signal, sampling_rate, eeg_names)
        total_load_force = compute_total_load_force(kin_sig)
        y = total_load_force

        window_size = int(1 * sampling_rate)
        num_windows = eeg_cleaned.shape[0] // window_size

        X = []
        for window_idx in range(num_windows):
            window_start = window_idx * window_size
            window_end = window_start + window_size
            
            eeg_window = eeg_cleaned[window_start:window_end, :]
            if eeg_window.shape[0] == window_size:
                features = extract_statistical_features(eeg_window, sampling_rate)
                X.append(features)

        X = np.array(X)
        if X.shape[0] > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y[:X.shape[0]], test_size=0.2, random_state=42)
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            y_pred = linear_model.predict(X_test)

            save_results(y_test, y_pred, 'Total Load Force', output_dir)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

print('Script Completed')
