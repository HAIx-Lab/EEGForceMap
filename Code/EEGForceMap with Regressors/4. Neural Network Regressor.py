import os
import scipy.io
from scipy.io import loadmat
import numpy as np
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LeakyReLU
from sklearn.decomposition import PCA

# --- Input Data Information --- #

file_paths = # [Sample for Decoder Model File]
selected_channels = ['C3', 'CP1', 'CP2', 'Cz', 'FC2', 'FC6', 'Fp1', 'P3', 'C4'] # Channels Selected During Preprocessing
output_file = # Excel File Path to Store the Model Performance

# --- UTILITY FUNCTIONS --- #

# Functions to load the EEG and kinetic data from the .mat file
def flatten_nested_list(nested_list):
    """Recursively flattens a nested list."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, (list, np.ndarray)):
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)
    return flat_list
def load_kin_eeg_data(mat_file):
    # Load the .mat file
    mat_contents = scipy.io.loadmat(mat_file)
    
    if 'hs' in mat_contents:
        hs_data = mat_contents['hs'][0, 0]

        # Extract the field names of hs_data
        hs_fields = hs_data.dtype.names
        print(f"Fields in 'hs' data: {hs_fields}")
        
        # Check and extract kinetic data
        if 'kin' in hs_fields:
            kin_struct = hs_data['kin'][0, 0]
            print(f"Kinetic structure: {kin_struct}")
            if 'sig' in kin_struct.dtype.names and 'names' in kin_struct.dtype.names:
                kin_data = kin_struct['sig']  # Assuming 'sig' contains the kinetic signals
                
                # Check the structure of kin_data
                if isinstance(kin_data, np.ndarray):
                    print(f"Kinetic data shape: {kin_data.shape}")
                else:
                    print(f"Kinetic data has an unexpected format: {type(kin_data)}")
                    return None, None, None, None, None
                
                kin_names = kin_struct['names']  # Assuming 'names' field contains the kinetic channel names
                if isinstance(kin_names, np.ndarray) and len(kin_names) > 0:
                    # Flatten the nested list
                    kin_names = flatten_nested_list(kin_names)
                    # Convert all names to strings and strip any whitespace
                    kin_names = [str(name).strip() for name in kin_names]
                    print(f"Kinetic data channels after flattening: {kin_names}")
                else:
                    print(f"Unexpected structure for kinetic channel names in {mat_file}")
                    return None, None, None, None, None
                
            else:
                print(f"Unexpected structure for kinetic data in {mat_file}")
                return None, None, None, None, None
        else:
            print(f"Kinetic data not found in {mat_file}")
            return None, None, None, None, None

        # Check and extract EEG data
        if 'eeg' in hs_fields:
            eeg_struct = hs_data['eeg'][0, 0]
            print(f"EEG structure: {eeg_struct}")
            if 'sig' in eeg_struct.dtype.names and 'names' in eeg_struct.dtype.names and 'samplingrate' in eeg_struct.dtype.names:
                eeg_data = eeg_struct['sig']  # Assuming 'sig' field contains EEG signals
                eeg_names = eeg_struct['names']  # Assuming 'names' field contains EEG channel names
                sfreq = 500
                print(f"EEG data found with shape: {eeg_data.shape}")
                
                # Process eeg_names similarly to kin_names
                if isinstance(eeg_names, np.ndarray) and len(eeg_names) > 0:
                    eeg_names = flatten_nested_list(eeg_names)
                    eeg_names = [str(name).strip() for name in eeg_names]
                    print(f"EEG channel names after flattening: {eeg_names}")
                else:
                    print(f"Unexpected structure for EEG channel names in {mat_file}")
                    return None, None, None, None, None
                
            else:
                print(f"Unexpected structure for EEG data in {mat_file}")
                return None, None, None, None, None
        else:
            print(f"EEG data not found in {mat_file}")
            return None, None, None, None, None
    else:
        print(f"'hs' structure not found in {mat_file}")
        return None, None, None, None, None

    return kin_data, eeg_data, sfreq, kin_names, eeg_names
def clean_name(name):
    """Clean channel name by removing invisible characters and making it uppercase."""
    return name.strip().replace('\u200b', '').replace('\u00a0', '').upper()

# Function to compute total load force
def extract_kinetic_data(kin_data, kin_names):
    if kin_data is None:
        print("Kinetic data is None, cannot extract force channels.")
        return None

    # Extract force channels based on kin_names
    force_channels = [name for name in kin_names if 'force' in name.lower()]
    
    if not force_channels:
        print("No force channels found in kinetic data.")
        return None
    
    # Find indices of force_channels
    force_indices = [i for i, name in enumerate(kin_names) if 'force' in name.lower()]
    
    if not force_indices:
        print("No matching force channels found based on kin_names.")
        return None

    # Ensure kin_data is a 2D array
    if kin_data.ndim == 1:
        kin_data = kin_data.reshape(-1, 1)

    # Extract the corresponding columns from kin_data
    try:
        load_force = kin_data[:, force_indices]
        print(f"Extracted force channels: {force_channels}")
        print(f"Load force shape: {load_force.shape}")
    except IndexError as e:
        print(f"Error extracting force channels: {e}")
        return None
    
    return load_force

# 7. Plot a time series graph of predicted vs actual load force
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual Load Force', color='blue')
    plt.plot(y_pred, label='Predicted Load Force', color='red')
    plt.legend()
    plt.title('Neural Network Predicted vs Actual Load Force')
    plt.xlabel('Timestamp')
    plt.ylabel('Load Force')
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()

    # Plot Mean Absolute Error
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Functions to Plot and Record the Output
def save_results_to_excel(y_test, y_pred, mse, r2, output_file):
    # Calculate error
    error = np.abs(y_test - y_pred)
    
    # Create a DataFrame with actual, predicted, and error values
    results_df = pd.DataFrame({
        'Actual Load Force': y_test.flatten(),
        'Predicted Load Force': y_pred.flatten(),
        'Error': error.flatten()
    })
    
    try:
        # Save the DataFrame to an Excel file
        results_df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Failed to save results to {output_file}: {e}")

    # Print MSE and R2
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

# --- Main Methodology + Regressor Pseudo-Online Evaluation Loop --- #

# Function to remove artifact using ICA and ICLabel
def remove_artifact(raw):
    # Fit ICA
    ica = ICA(n_components=9, random_state=97, max_iter=800)
    ica.fit(raw)
    
    # Apply ICLabel to label components
    labels = label_components(raw, ica, method='iclabel')

    # Find components labeled as muscle, heart, or other
    bad_components = []
    for idx, label in enumerate(labels['labels']):
        if label in ['muscle', 'heart', 'other']:
            bad_components.append(idx)

    # Remove the identified bad components
    raw_clean = ica.apply(raw, exclude=bad_components)
    
    print(f"Removed components: {bad_components}")
    
    return raw_clean

# Load EEG and Kinetic data
def load_and_preprocess_data(eeg_data, eeg_names, sfreq, selected_channels=None):    
    # Assign all_channel_names directly from eeg_names
    all_channel_names = eeg_names

    # Select all channels if 'selected_channels' is not provided or set to None
    if not selected_channels:
        selected_channels = all_channel_names

    # Clean channel names to ensure consistent comparison
    cleaned_all_channel_names = [clean_name(ch) for ch in all_channel_names]
    cleaned_selected_channels = [clean_name(ch) for ch in selected_channels]

    # Find indices of selected channels
    selected_indices = [i for i, ch_name in enumerate(cleaned_all_channel_names)
                        if ch_name in cleaned_selected_channels]

    # Debugging: Print selected indices and corresponding channel names
    print(f"Selected indices: {selected_indices}")
    print(f"Selected channels: {[all_channel_names[i] for i in selected_indices]}")

    # Extract the selected EEG data
    selected_eeg_data = eeg_data[:, selected_indices]

    # Debugging: Print shape of the selected EEG data
    print(f"Selected EEG data shape: {selected_eeg_data.shape}")

    # Continue the preprocessing steps and set montage
    info = mne.create_info(ch_names=[all_channel_names[i] for i in selected_indices], sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(selected_eeg_data.T, info)

    # Try setting a standard montage
    montage = mne.channels.make_standard_montage('standard_1020')

    # Use on_missing='ignore' to avoid errors for unmatched channels
    raw.set_montage(montage, on_missing='ignore')

        # Remove artifacts using ICA and ICLabel
    raw_clean = remove_artifact(raw)

    return raw_clean

# Extract ERDS features with 100 ms baseline
def extract_erds(eeg_data, sfreq, baseline_period=(0, 0.1)):
    baseline_samples = int(baseline_period[1] * sfreq)
    baseline = np.mean(eeg_data[:, :baseline_samples], axis=1, keepdims=True)
    erds_features = (eeg_data - baseline) / baseline  # Normalized ERDS
    return erds_features

# Apply a Neural Network for Linear Regression
def apply_neural_network(X_train, y_train, X_test, y_test):
    # Enhanced Neural Network for Regression with LeakyReLU, BatchNorm, Dropout, and Learning Rate Scheduler
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64),
        LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32),
        LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(16),
        LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Learning rate scheduler callback
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )

    # Implement Early Stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )

    # Train the model with validation split and callbacks
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate mean squared error and r2 score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Plot training history
    plot_predictions(y_test, y_pred)
    plot_training_history(history)
    
    return y_pred, mse, r2

# Full pipeline function for multiple files
def run_pipeline(file_list, selected_channels, output_file):
    all_results = []
    
    # Convert output_file to a Path object
    output_path = Path(output_file)
    
    # Ensure the main output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for mat_file in file_list:
        print(f"Processing file: {mat_file}")
        kin_data, eeg_data, sfreq, kin_names, eeg_names = load_kin_eeg_data(mat_file)
        
        if kin_data is None or eeg_data is None:
            print(f"Skipping file {mat_file} due to missing data.")
            continue

        # Preprocess EEG data
        raw_clean = load_and_preprocess_data(eeg_data, eeg_names, sfreq, selected_channels)

        # Convert EEG data to numpy array for ERDS extraction
        eeg_data_clean = raw_clean.get_data()

        # Extract ERDS features
        erds_features = extract_erds(eeg_data_clean, sfreq)
        
        # Proceed with further steps only if kin_data is valid
        load_force = extract_kinetic_data(kin_data, kin_names)
        if load_force is None:
            print(f"Skipping file {mat_file} due to missing force channels.")
            continue
        
        # Debugging: Print shape before selection
        print(f"Load force shape before selection: {load_force.shape}")

        # Ensure single-output by selecting the first force channel
        if load_force.ndim > 1 and load_force.shape[1] > 1:
            print(f"Multiple force channels found in {mat_file}, selecting the first one for regression.")
            load_force = load_force[:, 0]  # Select the first column
        elif load_force.ndim == 1:
            load_force = load_force  # Already single-dimensional
        else:
            load_force = load_force[:, 0]  # Handle unexpected cases

        # Reshape load_force to be 2D (n_samples, 1)
        load_force = load_force.reshape(-1, 1)

        # Debugging: Print shape after selection
        print(f"Load force shape after selection: {load_force.shape}")

        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            erds_features.T, load_force, test_size=0.2, random_state=42
        )

        # Debugging: Print shapes of split data
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

        # Apply neural network for regression
        y_pred, mse, r2 = apply_neural_network(X_train, y_train, X_test, y_test)
        
        # Plot the predictions vs actual
        plot_predictions(y_test, y_pred)

        # Record results for each file
        file_results = {
            'File': mat_file,
            'MSE': mse,
            'R2': r2
        }
        all_results.append(file_results)

        # Construct the per-run output file path using pathlib
        mat_path = Path(mat_file)
        mat_name = mat_path.stem  # 'HS_P1_S1' from 'HS_P1_S1.mat'
        output_file_per_run = output_path.parent / f'nn_summary_results_{mat_name}_V9.xlsx'

        # Save individual results to the Excel output file
        save_results_to_excel(y_test, y_pred, mse, r2, str(output_file_per_run))

    # Create a summary DataFrame and save it as the main output file
    summary_df = pd.DataFrame(all_results)
    summary_df.to_excel(output_file, index=False)

    print(f"All files processed. Summary saved to {output_file}.")

run_pipeline(file_list, selected_channels, output_file)
