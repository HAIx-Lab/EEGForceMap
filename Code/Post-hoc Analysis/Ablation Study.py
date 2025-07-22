import scipy.io
from pathlib import Path
import numpy as np
import mne
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import re
from mne_icalabel import label_components
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import itertools

# Utility functions for Loading the File
def extract_participant_id(filename):
    match = re.search(r'P(\d+)', filename)
    return int(match.group(1)) if match else None
def clean_name(name):
    """Standardize channel names to match 10-20 system conventions"""
    return (
        name.strip()
        .replace('\u200b', '')
        .replace('\u00a0', '')
        .replace('FP', 'Fp')  # Fix FP1/Fp1 case mismatch
        .replace('OZ', 'Oz')  # Fix OZ/Oz case mismatch
        .replace('CZ', 'Cz')  # Fix CZ/Cz case mismatch
        .replace('PZ', 'Pz')  # Fix PZ/Pz mismatch
        .replace('FZ', 'Fz')  # Fix FZ/Fz mismatch
        .upper()  # Standardize to uppercase
    )
def load_kin_eeg_data(mat_file, selected_channels=None, use_all_channels=False):
    mat_contents = scipy.io.loadmat(mat_file)
    if 'hs' not in mat_contents:
        return None, None, None, None, None
    
    hs_data = mat_contents['hs'][0, 0]
    hs_fields = hs_data.dtype.names
    
    # Kinematic data loading
    if 'kin' not in hs_fields:
        return None, None, None, None, None
    kin_struct = hs_data['kin'][0, 0]
    kin_data = kin_struct['sig']
    kin_names = [str(name).strip() for name in kin_struct['names'].flatten()]

    # EEG data loading
    if 'eeg' not in hs_fields:
        return None, None, None, None, None
    eeg_struct = hs_data['eeg'][0, 0]
    eeg_data = eeg_struct['sig']
    eeg_names = [str(name).strip() for name in eeg_struct['names'].flatten()]
    sfreq = 500

    # Channel selection logic
    if not use_all_channels and selected_channels:
        selected_indices = [i for i, name in enumerate(eeg_names) 
                          if clean_name(name) in [clean_name(ch) for ch in selected_channels]]
        if not selected_indices:
            print(f"Warning: No selected channels found in {mat_file.name}. Using all channels.")
            selected_indices = list(range(len(eeg_names)))
    else:
        selected_indices = list(range(len(eeg_names)))

    eeg_data = eeg_data[:, selected_indices]
    eeg_names = [eeg_names[i] for i in selected_indices]

    return kin_data, eeg_data, sfreq, kin_names, eeg_names

# Modified preprocessing pipeline
def load_and_preprocess_data(eeg_data, eeg_names, sfreq, apply_line_filter=True, remove_components=True):
    # Keep original channel name handling from working code
    eeg_names = [str(name).strip() for name in eeg_names]
    
    info = mne.create_info(ch_names=eeg_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data.T, info)
    
    # Use original montage approach that worked
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # Match channels using original logic from working code
    raw.set_montage(montage, match_case=False, on_missing='warn')
    
    # Keep only selected channels that exist in montage
    montage_ch_names = [ch.upper() for ch in montage.ch_names]
    valid_channels = [ch for ch in raw.ch_names if clean_name(ch) in montage_ch_names]
    
    if not valid_channels:
        print("No valid channels remaining")
        return None
        
    raw.pick_channels(valid_channels, ordered=True)
    
    # Rest of processing remains the same
    raw.set_eeg_reference('average')
    raw.filter(1, 100)
    
    if apply_line_filter:
        raw.notch_filter(50)
        
    if remove_components:
        ica = mne.preprocessing.ICA(
            n_components=9,
            method='infomax',
            fit_params=dict(extended=True),
            random_state=97
        )
        ica.fit(raw)
        ic_labels = label_components(raw, ica, 'iclabel')
        artifact_indices = [i for i, label in enumerate(ic_labels["labels"]) 
                          if label in ["eye", "muscle", "other"]]
        ica.exclude = artifact_indices
        raw = ica.apply(raw)

    return raw

# Extract Kinetic Data
def extract_load_force(kin_data, force_indices, sfreq, window_size=0.1, step_size=0.05):
    win_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)
    num_windows = (kin_data.shape[0] - win_samples) // step_samples + 1
    load_force = []
    for i in range(num_windows):
        start = i * step_samples
        end = start + win_samples
        window_data = kin_data[start:end, force_indices]
        total_force = np.sum(window_data, axis=1)
        load_force.append(np.mean(total_force))
    return np.array(load_force).reshape(-1, 1)

# Extract ERDS features with 100 ms baseline
def calculate_baseline_psd(raw, kin_data, force_indices, sfreq, window_size=1.0, step_size=0.005, force_threshold=0.2):
    """Calculate baseline PSD using low-force windows"""
    data, _ = raw[:, :]
    win_samples = int(window_size * sfreq)
    
    # Extract load force for all windows
    load_force = extract_load_force(kin_data, force_indices, sfreq, window_size, step_size)
    
    # Find baseline windows with low force
    baseline_indices = np.where(load_force < force_threshold)[0]
    if len(baseline_indices) == 0:
        baseline_indices = [0]

    # Calculate baseline PSDs with window-specific parameters
    psds_mu, psds_theta = [], []
    for i in baseline_indices:
        start = i * int(step_size * sfreq)
        end = start + win_samples
        psd_mu, _ = mne.time_frequency.psd_array_welch(
            data[:, start:end], 
            sfreq=sfreq, 
            fmin=9, 
            fmax=11,
            n_fft=win_samples,    # Match FFT length to window size
            n_per_seg=win_samples, # No zero-padding
            window='hamming'      # Explicit window type
        )
        psd_theta, _ = mne.time_frequency.psd_array_welch(
            data[:, start:end], 
            sfreq=sfreq, 
            fmin=4, 
            fmax=7,
            n_fft=win_samples,
            n_per_seg=win_samples,
            window='hamming'
        )
        psds_mu.append(psd_mu)
        psds_theta.append(psd_theta)

    return np.mean(psds_mu, axis=0), np.mean(psds_theta, axis=0), load_force
def extract_erds_features(raw, baseline_mu, baseline_theta, sfreq, window_size=1.0, step_size=0.005):
    """Extract ERDS features using pre-calculated baseline"""
    data, _ = raw[:, :]
    win_samples = int(window_size * sfreq)
    step_samples = int(step_size * sfreq)
    num_windows = (data.shape[1] - win_samples) // step_samples + 1

    erds_features = []
    
    for i in range(num_windows):
        start = i * step_samples
        end = start + win_samples
        psds_mu, _ = mne.time_frequency.psd_array_welch(
            data[:, start:end],
            sfreq=sfreq,
            fmin=9,
            fmax=11,
            n_fft=win_samples,    # Match FFT length to window size
            n_per_seg=win_samples, # No zero-padding
            window='hamming'
        )
        psds_theta, _ = mne.time_frequency.psd_array_welch(
            data[:, start:end],
            sfreq=sfreq,
            fmin=4,
            fmax=7,
            n_fft=win_samples,
            n_per_seg=win_samples,
            window='hamming'
        )
        
        erds_mu = (psds_mu - baseline_mu) / baseline_mu
        erds_theta = (psds_theta - baseline_theta) / baseline_theta
        
        erds_features.append(np.hstack([erds_mu.flatten(), erds_theta.flatten()]))

    return np.array(erds_features)

# Ablation study implementation
def run_ablation_study(folder_path, selected_channels, output_folder):
    folder_path = Path(folder_path)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ablation_conditions = {
        'channel_selection': [True, False],
        'component_analysis': [True, False],
        'line_noise_filter': [True, False],
        'feature_extraction': [True, False]
    }
    
    all_results = []
    time_series_data = []
    
    # Generate all condition combinations
    for combo in itertools.product(*ablation_conditions.values()):
        condition = {k:v for k,v in zip(ablation_conditions.keys(), combo)}
        
        participant_data = defaultdict(lambda: {'X': [], 'y': []})
        for mat_file in folder_path.glob('*.mat'):
            participant_id = extract_participant_id(mat_file.stem)
            kin_data, eeg_data, sfreq, kin_names, eeg_names = load_kin_eeg_data(
                mat_file, selected_channels, use_all_channels=not condition['channel_selection']
            )
            
            if kin_data is None:
                continue

            raw = load_and_preprocess_data(
                eeg_data, eeg_names, sfreq,
                apply_line_filter=condition['line_noise_filter'],
                remove_components=condition['component_analysis']
            )
            
            # Calculate baseline if using ERDS features
            baseline_mu, baseline_theta = None, None
            if condition['feature_extraction']:
                force_indices = [i for i, name in enumerate(kin_names) if 'force' in name.lower()]
                baseline_mu, baseline_theta, _ = calculate_baseline_psd(
                    raw, kin_data, force_indices, sfreq
                )
            
            features = extract_erds_features(
                raw, baseline_mu, baseline_theta,
                use_erds=condition['feature_extraction']
            )
            
            # Create target variable
            load_force = extract_load_force(kin_data, force_indices, sfreq)
            participant_data[participant_id]['X'].append(features)
            participant_data[participant_id]['y'].append(load_force)
        
        # Train and evaluate for each participant
        for participant_id, data in participant_data.items():
            X = np.vstack(data['X'])
            y = np.vstack(data['y'])
            
            if X.size == 0 or y.size == 0:
                continue
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Model configuration
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='tanh', input_shape=(X.shape[1],)),
                tf.keras.layers.Dense(16, activation='tanh'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=50, verbose=0)
            
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            all_results.append({
                **condition,
                'participant_id': participant_id,
                'mse': mse,
                'r2': r2
            })
            
            # Store time series data for sample visualization
            if participant_id == 1 and len(time_series_data) < 5:
                time_series_data.append({
                    **condition,
                    'true': y_test[:100].flatten(),
                    'pred': y_pred[:100].flatten()
                })

    # Save results and perform analysis
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path / "ablation_results.csv", index=False)
    
    plot_ablation_results(results_df, time_series_data, output_path)
    perform_anova(results_df, output_path)
    
    return results_df

# Plotting functions
def plot_ablation_results(results_df, time_series_data, output_path):
    plt.figure(figsize=(12, 6))
    for metric in ['mse', 'r2']:
        plt.clf()
        sns.barplot(x='variable', y='value', hue='condition', 
                   data=pd.melt(results_df, id_vars=[c for c in results_df if c not in ['mse', 'r2']],
                                value_vars=[metric]))
        plt.title(f"{metric.upper()} Across Ablation Conditions")
        plt.savefig(output_path / f"ablation_{metric}.png", dpi=300)
    
    plt.figure(figsize=(15, 5))
    for i, ts in enumerate(time_series_data[:5]):
        plt.subplot(1, 5, i+1)
        plt.plot(ts['true'], label='True')
        plt.plot(ts['pred'], label='Predicted')
        plt.title(f"Condition: {ts['channel_selection']}-{ts['component_analysis']}-{ts['line_noise_filter']}-{ts['feature_extraction']}")
    plt.savefig(output_path / "time_series_comparison.png", dpi=300)

# Statistical analysis
def perform_anova(data_df, output_path):
    model = ols('mse ~ C(channel_selection) + C(component_analysis) + C(line_noise_filter) + C(feature_extraction) + C(participant_id)',
               data=data_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table.to_csv(output_path / "anova_results.csv")

# Main execution
if __name__ == "__main__":
    selected_channels = ['C3', 'CP1', 'CP2', 'Cz', 'FC2', 'FC6', 'Fp1', 'P3', 'C4']
    run_ablation_study(
        # [Sample for Decoder Model File], 
        selected_channels,
        # Output File Path = Excel File Path to Store Output
    )

