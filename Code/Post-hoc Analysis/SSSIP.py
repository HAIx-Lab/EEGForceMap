"""
Integrated EEG Force Prediction Function
Combines Subject-Specific and Subject-Independent Modeling
"""

import numpy as np
import pandas as pd
import scipy.io
import mne
import tensorflow as tf
import re
import os
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from scipy.signal import resample
from scipy.stats import skew
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from mne_icalabel import label_components
from collections import defaultdict

class EEGForcePredictor:

    # Basic Configuration
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.files = list(self.data_path.glob('*.mat'))
        self.selected_channels = ['C3', 'CP1', 'CP2', 'Cz', 'FC2', 'FC6', 'Fp1', 'P3', 'C4']
        self.results = {
            'subject_specific': defaultdict(dict),
            'subject_independent': defaultdict(dict)
        }
    
    # Functions to Load EEG Data
    def _extract_subject_id(self, filename):
        match = re.search(r'P(\d+)', filename.name)
        return int(match.group(1)) if match else None
    def _load_data(self, file_path):
        mat_data = scipy.io.loadmat(file_path)
        hs_data = mat_data['hs'][0,0]
        
        # Load EEG data
        eeg_struct = hs_data['eeg'][0,0]
        eeg_signal = eeg_struct['sig']
        sfreq = eeg_struct['samplingrate'][0][0]
        eeg_names = [str(name[0]) for name in eeg_struct['names'][0]]
        
        # Load kinetic data
        kin_struct = hs_data['kin'][0,0]
        kin_sig = kin_struct['sig']
        kin_names = [str(name[0]) for name in kin_struct['names'][0]]
        
        return kin_sig, eeg_signal, sfreq, kin_names, eeg_names

    # Preprocess the EEG Data through the Proposed Methodology
    def _preprocess_eeg(self, eeg_signal, sfreq, eeg_names):
        info = mne.create_info(ch_names=eeg_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_signal.T, info)
        raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
        raw.filter(1, 40)
        raw.notch_filter(50)
        
        # ICA Artifact removal
        ica = mne.preprocessing.ICA(n_components=9, random_state=97)
        ica.fit(raw)
        ic_labels = label_components(raw, ica, 'iclabel')
        bad_comps = [i for i, lbl in enumerate(ic_labels['labels']) 
                    if lbl in ['eye blink', 'muscle artifact', 'heart beat']]
        ica.exclude = bad_comps
        return ica.apply(raw)
    def _extract_features(self, raw, window_size=1.0, step_size=0.1):
        sfreq = raw.info['sfreq']
        window_samples = int(window_size * sfreq)
        step_samples = int(step_size * sfreq)
        
        features = []
        n_samples = raw.n_times
        start_idx = 0
        
        while start_idx + window_samples <= n_samples:
            # Get data for current window
            data, _ = raw[:, start_idx:start_idx + window_samples]
            data = data.T  # Transpose to (samples, channels)
            
            # Time-domain features
            means = np.mean(data, axis=0)
            variances = np.var(data, axis=0)
            skews = skew(data, axis=0)
            
            # Frequency-domain features
            psd, freqs = mne.time_frequency.psd_array_welch(
                data.T, sfreq, fmin=1, fmax=40, n_fft=window_samples)
            
            # ERD/ERS features
            alpha = np.mean(psd[:, (freqs >= 8) & (freqs <= 12)], axis=1)
            beta = np.mean(psd[:, (freqs >= 13) & (freqs <= 30)], axis=1)
            
            features.append(np.concatenate([means, variances, skews, alpha, beta]))
            
            # Move to next window
            start_idx += step_samples
        
        return np.array(features)

    # Extract Kinetic Data
    def _prepare_force_data(self, kin_sig, sfreq, window_size=1.0, step_size=0.1):
        window_samples = int(window_size * sfreq)
        step_samples = int(step_size * sfreq)
        return np.array([np.mean(kin_sig[i:i+window_samples]) 
                       for i in range(0, len(kin_sig)-window_samples, step_samples)])

    # Subject Specific Models
    def _train_subject_specific(self, X, y, subject_id):
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X, y)
        lr_pred = lr.predict(X)
        lr_metrics = {
            'r2': r2_score(y, lr_pred),
            'mse': mean_squared_error(y, lr_pred)
        }
        # PLS Regression
        pls = PLSRegression(n_components=5)
        pls.fit(X, y)
        pls_pred = pls.predict(X)
        pls_metrics = {
            'r2': r2_score(y, pls_pred),
            'mse': mean_squared_error(y, pls_pred)
        }
        # Neural Network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y.reshape(-1,1))
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(X.shape[1],)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_scaled, y_scaled, epochs=50, verbose=0)
        nn_pred = model.predict(X_scaled)
        nn_metrics = {
            'r2': r2_score(y_scaled, nn_pred),
            'mse': mean_squared_error(y_scaled, nn_pred)
        }
        
        return {
            'LinearRegression': lr_metrics,
            'PLSRegression': pls_metrics,
            'NeuralNetwork': nn_metrics
        }

    # Subject Independent Models
    def _train_subject_independent(self, all_features, all_targets, subject_ids):
        unique_subjects = np.unique(subject_ids)
        results = {}
        
        for test_subject in unique_subjects:
            train_subjects = [s for s in unique_subjects if s != test_subject]
            
            train_mask = np.isin(subject_ids, train_subjects)
            test_mask = (subject_ids == test_subject)
            
            X_train, X_test = all_features[train_mask], all_features[test_mask]
            y_train, y_test = all_targets[train_mask], all_targets[test_mask]
            
            # PLS Model
            pls = PLSRegression(n_components=5)
            pls.fit(X_train, y_train)
            pls_pred = pls.predict(X_test)
            
            # Neural Network
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_scaled, y_train, epochs=50, verbose=0)
            nn_pred = model.predict(X_test_scaled)
            
            results[test_subject] = {
                'PLS': {
                    'r2': r2_score(y_test, pls_pred),
                    'mse': mean_squared_error(y_test, pls_pred)
                },
                'NeuralNetwork': {
                    'r2': r2_score(y_test, nn_pred),
                    'mse': mean_squared_error(y_test, nn_pred)
                }
            }
        return results
    
    # Main Loop
    def run_pipeline(self):
        all_features = []
        all_targets = []
        subject_ids = []
        
        # Process all files
        for file_path in self.files:
            subject_id = self._extract_subject_id(file_path)
            kin_sig, eeg_signal, sfreq, kin_names, eeg_names = self._load_data(file_path)
            
            # Preprocessing
            raw = self._preprocess_eeg(eeg_signal, sfreq, eeg_names)
            features = self._extract_features(raw)
            targets = self._prepare_force_data(kin_sig[:,0], sfreq)
            
            # Store data
            all_features.append(features)
            all_targets.append(targets)
            subject_ids.extend([subject_id] * len(targets))
            
            # Train subject-specific models
            self.results['subject_specific'][subject_id] = self._train_subject_specific(
                features, targets, subject_id)
        
        # Combine data for subject-independent training
        all_features = np.vstack(all_features)
        all_targets = np.concatenate(all_targets)
        subject_ids = np.array(subject_ids)
        
        # Train subject-independent models
        self.results['subject_independent'] = self._train_subject_independent(
            all_features, all_targets, subject_ids)
        
        self._generate_comparison_plots()
        self._save_results()
    
    # Plotting and Saving the Results
    def _generate_comparison_plots(self):
        plt.figure(figsize=(14, 8))
        
        # Plot subject-specific results
        ss_r2 = [v['NeuralNetwork']['r2'] for v in self.results['subject_specific'].values()]
        plt.scatter(range(len(ss_r2)), ss_r2, label='Subject-Specific', c='blue', s=100)
        
        # Plot subject-independent results
        si_r2 = [v['NeuralNetwork']['r2'] for v in self.results['subject_independent'].values()]
        plt.scatter(range(len(si_r2)), si_r2, label='Subject-Independent', c='red', s=100)
        
        plt.title('Model Performance Comparison', fontsize=16)
        plt.xlabel('Participant ID', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.ylim(-0.5, 1.0)
        plt.axhline(0, color='black', linestyle='--')
        plt.legend()
        plt.grid(True)
        plt.savefig('model_comparison.png')
        plt.close()
    def _save_results(self):
        # Save metrics to Excel
        with pd.ExcelWriter('model_results.xlsx') as writer:
            # Subject-specific results
            ss_data = []
            for subj, metrics in self.results['subject_specific'].items():
                for model, vals in metrics.items():
                    ss_data.append({
                        'Subject': subj,
                        'Model': model,
                        'R2': vals['r2'],
                        'MSE': vals['mse']
                    })
            pd.DataFrame(ss_data).to_excel(writer, sheet_name='Subject_Specific', index=False)
            
            # Subject-independent results
            si_data = []
            for subj, metrics in self.results['subject_independent'].items():
                for model, vals in metrics.items():
                    si_data.append({
                        'Test_Subject': subj,
                        'Model': model,
                        'R2': vals['r2'],
                        'MSE': vals['mse']
                    })
            pd.DataFrame(si_data).to_excel(writer, sheet_name='Subject_Independent', index=False)

# Usage
if __name__ == "__main__":
    predictor = EEGForcePredictor(Input File Path*)
    predictor.run_pipeline()

# * Input File Path = Sample for Decoder Model