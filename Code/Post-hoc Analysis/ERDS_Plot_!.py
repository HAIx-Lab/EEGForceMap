import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
from mne.channels import make_standard_montage
import mne
from mne.preprocessing import ICA
import scipy.io
import pandas as pd
from mne_icalabel import label_components
import logging
from scipy.signal import hilbert as scipy_hilbert
from scipy.stats import skew, kurtosis, entropy
import os

# Load .mat file and extract EEG and kinesthetic data
def load_mat_file(mat_file_path):
    mat_file = scipy.io.loadmat(mat_file_path)
    hs_data = mat_file['hs']

    # EEG extraction
    eeg_info = hs_data['eeg'][0][0]
    ch_names = [name[0] for name in eeg_info['names'][0][0][0]]
    eeg_data = eeg_info['sig'][0][0].T
    sfreq = eeg_info['samplingrate'][0][0][0]

    # Kinesthetic data extraction
    kin_data = hs_data['kin'][0][0]['sig'][0][0]
    kin_ch_names = [name[0] for name in hs_data['kin'][0][0]['names'][0][0][0]]

    return eeg_data, ch_names, sfreq, kin_data, kin_ch_names

# Calculate load force by summing all 'force' channels
def calculate_load_force(kin_data, kin_ch_names):
    force_indices = [i for i, name in enumerate(kin_ch_names) if 'force' in name.lower()]
    
    if not force_indices:
        raise ValueError("No 'force' channels found in the kinesthetic data.")
    
    load_force = np.sum(kin_data[:, force_indices], axis=1)
    return load_force

# Remove artifacts (ICA + ICLabel)
def remove_artifact(raw):
    try:
        ica = ICA(n_components=9, random_state=97, max_iter=800)
        ica.fit(raw)
        labels = label_components(raw, ica, method='iclabel')
        bad_components = [idx for idx, label in enumerate(labels['labels']) if label in ['muscle', 'heart', 'other']]

        if bad_components:
            logging.info(f"Removing components: {bad_components}")
            raw_clean = ica.apply(raw, exclude=bad_components)
        else:
            logging.info("No bad components identified. No changes made.")
            raw_clean = raw

        return raw_clean
    except Exception as e:
        logging.error(f"Error during artifact removal: {e}")
        return raw

# Hilbert Transform and Power calculation
def apply_hilbert_transform(eeg_data, ch_names, sfreq, bands):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    hilbert_data = {}
    power_data = {}

    for band_name, (low_freq, high_freq) in bands.items():
        filtered_data = raw.copy().filter(l_freq=low_freq, h_freq=high_freq, fir_design='firwin')
        data_array = filtered_data.get_data()
        analytic_signal = np.abs(scipy_hilbert(data_array, axis=1))
        power = np.square(analytic_signal)
        hilbert_data[band_name] = analytic_signal
        power_data[band_name] = power

    return hilbert_data, power_data

# Baseline power calculation
def compute_baseline_power(power_data, num_baseline_windows):
    baseline_power = {}
    for band_name, power in power_data.items():
        baseline_power[band_name] = np.mean(power[:, :num_baseline_windows], axis=1)
    return baseline_power

# ERDS calculation
def compute_erds(event_power, baseline_power):
    erds_values = {}
    for band_name, event_pow in event_power.items():
        erds = (event_pow - baseline_power[band_name][:, None]) / baseline_power[band_name][:, None] * 100
        erds_values[band_name] = erds
    return erds_values

# Dynamic force level calculation
def determine_force_levels(load_force):
    sorted_force = np.sort(load_force)
    percentiles = [25, 33, 50, 67, 75]
    levels = [(sorted_force[0], sorted_force[int(len(sorted_force) * p / 100)]) for p in percentiles]
    levels.append((levels[-1][1], sorted_force[-1]))
    return levels

# Function to save the montage of selected channels
def save_montage(eeg_info, output_folder, selected_channels):
    montage = make_standard_montage('standard_1020')
    eeg_info.set_montage(montage)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fig, ax = plt.subplots(figsize=(5, 5))
    pos = montage.get_positions()['ch_pos']
    
    for ch_name, ch_pos in pos.items():
        if ch_name in selected_channels:
            ax.plot(ch_pos[0], ch_pos[1], 'o', color='red', markersize=8)
            ax.text(ch_pos[0], ch_pos[1], ch_name, color='red', ha='center', va='bottom', fontsize=10)
        else:
            ax.plot(ch_pos[0], ch_pos[1], 'o', color='gray', markersize=5)

    ax.set_aspect('equal')
    ax.axis('off')
    fig.savefig(f"{output_folder}/selected_channels_montage.png")
    plt.close(fig)

def plot_erds_topomaps(erds_data, eeg_info, load_force, force_levels, output_folder, selected_channels):
    montage = make_standard_montage('standard_1020')
    eeg_info.set_montage(montage)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_force_levels = len(force_levels)
    fig, axes = plt.subplots(len(erds_data), num_force_levels, figsize=(20, 5 * len(erds_data)),
                             gridspec_kw={'height_ratios': [1.5] * len(erds_data), 'hspace': 0.2, 'wspace': 0.2})

    # Adjust layout for force axis to be closer to the plots
    fig.subplots_adjust(bottom=0.1, top=0.59)
    force_axis = fig.add_axes([0.1, 0.08, 0.8, 0.05], frameon=False)
    force_axis.set_xlim(0, num_force_levels + 1)
    force_axis.set_xticks(range(1, num_force_levels + 1))
    force_axis.set_xticklabels([f'{level[1]:.2f}' for level in sorted(force_levels)],
                               fontsize=30, ha='center')
    force_axis.set_xlabel('Total Grasp Force (Newtons)', fontsize=30, labelpad=5)
    force_axis.tick_params(left=False, labelleft=False, length=5)
    force_axis.axhline(y=0, color='black', linewidth=2.5)
    for spine in force_axis.spines.values():
        spine.set_visible(False)

    # Plot each ERDS topomap with Greek letter for band names aligned properly
    for row_idx, (band_name, band_erds_data) in enumerate(erds_data.items()):
        # Align Greek letter exactly with the row of topoplots
        fig.text(0.02, 0.8 - row_idx * (0.5 / len(erds_data)), band_name,
                 va='center', ha='center', fontsize=30, fontweight='bold')

        for col_idx, level in enumerate(sorted(force_levels), start=1):
            level_indices = np.where((load_force > level[0]) & (load_force <= level[1]))[0]
            if level_indices.size == 0:
                axes[row_idx, col_idx - 1].axis('off')
                continue

            erds_level = np.mean(band_erds_data[:, level_indices], axis=1)
            im, _ = plot_topomap(erds_level, eeg_info, axes=axes[row_idx, col_idx - 1], show=False)

    # Add colorbar with larger font size for color bar labels and tick marks
    cbar = fig.colorbar(im, ax=axes[:, :], orientation='vertical', fraction=0.02, pad=0.08)
    cbar.set_label('ERDS (%)', fontsize=32)
    cbar.ax.tick_params(labelsize=24)

    # Main title and save the plot
    plt.savefig(f"{output_folder}/ERDS_AllBands_AllForce_1.png", bbox_inches='tight')
    plt.close(fig)

# Main data processing function
def process_data(mat_file_path, output_folder):
    eeg_data, ch_names, sfreq, kin_data, kin_ch_names = load_mat_file(mat_file_path)
    load_force = calculate_load_force(kin_data, kin_ch_names)
    force_levels = determine_force_levels(load_force)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    raw = remove_artifact(raw)

    bands = {
        r'$\delta$': (1, 4),
        r'$\theta$': (4, 8),
        r'$\mu$': (9, 11),
        r'$\beta$': (12, 30),
        r'$\alpha$': (8, 12)
    }
    hilbert_data, power_data = apply_hilbert_transform(raw.get_data(), ch_names, sfreq, bands)
    baseline_power = compute_baseline_power(power_data, num_baseline_windows=100)
    erds_data = compute_erds(power_data, baseline_power)

    plot_erds_topomaps(erds_data, raw.info, load_force, force_levels, output_folder, selected_channels)

# Example usage
file_path = r"C:\Users\parth\OneDrive\Desktop\Thesis\Sample for Decoder Model\HS_P1_S1.mat"
output_folder = r"C:\\Users\\parth\\OneDrive\\Desktop\\Thesis"
selected_channels = ['C3', 'CP1', 'CP2', 'Cz', 'FC2', 'FC6', 'Fp1', 'P3', 'C4']
process_data(file_path, output_folder)