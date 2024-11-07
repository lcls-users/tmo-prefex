#!/usr/bin/env python3

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import pickle  # Ensure pickle is imported

# Add the path to your custom model modules if needed
sys.path.append('/sdf/home/a/ajshack/TOF_ML/src')
from models.tof_to_energy_model import TofToEnergyModel, InteractionLayer, ScalingLayer, LogTransformLayer


def load_scalers(scalers_path):
    """Load scalers from the specified path."""
    if os.path.exists(scalers_path):
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
            min_values = scalers['min_values']
            max_values = scalers['max_values']
            print(f"Scalers loaded from {scalers_path}")
            return min_values, max_values
    else:
        raise FileNotFoundError(f"Scalers file not found at {scalers_path}")


def parse_params_line(params_line):
    """Parses a params line into a dictionary."""
    params_dict = {}
    if params_line:
        params_list = params_line.strip().split()
        for param in params_list:
            if param.startswith('--'):
                key_value = param[2:].split('=')
                if len(key_value) == 2:
                    key, value = key_value
                    params_dict[key] = value
    return params_dict


def convert_tof_to_energy(tof_array, retardation, batch_size=1024):
    """
    Converts a TOF array to an energy spectrum.

    Parameters:
        tof_array (np.ndarray): Input time-of-flight array.
        retardation (float): Retardation value.
        batch_size (int): Batch size for processing data.

    Returns:
        energy_spectrum (np.ndarray): Output energy spectrum.
    """
    # Hard-coded paths and parameters
    scalers_path = '/sdf/scratch/users/a/ajshack/scalers.pkl'
    model_path = '/sdf/scratch/users/a/ajshack/test2/12_tofs_complex_swish/main_model'
    params_line = "--batch_size=2048 --dropout=0.2 --layer_size=64 --learning_rate=0.01 --optimizer=Adam --job_name tofs_complex_swish"

    # Load the model
    main_model = tf.keras.models.load_model(model_path, custom_objects={
        'LogTransformLayer': LogTransformLayer,
        'InteractionLayer': InteractionLayer,
        'ScalingLayer': ScalingLayer,
        'TofToEnergyModel': TofToEnergyModel
    })
    main_model.min_values, main_model.max_values = load_scalers(scalers_path)
    main_model.params = parse_params_line(params_line)

    # Initialize list to store predictions
    all_predictions = []

    # Total number of samples
    total_length = len(tof_array)
    print(f"Processing TOF array with {total_length} samples.")

    # Process data in batches
    for i in range(0, total_length, batch_size):
        # Extract batch
        tof_batch = tof_array[i:i+batch_size]

        # Prepare input array
        retardation_col = np.full_like(tof_batch, retardation)
        mid1_ratio_col = np.full_like(tof_batch, 0.11248)  # Placeholder values
        mid2_ratio_col = np.full_like(tof_batch, 0.1354)   # Placeholder values
        input_array = np.column_stack([retardation_col, mid1_ratio_col, mid2_ratio_col, tof_batch]).astype(np.float32)

        # Make predictions
        y_pred_batch = main_model.predict(input_array, batch_size=batch_size).flatten()
        y_pred_conv_batch = 2 ** y_pred_batch  # Apply inverse log transformation

        # Store predictions
        all_predictions.append(y_pred_conv_batch)

    # Concatenate all predictions
    if all_predictions:
        energy_spectrum = np.concatenate(all_predictions)
    else:
        energy_spectrum = np.array([])  # Return empty array if no predictions

    return energy_spectrum

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process TOF data from an HDF5 file and plot the energy spectrum.")
    parser.add_argument('--h5_file', type=str, required=True, help='Path to the HDF5 file containing TOF data.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name within the HDF5 file (e.g., pks_0).')
    parser.add_argument('--t0', type=float, required=True, help='t0 value to subtract from TOF data.')
    parser.add_argument('--retardation', type=float, required=True, help='Retardation value.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for processing data.')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the plot (optional).')
    args = parser.parse_args()

    # Check if HDF5 file exists
    if not os.path.exists(args.h5_file):
        print(f"Error: HDF5 file '{args.h5_file}' not found.")
        return

    # Read TOF data from HDF5 file
    with h5py.File(args.h5_file, 'r') as hf:
        if args.dataset not in hf.keys():
            print(f"Error: Dataset '{args.dataset}' not found in HDF5 file.")
            return
        tof_data = hf[args.dataset][()]
        print(f"Loaded TOF data from dataset '{args.dataset}' in '{args.h5_file}'.")

    # Convert TOF to energy spectrum
    print("Starting TOF to energy conversion...")
    energy_spectrum = convert_tof_to_energy(tof_data, args.t0, args.retardation, args.batch_size)
    print(f"Energy spectrum calculated with {len(energy_spectrum)} entries.")

    if energy_spectrum.size == 0:
        print("No valid energy data obtained after processing.")
        return

    # Remove NaN and Inf values
    energy_spectrum = energy_spectrum[np.isfinite(energy_spectrum)]

    if energy_spectrum.size == 0:
        print("No valid energy data after removing NaN and Inf values.")
        return

    # Filter energy spectrum to the desired range
    energy_min = 0.0  # eV
    energy_max = 300.0  # eV
    energy_spectrum = energy_spectrum[(energy_spectrum >= energy_min) & (energy_spectrum <= energy_max)]

    if energy_spectrum.size == 0:
        print("No energy data in the specified range (0-300 eV).")
        return

    # Plot the energy spectrum
    print("Starting to plot energy spectrum...")
    plt.figure(figsize=(10, 6))

    # Set bin width and bins
    bin_width = 1.0  # eV
    bins = np.arange(energy_min, energy_max + bin_width, bin_width)

    counts, bin_edges, _ = plt.hist(energy_spectrum, bins=bins, alpha=0.7, edgecolor='black')

    plt.title('Energy Spectrum (0 - 300 eV)', fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    print("Plotting completed.")

    # Save or display the plot
    if args.save_path:
        plt.savefig(args.save_path)
        print(f"Plot saved to '{args.save_path}'.")
    else:
        plt.show()

if __name__ == "__main__":
    main()
