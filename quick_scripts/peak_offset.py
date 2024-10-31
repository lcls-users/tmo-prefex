#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
import pickle
from scipy.signal import find_peaks
# Add the path to your custom model modules if needed
sys.path.append('/sdf/home/a/ajshack/TOF_ML/src')
from models.tof_to_energy_model import TofToEnergyModel, InteractionLayer, ScalingLayer, LogTransformLayer
from convert_spectrum import convert_tof_to_energy, load_scalers, parse_params_line


def load_data(run_num, port, t0, data_path):
    """Load TOF data for a specific run and port, and subtract t0."""
    h5_file_path = os.path.join(data_path, f'run{run_num}_v2.h5')
    dataset_name = f'pks_{port}'

    if not os.path.exists(h5_file_path):
        print(f"Error: HDF5 file '{h5_file_path}' not found.")
        return None

    with h5py.File(h5_file_path, 'r') as hf:
        if dataset_name not in hf:
            print(f"Dataset '{dataset_name}' not found in HDF5 file.")
            return None
        tof_data = hf[dataset_name][()]
        tof_data = tof_data - t0  # Subtract t0
        tof_data = tof_data[tof_data > 0]  # Keep only positive TOF values
        return tof_data


def plot_spectra(run_num, ports, t0s, retardation, window_range, height, distance, prominence, bin_width, offset, energy_flag, save_path):
    """Process data, find peaks, and plot spectra with offset."""
    # Data path (adjust if necessary)
    data_path = '/path/to/your/data'  # Replace with the actual path to your data

    # Create figure
    plt.figure(figsize=(10, 6))

    # For offsetting histograms
    offset_multiplier = 0

    for port, t0 in zip(ports, t0s):
        # Load data
        data = load_data(run_num, port, t0, data_path)
        if data is None or len(data) == 0:
            continue

        # Convert to energy if needed
        if energy_flag:
            data = convert_tof_to_energy(data, retardation)
            xlabel = 'Energy (eV)'
        else:
            data = data * 1e6  # Convert to microseconds for TOF
            xlabel = 'Time of Flight (µs)'

        # Apply window range
        if window_range:
            data = data[(data >= window_range[0]) & (data <= window_range[1])]

        if len(data) == 0:
            print(f"No data in the specified window range for port {port}.")
            continue

        # Create histogram
        bins = np.arange(data.min(), data.max() + bin_width, bin_width)
        counts, bin_edges = np.histogram(data, bins=bins)

        # Normalize counts
        counts_normalized = counts / counts.max()

        # Offset counts
        counts_offset = counts_normalized + offset * offset_multiplier if offset else counts_normalized
        offset_multiplier += 1

        # Find peaks
        peaks, _ = find_peaks(counts_normalized, height=height, distance=distance, prominence=prominence)
        peak_positions = bin_edges[:-1][peaks] + bin_width / 2
        peak_heights = counts_offset[peaks]

        # Plot histogram
        plt.plot(bin_edges[:-1] + bin_width / 2, counts_offset, drawstyle='steps-post', label=f'Port {port}')

        # Mark peaks
        plt.plot(peak_positions, peak_heights, 'x', color='red')

        # Annotate peaks
        for x, y in zip(peak_positions, peak_heights):
            plt.annotate(f'{x:.2f}', xy=(x, y), xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=10, color='red')

    # Finalize plot
    plt.xlabel(xlabel)
    plt.ylabel('Normalized Counts')
    plt.title(f'{"Energy" if energy_flag else "TOF"} Spectrum with Peaks\nRun {run_num}, Retardation {retardation}')
    plt.legend()
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to '{save_path}'.")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot spectra and find peaks.")
    parser.add_argument('--run_num', type=int, required=True, help='Run number.')
    parser.add_argument('--retardation', type=float, required=True, help='Retardation value.')
    parser.add_argument('--ports', nargs='+', type=int, required=True, help='List of ports.')
    parser.add_argument('--t0s', nargs='+', type=float, required=True, help='List of t0 values for each port.')
    parser.add_argument('--window_range', nargs=2, type=float, default=None, help='Window range (optional).')
    parser.add_argument('--height', type=float, default=0.1, help='Minimum peak height.')
    parser.add_argument('--distance', type=float, default=None, help='Minimum distance between peaks.')
    parser.add_argument('--prominence', type=float, default=0.05, help='Minimum prominence of peaks.')
    parser.add_argument('--bin_width', type=float, default=1.0, help='Bin width for histogram.')
    parser.add_argument('--offset', type=float, default=0.0, help='Offset for plotting histograms.')
    parser.add_argument('--energy', action='store_true', help='Convert TOF data to energy.')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the plot.')
    args = parser.parse_args()

    # Ensure t0s and ports have the same length
    if len(args.t0s) != len(args.ports):
        print("Error: The number of t0s must match the number of ports.")
        sys.exit(1)

    # Convert window_range to tuple if provided
    if args.window_range:
        win_min, win_max = args.window_range
    else:
        win_min = None
        win_max = None

    # Call the plotting function
    plot_spectra(
        run_num=args.run_num,
        ports=args.ports,
        t0s=args.t0s,
        retardation=args.retardation,
        window_range=[win_min, win_max],
        height=args.height,
        distance=args.distance,
        prominence=args.prominence,
        bin_width=args.bin_width,
        offset=args.offset,
        energy_flag=args.energy,
        save_path=args.save_path
    )

if __name__ == '__main__':
    main()