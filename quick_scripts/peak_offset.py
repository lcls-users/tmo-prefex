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
from convert_spectrum import parse_params_line, convert_tof_to_energy, load_scalers


def plot_spectra(run_num, ports, t0s, retardation, window_range, height, distance, prominence, bin_width, offset, energy_flag, save_path):
    """
    Processes the data for each port, finds peaks, and plots the spectra with offset.

    Parameters:
        run_num (int): Run number.
        ports (list): List of ports.
        t0s (list): List of t0 values corresponding to each port.
        retardation (float): Retardation value.
        window_range (tuple or None): Window range for plotting and peak finding.
        height (float): Minimum height of peaks (after normalization).
        distance (float): Minimum distance between peaks in number of bins.
        prominence (float): Minimum prominence of peaks (after normalization).
        bin_width (float): Bin width for histograms.
        offset (float): Offset value to separate the histograms vertically.
        energy_flag (bool): Whether to convert TOF data to energy.
        save_path (str): Path to save the plot.
        main_model: Pre-loaded ML model for energy conversion.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for plotting
    colors = plt.cm.viridis(np.linspace(0, 1, len(ports)))

    # For indexing the offset
    offset_multiplier = 0

    # Path to the TOF data files
    data_path = '/sdf/scratch/lcls/ds/tmo/tmox1016823/scratch/preproc/v2'
    h5_file_path = os.path.join(data_path, f'run{run_num}_v2.h5')

    # Check if HDF5 file exists
    if not os.path.exists(h5_file_path):
        print(f"Error: HDF5 file '{h5_file_path}' not found.")
        return

    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as hf:

        for idx, (port, t0) in enumerate(zip(ports, t0s)):
            # Load TOF data for the port
            dataset_name = f'pks_{port}'
            if dataset_name not in hf.keys():
                print(f"Dataset '{dataset_name}' not found in HDF5 file.")
                continue
            tof_data = hf[dataset_name][()]
            print(f"Loaded TOF data from dataset '{dataset_name}' in '{h5_file_path}'.")

            # Subtract t0
            tof_data = tof_data - t0

            # Keep only positive TOF values
            tof_data = tof_data[tof_data > 0]

            if tof_data.size == 0:
                print(f"No valid TOF data after subtracting t0 for port {port}.")
                continue

            # If energy_flag is True, convert TOF to energy
            if energy_flag:
                # Convert tof_data to energy
                energy_data = convert_tof_to_energy(tof_data, t0, retardation, batch_size=1024)
                if energy_data.size == 0:
                    print(f"No valid energy data obtained after conversion for port {port}.")
                    continue
                data = energy_data
                xlabel = 'Energy (eV)'
            else:
                data = tof_data * 1e6  # Convert to microseconds for plotting
                xlabel = 'Time of Flight (µs)'

            # Determine window range if not specified
            if window_range is None:
                data_min = data.min()
                data_max = data.max()
                window_range_port = (0, data_max)
            else:
                window_range_port = window_range

            # Apply window range
            data = data[(data >= window_range_port[0]) & (data <= window_range_port[1])]

            if data.size == 0:
                print(f"No data in the specified window range for port {port}.")
                continue

            # Create histogram
            bins = np.arange(window_range_port[0], window_range_port[1] + bin_width, bin_width)
            counts, bin_edges = np.histogram(data, bins=bins)

            # Normalize counts
            max_count = counts.max()
            counts_normalized = counts / max_count

            # Perform peak finding on the normalized counts BEFORE offset
            peaks, properties = find_peaks(counts_normalized, height=height, distance=distance, prominence=prominence)

            # Offset counts for plotting
            if offset is not None:
                counts_offset = counts_normalized + offset * offset_multiplier
                offset_multiplier += 1
            else:
                counts_offset = counts_normalized

            # Compute bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Plot histogram
            ax.plot(bin_centers, counts_offset, drawstyle='steps-mid', label=f'Port {port}', color=colors[idx])

            # Adjust peak counts for plotting (add offset)
            peak_counts_offset = counts_normalized[peaks]
            if offset is not None:
                peak_counts_offset += offset * (offset_multiplier - 1)  # Adjust peak counts by the same offset

            # Mark peaks
            peak_positions = bin_centers[peaks]
            ax.plot(peak_positions, peak_counts_offset, 'x', color='red')

            # Annotate peaks
            for peak_pos, peak_count in zip(peak_positions, peak_counts_offset):
                ax.annotate(f'{peak_pos:.2f}',
                            xy=(peak_pos, peak_count),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', va='bottom', fontsize=10, color='red',
                            rotation=90,
                            arrowprops=dict(arrowstyle='->', color='red'))

    # Adjust font sizes and labels
    ax.set_title(f'{"Energy" if energy_flag else "TOF"} Spectrum with Peaks\nRun {run_num}, Retardation {retardation}', fontsize=20)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel('Normalized Counts', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=12)

    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to '{save_path}'.")
    else:
        plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process TOF or energy data and plot spectra with peak finding.")
    parser.add_argument('--run_num', type=int, required=True, help='Run number.')
    parser.add_argument('--retardation', type=float, required=True, help='Retardation value for the run.')
    parser.add_argument('--ports', nargs='+', type=int, required=True, help='List of ports (e.g., 0 1 2)')
    parser.add_argument('--t0s', nargs='+', type=float, required=True, help='List of t0 values for each port.')
    parser.add_argument('--window_range', nargs=2, type=float, default=None, help='Window range for plotting and peak finding (e.g., 0 300). If --energy is passed, this is in energy units. If not specified, plots the full spectrum starting from 0.')
    parser.add_argument('--height', type=float, default=0.1, help='Minimum height of peaks (after normalization, between 0 and 1)')
    parser.add_argument('--distance', type=float, default=None, help='Minimum distance between peaks in number of bins (optional)')
    parser.add_argument('--prominence', type=float, default=0.05, help='Minimum prominence of peaks (after normalization, between 0 and 1)')
    parser.add_argument('--bin_width', type=float, default=0.0025, help='Bin width for histograms (default=0.5)')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the plot (optional)')
    parser.add_argument('--offset', type=float, default=None, help='Offset value to separate the histograms vertically (optional)')
    parser.add_argument('--energy', action='store_true', help='If set, convert TOF data to energy before plotting.')
    args = parser.parse_args()

    # Check that the number of t0s matches the number of ports
    if len(args.t0s) != len(args.ports):
        print("Error: Number of t0s must match number of ports.")
        return

    # Convert window_range to tuple if provided
    if args.window_range is not None:
        window_range = tuple(args.window_range)
    else:
        window_range = None  # Indicates that full spectrum should be plotted starting from 0

    # Call the function to process data and plot
    plot_spectra(
        run_num=args.run_num,
        ports=args.ports,
        t0s=args.t0s,
        retardation=args.retardation,
        window_range=window_range,
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
