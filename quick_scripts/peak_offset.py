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
from convert_spectrum import convert_tof_to_energy


def append_to_save_path(save_path, suffix):
    if save_path:
        base, ext = os.path.splitext(save_path)
        return f"{base}{suffix}{ext}"
    else:
        return None


def load_and_preprocess_data(run, ports, sample_size=None):
    data_dict = {}
    data_path = '/sdf/scratch/lcls/ds/tmo/tmox1016823/scratch/preproc/v2'
    h5_file_path = os.path.join(data_path, f'run{run}_v2.h5')

    # Check if HDF5 file exists
    if not os.path.exists(h5_file_path):
        print(f"Error: HDF5 file '{h5_file_path}' not found.")
        return None

    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as hf:
        for port in ports:
            dataset_name = f'pks_{port}'
            if dataset_name not in hf.keys():
                print(f"Dataset '{dataset_name}' not found in HDF5 file.")
                continue
            if sample_size:
                tof_data = hf[dataset_name][:sample_size]
            else:
                tof_data = hf[dataset_name][()]
            print(f"Loaded TOF data from dataset '{dataset_name}' in '{h5_file_path}'.")

            # Convert to microseconds
            data = tof_data * 1e6  # Convert to µs

            # Mask out negative values
            data = data[data > 0]

            data_dict[port] = data

    return data_dict


def subtract_t0(data, t0, adjustment):
    data = data - (t0 - adjustment)
    data = data[data > 0]
    return data


def load_converted_h5(save_file, ports, scan=False):
    # Load data_dict_energy from the file
    data_dict_energy = {}
    with h5py.File(save_file, 'r') as hf:
        for port in ports:
            dataset_name = f'pks_{port}'
            if dataset_name in hf:
                port_data = hf[dataset_name][()]
                if scan:
                    data_dict_energy[port] = {}
                    scan_var_keys = sorted(port_data.keys())
                    for s in scan_var_keys:
                        data_dict_energy[port][s] = port_data[s][()]
                else:
                    data_dict_energy[port] = port_data
            else:
                data_dict_energy[port] = None
    return data_dict_energy


def convert_data_to_energy(data_dict, retardations, ports, t0s, batch_size=2048, scan=False):
    energy_dict = {}
    for idx, (port, retardation) in enumerate(zip(ports, retardations)):
        port_data = data_dict.get(port)
        t0 = t0s[idx]
        if scan:
            scan_var_keys = sorted(port_data.keys())
            energy_dict[port] = {}
            for s in scan_var_keys:
                data = subtract_t0(port_data[s], t0, 2e-3)
                energy_data = convert_tof_to_energy(data, retardation=retardation, batch_size=batch_size)
                energy_dict[port][s] = energy_data
        else:
            if port_data is not None and len(port_data) > 0:
                data = subtract_t0(port_data, t0, 2e-3)
                energy_data = convert_tof_to_energy(data, retardation=retardation, batch_size=batch_size)
                energy_dict[port] = energy_data
    return energy_dict


def find_t0(data_dict, run, retardation, ports, height_t0, distance_t0, prominence_t0, bins, save_path):
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()
    t0s = []

    for idx, port in enumerate(ports):
        ax = axes[idx]
        data = data_dict.get(port)
        if data is None or len(data) == 0:
            print(f"No data for port {port}.")
            t0s.append(None)
            continue

        # Create histogram using provided bins
        hist, bin_edges = np.histogram(data, bins=bins)

        # Initialize peak height parameter
        current_height = height_t0

        t0_found = False
        while not t0_found and current_height >= 5:
            # Use current_height, distance_t0, prominence_t0 in peak finding
            pks, properties = find_peaks(hist, height=current_height, distance=distance_t0, prominence=prominence_t0)

            if len(pks) == 0:
                print(f"No peaks found for port {port} with height {current_height}.")
                if current_height > 5:
                    current_height -= 2  # Reduce the height by 2
                    if current_height < 5:
                        current_height = 5  # Do not go below 5
                else:
                    break
            else:
                t0_found = True
                # Proceed with processing
                bin_width = bin_edges[1] - bin_edges[0]
                t0_bin = pks[0]
                t0 = bin_edges[t0_bin] + bin_width / 2
                t0s.append(t0)  # t0 is in µs

                # Adjust window around t0 peak
                window_start = t0 - 0.025  # 0.025 µs before t0
                window_end = t0 + 0.2      # 0.2 µs after t0

                idx_start = np.searchsorted(bin_edges, window_start)
                idx_end = np.searchsorted(bin_edges, window_end)
                hist_window = hist[idx_start: idx_end]
                bins_window = bin_edges[idx_start: idx_end+1]

                # Adjust y-axis height to 5x the t0 peak height
                max_height = hist[t0_bin] * 5

                # Plot histogram
                ax.stairs(hist_window, bins_window, label=f'Port {port}')
                # Plot t0 as dashed red line
                t0_label = f't0 = {t0:.4f} µs'
                ax.axvline(x=t0, linestyle='--', color='red', label=t0_label)
                ax.set_ylim(0, max_height)

                # Set labels and title
                ax.set_title(f'Run {run}, Retardation {retardation}, Port {port}', fontsize=14)
                ax.set_xlabel('Time of Flight (µs)', fontsize=12)
                ax.set_ylabel('Counts', fontsize=12)
                ax.legend(fontsize=10)

        if not t0_found:
            print(f"Failed to find t0 for port {port} even after reducing height to {current_height}.")
            t0s.append(None)
            # Optionally, you can plot an empty plot or skip plotting for this port

    # Hide any unused subplots
    total_subplots = 16
    for idx in range(len(ports), total_subplots):
        axes[idx].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"t0 Plot saved to '{save_path}'.")
    else:
        plt.show()
    plt.close(fig)
    return t0s


def plot_ports(data_dict, ports, window_range, height, distance, prominence, energy_flag, save_path):
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), constrained_layout=True)
    axes = axes.flatten()

    for idx, port in enumerate(ports):
        ax = axes[idx]
        data = data_dict.get(port)
        if data is None or len(data) == 0:
            print(f"No data for port {port}.")
            continue

        # Determine window range if not specified
        if window_range is None:
            data_min = data.min()
            data_max = data.max()
            window_range_port = (0, data_max)
        else:
            window_range_port = window_range

        # Apply window range
        data_in_range = data[(data >= window_range_port[0]) & (data <= window_range_port[1])]

        if data_in_range.size == 0:
            print(f"No data in the specified window range for port {port}.")
            continue

        # Create histogram
        if energy_flag:
            xlabel = 'Energy (eV)'
            bins = np.linspace(data_in_range.min(), data_in_range.max(), 5000)
        else:
            xlabel = 'Time of Flight (µs)'
            bins = np.linspace(0, 2, 5000)

        counts, bin_edges = np.histogram(data_in_range, bins=bins)

        # Normalize counts
        max_count = counts.max()
        counts_normalized = counts / max_count

        # Perform peak finding on the normalized counts
        peaks, properties = find_peaks(counts_normalized, height=height, distance=distance, prominence=prominence)

        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot histogram
        ax.plot(bin_centers, counts_normalized, drawstyle='steps-mid', label=f'Port {port}')
        # Mark peaks
        ax.plot(bin_centers[peaks], counts_normalized[peaks], 'x', color='red')
        # Annotate peaks
        for peak_pos, peak_count in zip(bin_centers[peaks], counts_normalized[peaks]):
            ax.annotate(f'{peak_pos:.2f}',
                        xy=(peak_pos, peak_count),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, color='red',
                        rotation=90,
                        arrowprops=dict(arrowstyle='->', color='red'))

        # Set labels and title
        ax.set_title(f'Port {port}', fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Normalized Counts', fontsize=12)
        ax.legend(fontsize=10)

    # Hide any unused subplots
    total_subplots = 16
    for idx in range(len(ports), total_subplots):
        axes[idx].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to '{save_path}'.")
    else:
        plt.show()


def plot_spectra(data_dict, run, retardations, t0s, ports, bins, window_range, height, distance, prominence, energy_flag, save_path):
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()

    for idx, port in enumerate(ports):
        ax = axes[idx]
        data = data_dict.get(port)
        if energy_flag:
            data = data - retardations[idx]
        if data is None or len(data) == 0:
            print(f"No data for port {port}.")
            continue

        # Determine window range
        if window_range is not None:
            # Apply window range to bins
            bin_mask = (bins >= window_range[0]) & (bins <= window_range[1])
            bins_window = bins[bin_mask]
        else:
            bins_window = bins

        # Apply window range to data
        data_in_range = data[(data >= bins_window[0]) & (data <= bins_window[-1])]

        if data_in_range.size == 0:
            print(f"No data in the specified window range for port {port}.")
            continue

        # Create histogram using provided bins
        counts, bin_edges = np.histogram(data_in_range, bins=bins_window)

        # Normalize counts
        max_count = counts.max()
        counts_normalized = counts / max_count

        # Perform peak finding on the normalized counts
        peaks, properties = find_peaks(counts_normalized, height=height, distance=distance, prominence=prominence)

        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot histogram
        ax.plot(bin_centers, counts_normalized, drawstyle='steps-mid', label=f'Port {port}')
        # Mark peaks
        ax.plot(bin_centers[peaks], counts_normalized[peaks], 'x', color='red')
        # Annotate peaks
        for peak_pos, peak_count in zip(bin_centers[peaks], counts_normalized[peaks]):
            ax.annotate(f'{peak_pos:.2f}',
                        xy=(peak_pos, peak_count),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, color='red',
                        rotation=90,
                        arrowprops=dict(arrowstyle='->', color='red'))

        # Set labels and title
        t0_value = t0s[idx] if t0s is not None and t0s[idx] is not None else 'N/A'
        retardation_value = retardations[idx] if retardations is not None else 'N/A'
        xlabel = 'Energy (eV)' if energy_flag else 'Time of Flight (µs)'
        ax.set_title(f'Run {run}, Retardation {retardation_value}, Port {port}, t0={t0_value}', fontsize=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Normalized Counts', fontsize=12)
        ax.legend(fontsize=10)

    # Hide any unused subplots
    total_subplots = 16
    for idx in range(len(ports), total_subplots):
        axes[idx].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Spectra Plot saved to '{save_path}'.")
    else:
        plt.show()
    plt.close(fig)



def plot_spectra_waterfall(data_dict, ports, window_range, height, distance, prominence, bin_width, offset, energy_flag, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for plotting
    colors = plt.cm.viridis(np.linspace(0, 1, len(ports)))

    # For indexing the offset
    offset_multiplier = 0

    for idx, port in enumerate(ports):
        data = data_dict.get(port)
        if data is None or len(data) == 0:
            print(f"No data for port {port}.")
            continue

        # Determine window range if not specified
        if window_range is None:
            data_min = data.min()
            data_max = data.max()
            window_range_port = (0, data_max)
        else:
            window_range_port = window_range

        # Apply window range
        data_in_range = data[(data >= window_range_port[0]) & (data <= window_range_port[1])]

        if data_in_range.size == 0:
            print(f"No data in the specified window range for port {port}.")
            continue

        # Create histogram
        if energy_flag:
            xlabel = 'Energy (eV)'
        else:
            xlabel = 'Time of Flight (µs)'

        bins = np.arange(window_range_port[0], window_range_port[1] + bin_width, bin_width)
        counts, bin_edges = np.histogram(data_in_range, bins=bins)

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
    ax.set_title(f'{"Energy" if energy_flag else "TOF"} Spectrum with Peaks', fontsize=20)
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
    parser.add_argument('--retardation', nargs='+', type=float, required=True, help='Retardation value(s) for the run.')
    parser.add_argument('--ports', nargs='+', type=int, required=True, help='List of ports (e.g., 0 1 2)')
    parser.add_argument('--t0s', nargs='+', type=float, help='List of t0 values for each port.')
    parser.add_argument('--window_range', nargs=2, type=float, default=None, help='Window range for plotting and peak finding (e.g., 0 300). If --energy is passed, this is in energy units. If not specified, plots the full spectrum starting from 0.')
    parser.add_argument('--height', type=float, default=0.1, help='Minimum height of peaks (after normalization, between 0 and 1)')
    parser.add_argument('--distance', type=float, default=None, help='Minimum distance between peaks in number of bins (optional)')
    parser.add_argument('--prominence', type=float, default=0.05, help='Minimum prominence of peaks (after normalization, between 0 and 1)')
    parser.add_argument('--height_t0', type=float, default=100, help='Minimum height of peaks for t0 finding.')
    parser.add_argument('--distance_t0', type=float, default=20, help='Minimum distance between peaks for t0 finding.')
    parser.add_argument('--prominence_t0', type=float, default=20, help='Minimum prominence of peaks for t0 finding.')
    parser.add_argument('--bin_width', type=float, default=0.0025, help='Bin width for histograms (default=0.0025)')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the plot (optional)')
    parser.add_argument('--offset', type=float, default=None, help='Offset value to separate the histograms vertically (optional)')
    parser.add_argument('--energy', action='store_true', help='If set, convert TOF data to energy before plotting.')
    parser.add_argument('--plot_ports', action='store_true', help='If set, plot each port individually.')
    parser.add_argument('--find_t0', action='store_true', help='If set, find t0 values.')
    parser.add_argument('--plot_spectra', action='store_true', help='If set, plot spectra with each port on a different axis.')
    parser.add_argument('--plot_spectra_waterfall', action='store_true', help='If set, plot spectra using the waterfall method.')
    args = parser.parse_args()

    # Process retardation values
    if len(args.retardation) == 1:
        retardations = args.retardation * len(args.ports)
    elif len(args.retardation) == len(args.ports):
        retardations = args.retardation
    else:
        print("Error: Number of retardation values must be either 1 or match the number of ports.")
        return

    # Load and preprocess data
    data_dict = load_and_preprocess_data(args.run_num, args.ports, subsample=1024*4)
    if data_dict is None:
        return

    # Find t0s if needed
    t0s = None
    if args.find_t0:
        t0s = find_t0(
            data_dict=data_dict,
            run=args.run_num,
            retardation=args.retardation,
            ports=args.ports,
            height_t0=args.height_t0,
            distance_t0=args.distance_t0,
            prominence_t0=args.prominence_t0,
            save_path=append_to_save_path(args.save_path, "_t0")
        )
        # Subtract t0s and mask negative values again
        data_dict = subtract_t0(data_dict, t0s, args.ports)
    elif args.t0s is not None:
        if len(args.t0s) != len(args.ports):
            print("Error: Number of t0s must match number of ports.")
            return
        t0s = args.t0s
        # Subtract t0s and mask negative values again
        data_dict = subtract_t0(data_dict, t0s, args.ports)
    else:
        print("Error: t0s must be provided either via --t0s or by using --find_t0.")
        return

    # Convert to energy if energy flag is set
    if args.energy:
        data_dict = convert_data_to_energy(data_dict, retardations, args.ports)

    # Now call the plotting functions based on flags
    if args.plot_ports:
        plot_ports(
            data_dict=data_dict,
            ports=args.ports,
            window_range=args.window_range,
            height=args.height,
            distance=args.distance,
            prominence=args.prominence,
            energy_flag=args.energy,
            save_path=append_to_save_path(args.save_path, "_plot_ports")
        )
    if args.plot_spectra:
        plot_spectra(
            data_dict=data_dict,
            run=args.run_num,
            retardation=args.retardation,
            ports=args.ports,
            window_range=args.window_range,
            height=args.height,
            distance=args.distance,
            prominence=args.prominence,
            energy_flag=args.energy,
            save_path=append_to_save_path(args.save_path, "_plot_spectra")
        )
    if args.plot_spectra_waterfall:
        plot_spectra_waterfall(
            data_dict=data_dict,
            ports=args.ports,
            window_range=args.window_range,
            height=args.height,
            distance=args.distance,
            prominence=args.prominence,
            bin_width=args.bin_width,
            offset=args.offset,
            energy_flag=args.energy,
            save_path=append_to_save_path(args.save_path, "_waterfall")
        )

if __name__ == '__main__':
    main()


