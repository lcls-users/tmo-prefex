#!/usr/bin/env python3

import argparse
import os
from peak_offset import (load_and_preprocess_data, convert_tof_to_energy, find_t0,
                         plot_spectra, convert_data_to_energy, load_converted_h5, determine_subplot_grid)
import h5py
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def plot_hv_pcolormesh(data_dict, run, ports, bins, window_range, energy_flag, retardation, save_path):
    n_ports = len(ports)
    n_rows, n_cols = determine_subplot_grid(n_ports)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # If there's only one subplot, make axes iterable
    if n_ports == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, port in enumerate(ports):
        ax = axes[idx]
        port_data = data_dict.get(port)
        if port_data is None:
            continue

        scan_var_keys = sorted(port_data.keys())
        scan_values = np.array(scan_var_keys)

        hist_matrix = []
        for scan_value in scan_var_keys:
            data = port_data[scan_value]

            if energy_flag:
                data = data - retardation

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
                print(f"No data in the specified window range for port {port}, scan {scan_value}.")
                hist_counts = np.zeros(len(bins_window) - 1)
            else:
                hist_counts, _ = np.histogram(data_in_range, bins=bins_window)

            hist_matrix.append(hist_counts)

        hist_matrix = np.array(hist_matrix)
        if hist_matrix.size == 0:
            print(f"No histogram data for port {port}.")
            continue

        # Create X and Y axes for pcolormesh
        xlabel = 'Energy (eV)' if energy_flag else 'Time of Flight (µs)'
        ylabel = 'Photon Energy (arb. units)'
        x_bins = (bins_window[:-1] + bins_window[1:]) / 2  # Bin centers
        X, Y = np.meshgrid(x_bins, scan_values)

        # Plot pcolormesh
        pcm = ax.pcolormesh(X, Y, hist_matrix, shading='auto')
        fig.colorbar(pcm, ax=ax, label='Counts')

        ax.set_title(f'Run {run}, Port {port}, Ret {retardation}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Hide any unused subplots
    total_subplots = len(axes)
    for idx in range(len(ports), total_subplots):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Spectra plot saved to '{save_path}'.")
    else:
        plt.show()
    plt.close(fig)


def find_t0_waterfall(data_dict, run, retardation, ports, height_t0, distance_t0, prominence_t0,
                      bins, save_path=None):
    fig, axes = plt.subplots(4, 4, figsize=(15, 15), constrained_layout=True)
    axes = axes.flatten()
    t0s = []

    for idx, port in enumerate(ports):
        ax = axes[idx]
        port_data = data_dict.get(port)
        if port_data is None:
            print(f"No data found for port {port}. Skipping.")
            t0s.append(None)
            continue

        scan_var_keys = sorted(port_data.keys())
        if not scan_var_keys:
            print(f"No scan variables found for port {port}. Skipping.")
            t0s.append(None)
            continue

        # Aggregate all data for peak finding
        all_data = np.concatenate([port_data[scan] for scan in scan_var_keys if len(port_data[scan]) > 0])

        # Create histogram using provided bins
        hist, bin_edges = np.histogram(all_data, bins=bins)

        # Adaptive peak finding
        current_height = height_t0
        t0_found = False
        while not t0_found and current_height >= 5:
            peaks, properties = find_peaks(
                hist,
                height=current_height,
                distance=distance_t0,
                prominence=prominence_t0
            )
            if len(peaks) == 0:
                print(f"No peaks found for port {port} with height threshold {current_height}.")
                if current_height > 5:
                    current_height -= 2
                    print(f"Reducing height threshold to {current_height} and retrying.")
                else:
                    break
            else:
                t0_found = True
                # Select the first peak as t0
                t0_bin = peaks[0]
                t0 = (bin_edges[t0_bin] + bin_edges[t0_bin + 1]) / 2
                t0s.append(t0)
                print(f"t0 for port {port} found at {t0:.4f} µs with peak height {properties['peak_heights'][0]}.")

        if not t0_found:
            print(f"Failed to find t0 for port {port} even after reducing height to {current_height}.")
            t0s.append(None)

        # Define window around t0 if found
        if t0_found:
            window_start = t0 - 0.015  # 0.025 µs before t0
            window_end = t0 + 0.15  # 0.2 µs after t0
            idx_start = np.searchsorted(bin_edges, window_start)
            idx_end = np.searchsorted(bin_edges, window_end)
            bins_window = bin_edges[idx_start: idx_end + 1]
        else:
            # If t0 not found, use the original window
            bins_window = bins
        x_limits = (bins_window[0], bins_window[-1])
        offset = 0
        max_height = 0

        # Plot histograms for each scan variable
        for scan in scan_var_keys:
            data_scan = port_data[scan]
            # Apply window around t0 if found
            hist_scan, bins_scan = np.histogram(data_scan, bins=bins_window)
            bin_centers = (bins_scan[:-1] + bins_scan[1:]) / 2
            ax.plot(bin_centers, hist_scan + offset, label=f'Scan {scan}')
            offset += hist_scan.max()  # Increment offset
            max_height = max(max_height, hist_scan.max() + offset)

        # Plot t0 line if found
        if t0_found:
            ax.axvline(t0, color='red', linestyle='--', label=f't0 = {t0:.4f} µs')

        # Set plot limits and labels
        ax.set_xlim(x_limits)
        ax.set_title(f'Run {run}, Port {port}, Ret {retardation}')
        ax.set_xlabel('Time of Flight (µs)')
        ax.set_ylabel('Counts')
        ax.legend(fontsize=8)

    # Save or display the plot
    if save_path:
        plt.savefig(save_path)
        print(f"t0 waterfall plot for run {run} saved to '{save_path}'.")
    else:
        plt.show()

    plt.close(fig)
    return t0s


def load_and_preprocess_data_photon_energy(run, ports, sample_size=None):
    data_dict = {}
    data_path = '/sdf/data/lcls/ds/tmo/tmox1016823/scratch/preproc/v2'
    h5_file_path = os.path.join(data_path, f'run{run}_v2.h5')

    # Check if HDF5 file exists
    if not os.path.exists(h5_file_path):
        print(f"Error: HDF5 file '{h5_file_path}' not found.")
        return None

    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as hf:
        timestamp = np.argsort(np.array(hf['timestamp'], dtype=int))
        gmd = np.array(hf['xgmd'])[timestamp]
        scan_var_1 = np.array(hf['scan_var_1'])[timestamp]

        # Limit the sample size if specified
        if sample_size:
            timestamp = timestamp[:sample_size]
            gmd = gmd[:sample_size]
            scan_var_1 = scan_var_1[:sample_size]

        scan_var_1_unq = np.unique(scan_var_1)
        for port in ports:
            print(f'\t Collecting data for port {port}')
            data_dict[port] = {}
            pks = np.array(hf[f'pks_{port}'])[timestamp]
            for scan_value in scan_var_1_unq:
                flag = scan_var_1 == scan_value
                pks_i = pks[flag] * 1e6  # Convert to µs
                pks_i = pks_i[pks_i > 0].flatten()
                data_dict[port][scan_value] = pks_i
        data_dict['gmd'] = gmd
    return data_dict, scan_var_1_unq


def convert_data_to_energy_photon_energy(data_dict, t0s, ports, retardation, tof_bins, batch_size=2048):
    data_dict_energy = {}
    energy_bins = None  # Initialize energy_bins

    # Convert TOF bins to energy bins with retardation=0
    energy_bins = convert_tof_to_energy(tof_bins, retardation=0)

    for idx, port in enumerate(ports):
        t0 = t0s[idx]
        if t0 is None:
            print(f"Skipping port {port} due to missing t0.")
            continue
        port_data = data_dict.get(port)
        if port_data is None:
            continue
        data_dict_energy[port] = {}
        for scan_value, data in port_data.items():
            data_tof = data - (t0 - 2e-3)
            data_tof = data_tof[data_tof > 0]  # Keep positive TOF values
            energy_data = convert_tof_to_energy(data_tof, retardation=retardation, batch_size=batch_size)
            data_dict_energy[port][scan_value] = energy_data
            data_dict[port][scan_value] = data_tof
    return data_dict_energy, energy_bins


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process multiple runs and plot TOF and energy spectra.")
    parser.add_argument('--runs', nargs='+', type=int, required=True, help='List of run numbers (e.g., 10 11 12).')
    parser.add_argument('--retardations', nargs='+', type=float, required=True, help='Retardation value(s) for the runs.')
    parser.add_argument('--ports', nargs='+', type=int, required=True, help='List of ports (e.g., 0 1 2).')
    parser.add_argument('--t0s', nargs='+', type=float, help='List of t0 values for each port.')
    parser.add_argument('--find_t0', action='store_true', help='If set, find t0 values.')
    parser.add_argument('--plotting', action='store_true', help='If set, generate and save plots.')
    parser.add_argument('--save_path', type=str, default=None, help='Base path to save the plots (optional).')
    parser.add_argument('--save_data_path', type=str, default=None, help='Path to save the converted energy data (optional).')
    parser.add_argument('--tof_window_range', nargs=2, type=float, default=None, help='Window range for TOF plotting and peak finding (e.g., 0 2).')
    parser.add_argument('--energy_window_range', nargs=2, type=float, default=None, help='Window range for energy plotting and peak finding (e.g., 0 300).')
    parser.add_argument('--tof_bin_width', type=float, default=0.0025, help='Bin width for TOF histograms.')
    parser.add_argument('--energy_bin_width', type=float, default=1.0, help='Bin width for energy histograms.')
    parser.add_argument('--height', type=float, default=0.1, help='Minimum height of peaks (after normalization, between 0 and 1).')
    parser.add_argument('--distance', type=float, default=None, help='Minimum distance between peaks in number of bins (optional).')
    parser.add_argument('--prominence', type=float, default=0.05, help='Minimum prominence of peaks (after normalization, between 0 and 1).')
    parser.add_argument('--height_t0', type=float, default=100, help='Minimum height of peaks for t0 finding.')
    parser.add_argument('--distance_t0', type=float, default=20, help='Minimum distance between peaks for t0 finding.')
    parser.add_argument('--prominence_t0', type=float, default=20, help='Minimum prominence of peaks for t0 finding.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for processing data (optional)')
    parser.add_argument('--sample_size', type=int, default=None, help='Subsample size. If not provided, get all data.')
    parser.add_argument('--overwrite', action='store_true', help='If set, overwrite existing converted data files.')
    parser.add_argument('--photon_energy', action='store_true', help='If set, process data based on photon energy scan.')
    parser.add_argument('--hv_select', type=float, default=None, help='Select a specific scan_var value to plot.')

    args = parser.parse_args()

    # Process retardation values
    if len(args.retardations) == 1:
        retardations = args.retardations * len(args.runs)
    elif len(args.retardations) == len(args.runs):
        retardations = args.retardations
    else:
        print("Error: Number of retardation values must be either 1 or match the number of runs.")
        return

    # Define TOF bins globally
    tof_bins = np.linspace(0, 2, 5000)
    tof_bins = tof_bins[1:]

    # Loop through each run
    for idx, run in enumerate(args.runs):
        print(f"\nProcessing Run {run}...")
        retardation = retardations[idx]
        if args.photon_energy:
            # Load data
            data_dict, scan_var_1_unq = load_and_preprocess_data_photon_energy(run, args.ports,
                                                                               sample_size=args.sample_size)
            if data_dict is None:
                continue
            t0_save_path = os.path.join(args.save_path, f"run{run}_t0.pdf") if args.save_path else None
            # Find t0
            if args.find_t0:
                t0s = find_t0_waterfall(
                    data_dict=data_dict,
                    run=run,
                    retardation=retardation,
                    ports=args.ports,
                    height_t0=args.height_t0,
                    distance_t0=args.distance_t0,
                    prominence_t0=args.prominence_t0,
                    bins=tof_bins,
                    save_path=t0_save_path
                )
            elif args.t0s is not None:
                if len(args.t0s) != len(args.ports):
                    print("Error: Number of t0s must match number of ports.")
                    continue
                t0s = args.t0s
            else:
                print("Error: t0s must be provided either via --t0s or by using --find_t0.")
                continue

            if args.save_data_path:
                save_file = os.path.join(args.save_data_path, f"run{run}_converted.h5")
                converted_file_exists = os.path.exists(save_file)
            else:
                converted_file_exists = False

            if converted_file_exists and not args.overwrite:
                print(f"Converted data for run {run} already exists at '{save_file}'. Loading data.")
                data_dict_energy = load_converted_h5(save_file, args.ports, scan=True)

            else:
                # Either overwrite is set, or the file does not exist
                # Convert to energy and save data
                data_dict_energy, energy_bins = convert_data_to_energy_photon_energy(
                    data_dict, t0s, args.ports, retardation, tof_bins
                )
                # Save converted energy data to HDF5 file
                if args.save_data_path:
                    with h5py.File(save_file, 'w') as hf:
                        for port in args.ports:
                            port_group = hf.create_group(f'port_{port}')
                            energy_data = data_dict_energy.get(port)
                            if energy_data is not None:
                                for scan_value, energy_array in energy_data.items():
                                    dataset_name = f'{scan_value}'
                                    port_group.create_dataset(dataset_name, data=energy_array)
                            else:
                                print(f"No energy data for port {port} to save.")
                    print(f"All energy data saved to '{save_file}'.")

            energy_bins = np.array(sorted(convert_tof_to_energy(tof_bins, retardation=retardation))) - retardation
            for i, port in enumerate(args.ports):
                port_data = data_dict[port]
                for scan_value, data in port_data.items():
                    data_tof = data - (t0s[i] - 2e-3)
                    data_dict[port][scan_value] = data_tof[data_tof > 0]
            if args.plotting:
                if args.hv_select:
                    hv_select = args.hv_select
                    data_subset = {port: data_dict[port].get(hv_select) for port in args.ports}
                    energy_data_subset = {port: data_dict_energy[port].get(hv_select) for port in
                                          args.ports}
                    tof_save_path = os.path.join(args.save_path, f"run{run}_tof_hv_{args.hv_select}.pdf") if args.save_path else None
                    plot_spectra(
                        data_dict=data_subset,
                        run=run,
                        retardations=[retardation] * len(args.ports),
                        t0s=t0s,
                        ports=args.ports,
                        bins=tof_bins,
                        window_range=args.tof_window_range,
                        height=args.height,
                        distance=args.distance,
                        prominence=args.prominence,
                        energy_flag=False,
                        save_path=tof_save_path
                    )

                    # Plot energy spectra
                    energy_save_path = os.path.join(args.save_path, f"run{run}_energy_hv_{args.hv_select}.pdf") if args.save_path else None
                    plot_spectra(
                        data_dict=energy_data_subset,
                        run=run,
                        retardations=[retardation] * len(args.ports),
                        t0s=t0s,
                        ports=args.ports,
                        bins=energy_bins,
                        window_range=args.energy_window_range,
                        height=args.height,
                        distance=args.distance,
                        prominence=args.prominence,
                        energy_flag=True,
                        save_path=energy_save_path
                    )
                else:
                    # Plot TOF spectra
                    tof_spectra_save_path = os.path.join(args.save_path,
                                                         f"run{run}_hv_tof_spectra.pdf") if args.save_path else None
                    plot_hv_pcolormesh(
                        data_dict=data_dict,
                        run=run,
                        ports=args.ports,
                        bins=tof_bins,  # Use TOF bins
                        window_range=args.tof_window_range,
                        energy_flag=False,
                        retardation=retardation,
                        save_path=tof_spectra_save_path
                    )
                    # Plot Energy spectra
                    energy_spectra_save_path = os.path.join(args.save_path,
                                                            f"run{run}_hv_energy_spectra.pdf") if args.save_path else None
                    plot_hv_pcolormesh(
                        data_dict=data_dict_energy,
                        run=run,
                        ports=args.ports,
                        bins=energy_bins,  # Use energy bins
                        window_range=args.energy_window_range,
                        energy_flag=True,
                        retardation=retardation,
                        save_path=energy_spectra_save_path
                    )

        else:
            data_dict = load_and_preprocess_data(run, args.ports, sample_size=args.sample_size)
            if data_dict is None:
                continue

            # Find t0s if needed
            t0s = None
            if args.find_t0:
                # Generate t0 plot and save to PDF
                t0_save_path = os.path.join(args.save_path, f"run{run}_t0.pdf") if args.save_path else None
                t0s = find_t0(
                    data_dict=data_dict,
                    run=run,
                    retardation=retardation,
                    ports=args.ports,
                    height_t0=args.height_t0,
                    distance_t0=args.distance_t0,
                    prominence_t0=args.prominence_t0,
                    bins=tof_bins,
                    save_path=t0_save_path
                )
            elif args.t0s is not None:
                if len(args.t0s) != len(args.ports):
                    print("Error: Number of t0s must match number of ports.")
                    continue
                t0s = args.t0s
            else:
                print("Error: t0s must be provided either via --t0s or by using --find_t0.")
                continue

            if args.save_data_path:
                save_file = os.path.join(args.save_data_path, f"run{run}_converted.h5")
                converted_file_exists = os.path.exists(save_file)
            else:
                converted_file_exists = False

            if converted_file_exists and not args.overwrite:
                print(f"Converted data for run {run} already exists at '{save_file}'. Loading data.")
                data_dict_energy = load_converted_h5(save_file, args.ports, scan=False)

            else:
                # Either overwrite is set, or the file does not exist
                # Convert to energy and save data
                data_dict_energy = convert_data_to_energy(
                    data_dict, [retardation] * len(args.ports), args.ports, t0s, tof_bins
                )

                # Save converted energy data to HDF5 file
                if args.save_data_path:
                    with h5py.File(save_file, 'w') as hf:
                        for port in args.ports:
                            energy_data = data_dict_energy.get(port)
                            if energy_data is not None:
                                hf.create_dataset(f'pks_{port}', data=energy_data)
                                print(f"Saved energy data for port {port} to '{save_file}'.")
                    print(f"All energy data saved to '{save_file}'.")
            energy_bins = np.array(sorted(convert_tof_to_energy(tof_bins, retardation=0))) - retardation
            print(energy_bins)
            for i, port in enumerate(args.ports):
                port_data = data_dict[port]
                data_tof = port_data - (t0s[i] - 2e-3)
                data_dict[port] = data_tof[data_tof > 0]
            # Generate plots if plotting flag is set
            if args.plotting:
                # Plot TOF spectra
                tof_save_path = os.path.join(args.save_path, f"run{run}_tof.pdf") if args.save_path else None
                plot_spectra(
                    data_dict=data_dict,
                    run=run,
                    retardations=[retardation]*len(args.ports),
                    t0s=t0s,
                    ports=args.ports,
                    bins=tof_bins,
                    window_range=args.tof_window_range,
                    height=args.height,
                    distance=args.distance,
                    prominence=args.prominence,
                    energy_flag=False,
                    save_path=tof_save_path
                )

                # Plot energy spectra
                energy_save_path = os.path.join(args.save_path, f"run{run}_energy.pdf") if args.save_path else None
                plot_spectra(
                    data_dict=data_dict_energy,
                    run=run,
                    retardations=[retardation]*len(args.ports),
                    t0s=t0s,
                    ports=args.ports,
                    bins=energy_bins,
                    window_range=args.energy_window_range,
                    height=args.height,
                    distance=args.distance,
                    prominence=args.prominence,
                    energy_flag=True,
                    save_path=energy_save_path
                )

if __name__ == '__main__':
    main()
