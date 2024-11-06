#!/usr/bin/env python3

import argparse
import os
from peak_offset import (
    load_and_preprocess_data,
    subtract_t0,
    convert_tof_to_energy,
    find_t0,
    plot_spectra,
)
import h5py
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def plot_hv_energy_spectra(data_dict_energy, run, ports, energy_bin_width, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()

    for idx, port in enumerate(ports):
        ax = axes[idx]
        port_data = data_dict_energy.get(port)
        if port_data is None:
            continue

        scan_var_keys = sorted(port_data.keys())
        num_scans = len(scan_var_keys)

        # Determine bin edges
        all_energy = np.concatenate([port_data[scan_value] for scan_value in scan_var_keys if len(port_data[scan_value]) > 0])
        if len(all_energy) == 0:
            print(f"No energy data for port {port}")
            continue
        energy_min = all_energy.min()
        energy_max = all_energy.max()
        bins = np.arange(energy_min, energy_max + energy_bin_width, energy_bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Create histogram matrix
        hist_matrix = np.zeros((num_scans, len(bin_centers)))
        for i, scan_value in enumerate(scan_var_keys):
            energy_data = port_data[scan_value]
            if len(energy_data) == 0:
                hist_matrix[i] = np.zeros(len(bin_centers))
                continue
            hist, _ = np.histogram(energy_data, bins=bins)
            hist_matrix[i] = hist

        # Plot the summed histogram
        summed_hist = np.sum(hist_matrix, axis=0)
        ax.plot(bin_centers, summed_hist, drawstyle='steps-mid', label=f'Port {port}')
        ax.set_title(f'Run {run}, Port {port}')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Counts')
        ax.legend()

    # Hide any unused subplots
    total_subplots = 16
    for idx in range(len(ports), total_subplots):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Energy spectra plot saved to '{save_path}'.")
    else:
        plt.show()
    plt.close(fig)



def find_t0_waterfall(data_dict, run, ports, height_t0, distance_t0, prominence_t0, save_path):
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()
    t0s = []

    for idx, port in enumerate(ports):
        ax = axes[idx]
        port_data = data_dict.get(port)
        if port_data is None:
            t0s.append(None)
            continue

        scan_var_keys = sorted(port_data.keys())
        num_scans = len(scan_var_keys)

        offset = 0
        max_height = 0
        for scan_value in scan_var_keys:
            data = port_data[scan_value]
            hist, bins = np.histogram(data, bins=5000, range=(0, 2))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.plot(bin_centers, hist + offset, label=f'Scan {scan_value}')
            offset += hist.max() * 0.5  # Offset for the next plot
            max_height = max(max_height, hist.max() + offset)

        # Use the first scan variable to find t0
        reference_scan = scan_var_keys[0]
        reference_data = port_data[reference_scan]
        hist_ref, bins_ref = np.histogram(reference_data, bins=5000, range=(0, 2))
        pks, _ = find_peaks(hist_ref, height=height_t0, distance=distance_t0, prominence=prominence_t0)
        if len(pks) == 0:
            print(f"No t0 found for port {port}.")
            t0s.append(None)
        else:
            t0_bin = pks[0]
            t0 = (bins_ref[t0_bin] + bins_ref[t0_bin + 1]) / 2
            t0s.append(t0)
            ax.axvline(t0, color='red', linestyle='--', label=f't0 = {t0:.4f} µs')

        ax.set_title(f'Run {run}, Port {port}')
        ax.set_xlabel('Time of Flight (µs)')
        ax.set_ylabel('Counts (Offset for clarity)')
        ax.legend()

    # Hide any unused subplots
    total_subplots = 16
    for idx in range(len(ports), total_subplots):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"t0 waterfall plot saved to '{save_path}'.")
    else:
        plt.show()
    plt.close(fig)

    return t0s


def plot_hv_tof_spectra(data_dict, run, ports, tof_bin_width, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.flatten()

    for idx, port in enumerate(ports):
        ax = axes[idx]
        port_data = data_dict.get(port)
        if port_data is None:
            continue

        scan_var_keys = sorted(port_data.keys())
        num_scans = len(scan_var_keys)

        # Determine bin edges
        all_tof = np.concatenate([port_data[scan_value] for scan_value in scan_var_keys if len(port_data[scan_value]) > 0])
        if len(all_tof) == 0:
            print(f"No TOF data for port {port}")
            continue
        tof_min = all_tof.min()
        tof_max = all_tof.max()
        bins = np.arange(tof_min, tof_max + tof_bin_width, tof_bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Create histogram matrix
        hist_matrix = np.zeros((num_scans, len(bin_centers)))
        for i, scan_value in enumerate(scan_var_keys):
            tof_data = port_data[scan_value]
            if len(tof_data) == 0:
                hist_matrix[i] = np.zeros(len(bin_centers))
                continue
            hist, _ = np.histogram(tof_data, bins=bins)
            hist_matrix[i] = hist

        # Plot the summed histogram
        summed_hist = np.sum(hist_matrix, axis=0)
        ax.plot(bin_centers, summed_hist, drawstyle='steps-mid', label=f'Port {port}')
        ax.set_title(f'Run {run}, Port {port}')
        ax.set_xlabel('Time of Flight (µs)')
        ax.set_ylabel('Counts')
        ax.legend()

    # Hide any unused subplots
    total_subplots = 16
    for idx in range(len(ports), total_subplots):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"TOF spectra plot saved to '{save_path}'.")
    else:
        plt.show()
    plt.close(fig)



def load_and_preprocess_data_photon_energy(run, ports, sample_size=None):
    import numpy as np
    import h5py
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


def convert_data_to_energy_photon_energy(data_dict, t0s, ports, retardation, batch_size=2048):
    data_dict_energy = {}
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
            data_tof = data - t0
            data_tof = data_tof[data_tof > 0]  # Keep positive TOF values
            energy_data = convert_tof_to_energy(data_tof, retardation=retardation, batch_size=batch_size)
            data_dict_energy[port][scan_value] = energy_data
    return data_dict_energy


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
    args = parser.parse_args()

    # Process retardation values
    if len(args.retardations) == 1:
        retardations = args.retardations * len(args.runs)
    elif len(args.retardations) == len(args.runs):
        retardations = args.retardations
    else:
        print("Error: Number of retardation values must be either 1 or match the number of runs.")
        return

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

            # Find t0
            t0_save_path = os.path.join(args.save_path, f"run{run}_t0_waterfall.pdf") if args.save_path else None
            t0s = find_t0_waterfall(
                data_dict=data_dict,
                run=run,
                ports=args.ports,
                height_t0=args.height_t0,
                distance_t0=args.distance_t0,
                prominence_t0=args.prominence_t0,
                save_path=t0_save_path
            )

            # Convert to energy
            data_dict_energy = convert_data_to_energy_photon_energy(
                data_dict, t0s, args.ports, retardation, batch_size=args.batch_size
            )

            # Plot HV TOF spectra
            if args.plotting:
                tof_spectra_save_path = os.path.join(args.save_path,
                                                     f"run{run}_hv_tof_spectra.pdf") if args.save_path else None
                plot_hv_tof_spectra(
                    data_dict=data_dict,
                    run=run,
                    ports=args.ports,
                    tof_bin_width=args.tof_bin_width,
                    save_path=tof_spectra_save_path
                )

                # Plot HV Energy spectra
                energy_spectra_save_path = os.path.join(args.save_path,
                                                        f"run{run}_hv_energy_spectra.pdf") if args.save_path else None
                plot_hv_energy_spectra(
                    data_dict_energy=data_dict_energy,
                    run=run,
                    ports=args.ports,
                    energy_bin_width=args.energy_bin_width,
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
                    save_path=t0_save_path
                )
                # Subtract t0s and mask negative values again
                data_dict = subtract_t0(data_dict, t0s, args.ports)
            elif args.t0s is not None:
                if len(args.t0s) != len(args.ports):
                    print("Error: Number of t0s must match number of ports.")
                    continue
                t0s = args.t0s
                # Subtract t0s and mask negative values again
                data_dict = subtract_t0(data_dict, t0s, args.ports)
            else:
                print("Error: t0s must be provided either via --t0s or by using --find_t0.")
                continue

            # Check if the converted HDF5 file exists
            data_dict_energy = {}
            if args.save_data_path:
                save_file = os.path.join(args.save_data_path, f"run{run}_converted.h5")
                converted_file_exists = os.path.exists(save_file)
            else:
                converted_file_exists = False

            if converted_file_exists and not args.overwrite:
                print(f"Converted data for run {run} already exists at '{save_file}'. Loading data.")
                # Load data_dict_energy from the file
                with h5py.File(save_file, 'r') as hf:
                    for port in args.ports:
                        dataset_name = f'pks_{port}'
                        if dataset_name in hf:
                            data_dict_energy[port] = hf[dataset_name][()]
                        else:
                            data_dict_energy[port] = None
            else:
                # Either overwrite is set, or the file does not exist
                # Convert to energy and save data
                data_dict_energy = {}
                for port in args.ports:
                    data = data_dict.get(port)
                    if data is not None and len(data) > 0:
                        print(f"Converting data for port {port} to energy...")
                        batch_size = args.batch_size
                        energy_data = convert_tof_to_energy(data, retardation=retardation, batch_size=batch_size)
                        data_dict_energy[port] = energy_data
                    else:
                        data_dict_energy[port] = None

                # Save converted energy data to HDF5 file
                if args.save_data_path:
                    with h5py.File(save_file, 'w') as hf:
                        for port in args.ports:
                            energy_data = data_dict_energy.get(port)
                            if energy_data is not None:
                                hf.create_dataset(f'pks_{port}', data=energy_data)
                                print(f"Saved energy data for port {port} to '{save_file}'.")
                    print(f"All energy data saved to '{save_file}'.")

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
                    window_range=args.tof_window_range,
                    height=args.height,
                    distance=args.distance,
                    prominence=args.prominence,
                    bin_width=args.tof_bin_width,
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
                    window_range=args.energy_window_range,
                    height=args.height,
                    distance=args.distance,
                    prominence=args.prominence,
                    bin_width=args.energy_bin_width,
                    energy_flag=True,
                    save_path=energy_save_path
                )

if __name__ == '__main__':
    main()