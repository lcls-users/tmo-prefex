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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process multiple runs and plot TOF and energy spectra.")
    parser.add_argument('--runs', nargs='+', type=int, required=True, help='List of run numbers (e.g., 10 11 12).')
    parser.add_argument('--retardations', nargs='+', type=float, required=True,
                        help='Retardation value(s) for the runs.')
    parser.add_argument('--ports', nargs='+', type=int, required=True, help='List of ports (e.g., 0 1 2).')
    parser.add_argument('--t0s', nargs='+', type=float, help='List of t0 values for each port.')
    parser.add_argument('--find_t0', action='store_true', help='If set, find t0 values.')
    parser.add_argument('--plotting', action='store_true', help='If set, generate and save plots.')
    parser.add_argument('--save_path', type=str, default=None, help='Base path to save the plots (optional).')
    parser.add_argument('--save_data_path', type=str, default=None,
                        help='Path to save the converted energy data (optional).')
    parser.add_argument('--tof_window_range', nargs=2, type=float, default=None,
                        help='Window range for TOF plotting and peak finding (e.g., 0 2).')
    parser.add_argument('--energy_window_range', nargs=2, type=float, default=None,
                        help='Window range for energy plotting and peak finding (e.g., 0 300).')
    parser.add_argument('--tof_bin_width', type=float, default=0.0025, help='Bin width for TOF histograms.')
    parser.add_argument('--energy_bin_width', type=float, default=1.0, help='Bin width for energy histograms.')
    parser.add_argument('--height', type=float, default=0.1,
                        help='Minimum height of peaks (after normalization, between 0 and 1).')
    parser.add_argument('--distance', type=float, default=None,
                        help='Minimum distance between peaks in number of bins (optional).')
    parser.add_argument('--prominence', type=float, default=0.05,
                        help='Minimum prominence of peaks (after normalization, between 0 and 1).')
    parser.add_argument('--height_t0', type=float, default=100, help='Minimum height of peaks for t0 finding.')
    parser.add_argument('--distance_t0', type=float, default=20, help='Minimum distance between peaks for t0 finding.')
    parser.add_argument('--prominence_t0', type=float, default=20, help='Minimum prominence of peaks for t0 finding.')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size for processing data (optional)')
    parser.add_argument('--sample_size', type=int, default=None, help='Subsample size. If not provided, get all data.')
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
        # Get the retardation for this run
        retardation = retardations[idx]

        # Load and preprocess data
        data_dict = load_and_preprocess_data(run, args.ports, sample_size=args.sample_size)
        if data_dict is None:
            continue

        # Prepare to save plots to a single PDF per run
        pdf_save_path = os.path.join(args.save_path, f"run{run}.pdf") if args.save_path else None
        if pdf_save_path:
            pdf = PdfPages(pdf_save_path)
        else:
            pdf = None

        # Find t0s if needed
        t0s = None
        if args.find_t0:
            # Generate t0 plot and save to PDF
            fig_t0, t0s = find_t0(
                data_dict=data_dict,
                run=run,
                retardation=retardation,
                ports=args.ports,
                height_t0=args.height_t0,
                distance_t0=args.distance_t0,
                prominence_t0=args.prominence_t0,
                save_path=None  # We'll save it to PDF directly
            )
            # Subtract t0s and mask negative values again
            data_dict = subtract_t0(data_dict, t0s, args.ports)
            if pdf:
                pdf.savefig(fig_t0)
                plt.close(fig_t0)
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

        # Generate plots if plotting flag is set
        if args.plotting:
            # Plot TOF spectra
            fig_tof = plot_spectra(
                data_dict=data_dict,
                run=run,
                retardation=[retardation]*len(args.ports),
                t0s=t0s,
                ports=args.ports,
                window_range=args.tof_window_range,
                height=args.height,
                distance=args.distance,
                prominence=args.prominence,
                bin_width=args.tof_bin_width,
                energy_flag=False,
                save_path=None  # We'll save it to PDF directly
            )
            if pdf:
                pdf.savefig(fig_tof)
                plt.close(fig_tof)

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
                    save_file = os.path.join(args.save_data_path, f"run{run}_converted.h5")
                    with h5py.File(save_file, 'w') as hf:
                        for port in args.ports:
                            energy_data = data_dict_energy.get(port)
                            if energy_data is not None:
                                hf.create_dataset(f'pks_{port}', data=energy_data)
                                print(f"Saved energy data for port {port} to '{save_file}'.")
                    print(f"All energy data saved to '{save_file}'.")

                # Generate energy spectra plots
                if args.plotting:
                    fig_energy = plot_spectra(
                        data_dict=data_dict_energy,
                        run=run,
                        retardation=[retardation]*len(args.ports),
                        t0s=t0s,
                        ports=args.ports,
                        window_range=args.energy_window_range,
                        height=args.height,
                        distance=args.distance,
                        prominence=args.prominence,
                        bin_width=args.energy_bin_width,
                        energy_flag=True,
                        save_path=None  # We'll save it to PDF directly
                    )
                    if pdf:
                        pdf.savefig(fig_energy)
                        plt.close(fig_energy)

                # Close the PDF file after saving all figures
                if pdf:
                    pdf.close()
                    print(f"All plots saved to '{pdf_save_path}'.")

        if __name__ == '__main__':
            main()


