#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
from matplotlib.backends.backend_pdf import PdfPages

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot energy histograms and find peaks.")
    parser.add_argument('--runs', nargs='+', required=True, help='List of run numbers (e.g., 35 36 37)')
    parser.add_argument('--ports', nargs='+', required=True, help='List of ports (e.g., 0 1 2)')
    parser.add_argument('--retardations', nargs='+', required=True, help='List of retardation values corresponding to runs')
    parser.add_argument('--energy_range', nargs=2, type=float, required=True, help='Energy range to plot and find peaks (e.g., 0 300)')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the PDF file (optional)')
    parser.add_argument('--height', type=float, default=None, help='Minimum height of peaks (optional)')
    parser.add_argument('--distance', type=float, default=None, help='Minimum distance between peaks in number of bins (optional)')
    parser.add_argument('--prominence', type=float, default=None, help='Minimum prominence of peaks (optional)')
    parser.add_argument('--bin_width', type=float, default=1.0, help='Bin width for histograms (optional, default=1 eV)')
    args = parser.parse_args()

    # Convert runs, ports, retardations to appropriate types
    runs = [int(run) for run in args.runs]
    ports = [int(port) for port in args.ports]
    retardations = [float(r) for r in args.retardations]

    if len(retardations) != len(runs):
        print("Error: Number of retardations must match number of runs.")
        return

    energy_min, energy_max = args.energy_range

    # Peak finding parameters
    height = args.height
    distance = args.distance
    prominence = args.prominence

    # Bin width for histograms
    bin_width = args.bin_width

    # Path to the converted data files
    data_path = '/sdf/scratch/users/a/ajshack/tmox1016823/energy_data'

    # Initialize list to collect figures if saving to PDF
    figures = []

    # For each run and port
    for run_idx, run_num in enumerate(runs):
        retardation = retardations[run_idx]
        for port in ports:
            # Load energy data
            energy_data = load_energy_data(data_path, run_num, port)
            if energy_data is None:
                print(f"No data found for run {run_num}, port {port}. Skipping.")
                continue

            # Apply energy range
            energy_filtered = energy_data[(energy_data >= energy_min) & (energy_data <= energy_max)]
            if energy_filtered.size == 0:
                print(f"No data in the specified energy range for run {run_num}, port {port}. Skipping.")
                continue

            # Plot histogram and find peaks
            fig = plot_histogram_with_peaks(energy_filtered, run_num, port, retardation,
                                            energy_min, energy_max, bin_width,
                                            height, distance, prominence)
            if args.save_path:
                figures.append(fig)
            else:
                plt.show()

    # Save all figures to PDF if save_path is specified
    if args.save_path and figures:
        with PdfPages(args.save_path) as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Plots saved to '{args.save_path}'.")

def load_energy_data(data_path, run_num, port):
    # Construct file path
    file_path = os.path.join(data_path, f'run{run_num}_converted.h5')
    dataset_name = f'energy_{port}'
    if not os.path.exists(file_path):
        print(f"File '{file_path}' does not exist.")
        return None
    # Load data
    with h5py.File(file_path, 'r') as hf:
        if dataset_name not in hf.keys():
            print(f"Dataset '{dataset_name}' not found in '{file_path}'.")
            return None
        energy_data = hf[dataset_name][()]
    return energy_data

def plot_histogram_with_peaks(energy_data, run_num, port, retardation,
                              energy_min, energy_max, bin_width,
                              height, distance, prominence):
    # Create histogram
    bins = np.arange(energy_min, energy_max + bin_width, bin_width)
    counts, bin_edges = np.histogram(energy_data, bins=bins)

    # Find peaks in the histogram
    peaks, properties = find_peaks(counts, height=height, distance=distance, prominence=prominence)

    # Create figure and plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centers, counts, width=bin_width, align='center', alpha=0.7, edgecolor='black')

    # Mark peaks
    peak_energies = bin_centers[peaks]
    peak_counts = counts[peaks]
    ax.vlines(peak_energies, ymin=0, ymax=peak_counts, color='red', linestyle='dashed', label='Peaks')

    # Annotate peaks with energy values
    # Adjust text to prevent overlap
    texts = []
    for i, (peak_energy, peak_count) in enumerate(zip(peak_energies, peak_counts)):
        # Calculate vertical offset for text to prevent overlap
        if i > 0 and abs(peak_energy - peak_energies[i - 1]) < bin_width * 5:
            # If peaks are close, offset the text vertically
            offset = 15
        else:
            offset = 5
        text = ax.annotate(f'{peak_energy:.2f} eV', xy=(peak_energy, peak_count),
                           xytext=(0, offset), textcoords='offset points',
                           ha='center', va='bottom', fontsize=12, color='red')
        texts.append(text)

    # Adjust font sizes
    ax.set_title(f'Energy Histogram with Peaks\nRun {run_num}, Port {port}, Retardation {retardation}',
                 fontsize=20)
    ax.set_xlabel('Energy (eV)', fontsize=20)
    ax.set_ylabel('Counts', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(energy_min, energy_max)
    ax.legend(fontsize=14)

    plt.tight_layout()
    return fig

if __name__ == '__main__':
    main()
