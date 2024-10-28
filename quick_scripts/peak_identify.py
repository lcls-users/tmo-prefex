#!/usr/bin/env python3

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import os
from matplotlib.backends.backend_pdf import PdfPages
import sys
import tensorflow as tf
import pickle  # Make sure to import pickle if using scalers

# Add the path to your custom model modules if needed
sys.path.append('/sdf/home/a/ajshack/TOF_ML/src')
from models.tof_to_energy_model import TofToEnergyModel, InteractionLayer, ScalingLayer, LogTransformLayer


def gaussian(x, A, mu, sigma):
    """Gaussian function."""
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def load_scalers(scalers_path):
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
                              height, distance, prominence,
                              t0_energy):
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
    # Rotate text by 90 degrees and use leader lines
    for i, (peak_energy, peak_count) in enumerate(zip(peak_energies, peak_counts)):
        # Annotate peak with energy
        ax.annotate(f'{peak_energy:.2f} eV',
                    xy=(peak_energy, peak_count),
                    xytext=(0, 50), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12, color='red',
                    rotation=90,
                    arrowprops=dict(arrowstyle='->', color='red'))

    # Plot t0 energy if provided and within range
    if t0_energy is not None and energy_min <= t0_energy <= energy_max:
        ax.axvline(x=t0_energy, color='black', linestyle='-', label='Light Peak')
        ax.text(t0_energy, ax.get_ylim()[1]*0.9, f'Light Peak at {t0_energy:.2f} eV',
                rotation=90, verticalalignment='top', horizontalalignment='right', fontsize=12, color='black')

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


def plot_histogram_with_peaks_and_fits(energy_data, run_num, port, retardation,
                                       energy_min, energy_max, bin_width,
                                       height, distance, prominence,
                                       t0_energy):
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

    # Annotate peaks with energy values and fit Gaussians
    for i, (peak_energy, peak_count) in enumerate(zip(peak_energies, peak_counts)):
        # Gaussian fitting around the peak
        try:
            # Define window around the peak for fitting
            window_size = int(5 / bin_width)  # 5 eV on each side
            start = max(0, peaks[i] - window_size)
            end = min(len(counts), peaks[i] + window_size)
            x_fit = bin_centers[start:end]
            y_fit = counts[start:end]

            # Initial guesses for A, mu, sigma
            A_guess = peak_count
            mu_guess = peak_energy
            sigma_guess = bin_width * 2  # Rough estimate

            popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=[A_guess, mu_guess, sigma_guess])

            # Extract fitted parameters
            A_fit, mu_fit, sigma_fit = popt
            fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(sigma_fit)

            # Plot fitted Gaussian
            x_gauss = np.linspace(x_fit[0], x_fit[-1], 100)
            y_gauss = gaussian(x_gauss, *popt)
            ax.plot(x_gauss, y_gauss, color='green', linestyle='--', label='Gaussian Fit' if i == 0 else None)

            # Annotate peak with energy and FWHM
            # Rotate text by 90 degrees and use leader lines
            ax.annotate(f'{mu_fit:.2f} eV\nFWHM: {fwhm:.2f} eV',
                        xy=(mu_fit, A_fit),
                        xytext=(0, 50), textcoords='offset points',
                        ha='center', va='bottom', fontsize=12, color='blue',
                        rotation=90,
                        arrowprops=dict(arrowstyle='->', color='blue'))
        except RuntimeError:
            print(f"Could not fit Gaussian for peak at {peak_energy:.2f} eV in run {run_num}, port {port}.")

    # Plot t0 energy if provided and within range
    if t0_energy is not None and energy_min <= t0_energy <= energy_max:
        ax.axvline(x=t0_energy, color='black', linestyle='-', label='Light Peak')
        ax.text(t0_energy, ax.get_ylim()[1]*0.9, f'Light Peak at {t0_energy:.2f} eV',
                rotation=90, verticalalignment='top', horizontalalignment='right', fontsize=12, color='black')

    # Adjust font sizes
    ax.set_title(f'Energy Histogram with Peaks and Gaussian Fits\nRun {run_num}, Port {port}, Retardation {retardation}',
                 fontsize=20)
    ax.set_xlabel('Energy (eV)', fontsize=20)
    ax.set_ylabel('Counts', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(energy_min, energy_max)
    ax.legend(fontsize=14)

    plt.tight_layout()
    return fig


def convert_t0_to_energy(t0, retardation, model):
    """Converts t0 to energy using the provided model."""
    # Prepare input array for the model
    # Note: Adjust the input features as per your model's requirements
    # For example, using default values for mid1_ratio_col, mid2_ratio_col
    hist_t0 = np.array([t0]) * 1e6  # Convert to microseconds if needed
    retardation_col = np.array([retardation])  # Use the retardation value for the current run
    mid1_ratio_col = np.array([0.11248])
    mid2_ratio_col = np.array([0.1354])
    input_array = np.column_stack([retardation_col, mid1_ratio_col, mid2_ratio_col, hist_t0])

    # Make prediction
    y_pred = model.predict(input_array).flatten()
    energy = 2 ** y_pred[0]  # Apply inverse transformation if needed

    return energy


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot energy histograms, find peaks, and optionally fit Gaussians.")
    parser.add_argument('--runs', nargs='+', required=True, help='List of run numbers (e.g., 35 36 37)')
    parser.add_argument('--ports', nargs='+', required=True, help='List of ports (e.g., 0 1 2)')
    parser.add_argument('--retardations', nargs='+', required=True, help='List of retardation values corresponding to runs')
    parser.add_argument('--energy_range', nargs=2, type=float, required=True, help='Energy range to plot and find peaks (e.g., 0 300)')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the PDF file (optional)')
    parser.add_argument('--height', type=float, default=2000, help='Minimum height of peaks (optional)')
    parser.add_argument('--distance', type=float, default=None, help='Minimum distance between peaks in number of bins (optional)')
    parser.add_argument('--prominence', type=float, default=3, help='Minimum prominence of peaks (optional)')
    parser.add_argument('--bin_width', type=float, default=0.5, help='Bin width for histograms (optional, default=1 eV)')
    parser.add_argument('--t0', type=float, default=None, help='t0 value to convert to energy (optional)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the ML model (required if t0 is specified)')
    parser.add_argument('--scalers_path', type=str, default=None, help='Path to the scalers file (required if t0 is specified)')
    parser.add_argument('--peaks_only', action='store_true', help='If set, only plot peaks without Gaussian fits')
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

    # Load ML model if t0 is specified
    if args.t0 is not None:
        if args.model_path is None or args.scalers_path is None:
            print("Error: --model_path and --scalers_path must be specified when using --t0.")
            return
        # Load the model
        main_model = tf.keras.models.load_model(args.model_path, custom_objects={
            'LogTransformLayer': LogTransformLayer,
            'InteractionLayer': InteractionLayer,
            'ScalingLayer': ScalingLayer,
            'TofToEnergyModel': TofToEnergyModel
        })
        main_model.min_values, main_model.max_values = load_scalers(args.scalers_path)
        params_line = "--batch_size=1024 --dropout=0.2 --layer_size=64 --learning_rate=0.01 --optimizer=Adam"
        main_model.params = parse_params_line(params_line)
    else:
        main_model = None

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

            # Convert t0 to energy for the current run and retardation
            if args.t0 is not None and main_model is not None:
                t0_energy = convert_t0_to_energy(args.t0, retardation, main_model)
                print(f"Converted t0 ({args.t0}) to energy: {t0_energy:.2f} eV for run {run_num}")
            else:
                t0_energy = None

            # Plot histogram and find peaks
            if args.peaks_only:
                fig = plot_histogram_with_peaks(energy_filtered, run_num, port, retardation,
                                                energy_min, energy_max, bin_width,
                                                height, distance, prominence,
                                                t0_energy)
            else:
                fig = plot_histogram_with_peaks_and_fits(energy_filtered, run_num, port, retardation,
                                                         energy_min, energy_max, bin_width,
                                                         height, distance, prominence,
                                                         t0_energy)
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


if __name__ == '__main__':
    main()

