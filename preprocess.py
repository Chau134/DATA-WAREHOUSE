import os
import argparse
import pandas as pd
import numpy as np
import mne
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch


def bandpower(data, sf, band):
    """Calculate the absolute bandpower of a signal using Welch's method."""
    band = np.asarray(band)
    low, high = band

    freqs, psd = welch(data, sf, nperseg=int(sf * 2))
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return np.trapezoid(psd[idx_band], freqs[idx_band])


def create_epochs_and_normalize(signal, samples_per_epoch):
    """Split signal into 30s epochs and apply per-epoch Z-score normalization."""
    epochs = []
    total_samples = len(signal)

    for start in range(0, total_samples, samples_per_epoch):
        end = start + samples_per_epoch
        epoch_data = signal[start:end]

        if len(epoch_data) < samples_per_epoch:
            pad_length = samples_per_epoch - len(epoch_data)
            epoch_data = np.pad(epoch_data, (0, pad_length), mode='reflect')

        epoch_mean = np.mean(epoch_data)
        epoch_std = np.std(epoch_data)
        if epoch_std > 0:
            epoch_data = (epoch_data - epoch_mean) / epoch_std
        else:
            epoch_data = epoch_data - epoch_mean

        epochs.append(epoch_data)

    return np.array(epochs)


def spectral_entropy(psd, normalize=True):
    psd_norm = psd / np.sum(psd)
    se = entropy(psd_norm)
    if normalize:
        se /= np.log2(len(psd_norm))
    return se


def calc_rolling_feature(epoch, sfreq, window_sec=2):
    window_size = int(sfreq * window_sec)
    rolling_rms = pd.Series(epoch).rolling(window=window_size).apply(
        lambda x: np.sqrt(np.mean(x**2)), raw=True
    )
    return np.nanvar(rolling_rms)


def _load_epochs_from_edf(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    raw.pick(['EEG C3-M2'])
    raw.load_data()
    raw.filter(l_freq=0.5, h_freq=30.0, verbose=False)

    sfreq = raw.info['sfreq']
    samples_per_epoch = int(sfreq * 30)
    data = raw.get_data()[0]
    epochs_array = create_epochs_and_normalize(data, samples_per_epoch)
    return epochs_array, sfreq


def _build_feature_matrix(epochs_array, sfreq):
    features = []
    for epoch in epochs_array:
        freqs, psd = welch(epoch, sfreq, nperseg=int(sfreq * 2))

        mean_v = np.mean(epoch)
        var_v = np.var(epoch)
        skew_v = skew(epoch)
        kurt_v = kurtosis(epoch)

        dp = bandpower(epoch, sfreq, [0.5, 4])
        tp = bandpower(epoch, sfreq, [4, 8])
        ap = bandpower(epoch, sfreq, [8, 13])
        bp = bandpower(epoch, sfreq, [13, 30])

        theta_alpha_ratio = tp / (ap + 1e-6)
        spec_entropy = spectral_entropy(psd[freqs <= 30])
        roll_var = calc_rolling_feature(epoch, sfreq, window_sec=2)

        features.append([
            mean_v,
            var_v,
            skew_v,
            kurt_v,
            dp,
            tp,
            ap,
            bp,
            theta_alpha_ratio,
            spec_entropy,
            roll_var,
        ])

    base_feature_cols = [
        'Mean', 'Variance', 'Skewness', 'Kurtosis',
        'Power_Delta', 'Power_Theta', 'Power_Alpha', 'Power_Beta',
        'Theta_Alpha_Ratio', 'Spectral_Entropy', 'Rolling_Var'
    ]

    df_local = pd.DataFrame(np.array(features), columns=base_feature_cols)

    df_local['Delta_diff'] = df_local['Power_Delta'].diff().fillna(0)
    df_local['Theta_diff'] = df_local['Power_Theta'].diff().fillna(0)
    df_local['Alpha_diff'] = df_local['Power_Alpha'].diff().fillna(0)
    df_local['Beta_diff'] = df_local['Power_Beta'].diff().fillna(0)
    df_local['Delta_Theta_ratio'] = df_local['Power_Delta'] / (df_local['Power_Theta'] + 1e-6)
    df_local['Alpha_Beta_ratio'] = df_local['Power_Alpha'] / (df_local['Power_Beta'] + 1e-6)
    df_local['Rolling_Mean_Delta'] = df_local['Power_Delta'].rolling(5, min_periods=1).mean()
    df_local['Rolling_Mean_Theta'] = df_local['Power_Theta'].rolling(5, min_periods=1).mean()

    derived_cols = [
        'Delta_diff', 'Theta_diff', 'Alpha_diff', 'Beta_diff',
        'Delta_Theta_ratio', 'Alpha_Beta_ratio',
        'Rolling_Mean_Delta', 'Rolling_Mean_Theta'
    ]

    return df_local[base_feature_cols + derived_cols].values


def process_edf_file(edf_path):
    """Read EDF file and extract 19-feature matrix for all epochs."""
    epochs_array, sfreq = _load_epochs_from_edf(edf_path)
    return _build_feature_matrix(epochs_array, sfreq)


def process_single_file(edf_path, tsv_path):
    """Read EDF/TSV pair, extract features, and return X/y for labeled epochs."""
    epochs_array, sfreq = _load_epochs_from_edf(edf_path)

    annotations_df = pd.read_csv(tsv_path, sep='\t')
    annotations_df.columns = [str(col).strip().lower() for col in annotations_df.columns]

    required_cols = {'onset', 'duration', 'description'}
    missing_cols = required_cols - set(annotations_df.columns)
    if missing_cols:
        raise ValueError(
            f"TSV file `{tsv_path}` is missing required columns: {sorted(missing_cols)}"
        )

    ann2label = {
        'Sleep stage W': 0,
        'Sleep stage N1': 1,
        'Sleep stage N2': 2,
        'Sleep stage N3': 3,
        'Sleep stage R': 4,
    }

    y = np.full(len(epochs_array), -1)
    for _, row in annotations_df.iterrows():
        desc = str(row['description']).strip()
        if desc in ann2label:
            onset = float(row['onset'])
            duration = float(row['duration'])
            start_ep = int(onset // 30)
            end_ep = int((onset + duration) // 30)
            for ep_idx in range(start_ep, end_ep + 1):
                if ep_idx < len(epochs_array):
                    y[ep_idx] = ann2label[desc]

    X_full = _build_feature_matrix(epochs_array, sfreq)

    valid_indices = y != -1
    return X_full[valid_indices], y[valid_indices]


def process_all_datasets(input_folder, output_folder):
    """Process all EDF/TSV pairs in a folder and save one aggregated CSV."""
    all_X = []
    all_y = []

    print('Starting batch processing...')

    for filename in os.listdir(input_folder):
        if filename.endswith('.edf'):
            base_name = filename.replace('.edf', '')
            edf_path = os.path.join(input_folder, filename)
            tsv_path = os.path.join(input_folder, base_name + '.tsv')

            if os.path.exists(tsv_path):
                print(f'Processing file: {base_name}')
                try:
                    X_file, y_file = process_single_file(edf_path, tsv_path)
                    all_X.append(X_file)
                    all_y.append(y_file)
                except Exception as e:
                    print(f'Error processing {base_name}: {e}')

    if not all_X:
        raise RuntimeError('No data processed. Check input folder and file pairs.')

    X_final = np.vstack(all_X)
    y_final = np.concatenate(all_y)

    # base extracted features (from epoch-level calculations)
    base_feature_cols = [
        'Mean',
        'Variance',
        'Skewness',
        'Kurtosis',
        'Power_Delta',
        'Power_Theta',
        'Power_Alpha',
        'Power_Beta',
        'Theta_Alpha_Ratio',
        'Spectral_Entropy',
        'Rolling_Var',
    ]

    df = pd.DataFrame(X_final, columns=base_feature_cols)
    df['Label'] = y_final

    # ===== Derived features to match existing classifiers =====
    # Compute diffs, ratios and short rolling means across the aggregated dataset
    df['Delta_diff'] = df['Power_Delta'].diff().fillna(0)
    df['Theta_diff'] = df['Power_Theta'].diff().fillna(0)

    df['Alpha_diff'] = df['Power_Alpha'].diff().fillna(0)
    df['Beta_diff'] = df['Power_Beta'].diff().fillna(0)

    df['Delta_Theta_ratio'] = df['Power_Delta'] / (df['Power_Theta'] + 1e-6)
    df['Alpha_Beta_ratio'] = df['Power_Alpha'] / (df['Power_Beta'] + 1e-6)

    df['Rolling_Mean_Delta'] = df['Power_Delta'].rolling(5, min_periods=1).mean()
    df['Rolling_Mean_Theta'] = df['Power_Theta'].rolling(5, min_periods=1).mean()

    # final column ordering: base features followed by derived features (19 total)
    derived_cols = [
        'Delta_diff', 'Theta_diff', 'Alpha_diff', 'Beta_diff',
        'Delta_Theta_ratio', 'Alpha_Beta_ratio',
        'Rolling_Mean_Delta', 'Rolling_Mean_Theta'
    ]

    final_cols = base_feature_cols + derived_cols + ['Label']
    df = df[final_cols]

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'all_subjects_features.csv')
    df.to_csv(output_path, index=False)

    print(f'\nProcessing complete! Final dataset saved to: {output_path}')
    print(f'Total number of epochs collected: {len(df)}')


def main():
    parser = argparse.ArgumentParser(description='Process EDF/TSV dataset into feature CSV')
    parser.add_argument('--input', '-i', default='Dataset', help='Input folder containing EDF/TSV pairs')
    parser.add_argument('--output', '-o', default='Processed_Data', help='Output folder for generated CSV')
    args = parser.parse_args()

    process_all_datasets(args.input, args.output)


if __name__ == '__main__':
    main()
