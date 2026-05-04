import oracledb
import csv
import os
import argparse
import pandas as pd
import numpy as np
import mne
from datetime import datetime
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


def process_all_datasets(input_folder):
    """Process all EDF/TSV pairs and insert into Oracle DB."""
    
    conn = oracledb.connect(
        user="nch_sleep_dw",
        password="database",
        dsn="localhost:1521/FREE"
    )
    cursor = conn.cursor()

    genders = set()
    races = set()
    hispanics = set()

    with open("Dataset/DEMOGRAPHIC.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Lấy dữ liệu
            gender_cd = row["PCORI_GENDER_CD"]
            gender_descr = row["GENDER_DESCR"]

            race_cd = row["PCORI_RACE_CD"]
            race_descr = row["RACE_DESCR"]

            hispanic_cd = row["PCORI_HISPANIC_CD"]
            ethnicity_descr = row["ETHNICITY_DESCR"]

            # Thêm vào set (tránh trùng)
            genders.add((gender_cd, gender_descr))
            races.add((race_cd, race_descr))
            hispanics.add((hispanic_cd, ethnicity_descr))

    # Insert dim_gender
    for g in genders:
        try:
            cursor.execute(
                "INSERT INTO dim_gender (gender_cd, gender_descr) VALUES (:1, :2)",
                g
            )
        except:
            pass  # bỏ qua nếu trùng

    # Insert dim_race
    for r in races:
        try:
            cursor.execute(
                "INSERT INTO dim_race (race_cd, race_descr) VALUES (:1, :2)",
                r
            )
        except:
            pass

    # Insert dim_hispanic
    for h in hispanics:
        try:
            cursor.execute(
                "INSERT INTO dim_hispanic (hispanic_cd, ethnicity_descr) VALUES (:1, :2)",
                h
            )
        except:
            pass

    conn.commit()

    # Insert dim_patient
    with open("Dataset/DEMOGRAPHIC.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            try:
                study_pat_id = int(row["STUDY_PAT_ID"])

                # Convert date
                birth_date = datetime.strptime(row["BIRTH_DATE"], "%m/%d/%Y")

                cursor.execute("""
                    INSERT INTO dim_patient (
                        study_pat_id,
                        birth_date,
                        gender_cd,
                        race_cd,
                        hispanic_cd,
                        language_descr
                    ) VALUES (
                        :1,
                        :2,
                        :3,
                        :4,
                        :5,
                        :6
                    )
                """, (
                    study_pat_id,
                    birth_date,
                    row["PCORI_GENDER_CD"],
                    row["PCORI_RACE_CD"],
                    row["PCORI_HISPANIC_CD"],
                    row["LANGUAGE_DESCR"]
                ))

            except Exception as e:
                print("Error row:", row, e)

    conn.commit()

    print('Starting batch processing...')

    for filename in os.listdir(input_folder):
        if filename.endswith('.edf'):
            base_name = filename.replace('.edf', '')
            edf_path = os.path.join(input_folder, filename)
            tsv_path = os.path.join(input_folder, base_name + '.tsv')

            if not os.path.exists(tsv_path):
                continue

            print(f'Processing file: {base_name}')

            try:
                # 👉 lấy patient_id từ tên file (tùy dataset bạn)
                study_pat_id = int(base_name.split("_")[0])
                sleep_study_id = int(base_name.split("_")[1])

                X_file, y_file = process_single_file(edf_path, tsv_path)

                # ---- build DataFrame như cũ ----
                all_feature_cols = [
                    'Mean','Variance','Skewness','Kurtosis',
                    'Power_Delta','Power_Theta','Power_Alpha','Power_Beta',
                    'Theta_Alpha_Ratio','Spectral_Entropy','Rolling_Var',
                    'Delta_diff','Theta_diff','Alpha_diff','Beta_diff',
                    'Delta_Theta_ratio','Alpha_Beta_ratio',
                    'Rolling_Mean_Delta','Rolling_Mean_Theta'
                ]

                df = pd.DataFrame(X_file, columns=all_feature_cols)
                df['Label'] = y_file

                # ---- insert DB ----
                data = []

                for idx, row in df.iterrows():
                    time_id = sleep_study_id * 100000 + idx

                    data.append((
                        study_pat_id,
                        time_id,
                        row['Mean'],
                        row['Variance'],
                        row['Skewness'],
                        row['Kurtosis'],
                        row['Power_Delta'],
                        row['Power_Theta'],
                        row['Power_Alpha'],
                        row['Power_Beta'],
                        row['Theta_Alpha_Ratio'],
                        row['Spectral_Entropy'],
                        row['Rolling_Var'],
                        row['Delta_diff'],
                        row['Theta_diff'],
                        row['Alpha_diff'],
                        row['Beta_diff'],
                        row['Delta_Theta_ratio'],
                        row['Alpha_Beta_ratio'],
                        row['Rolling_Mean_Delta'],
                        row['Rolling_Mean_Theta'],
                        int(row['Label'])
                    ))

                # ---- insert dim_time ----
                time_data = [(sleep_study_id * 100000 + i, i, i*30) for i in range(len(df))]

                cursor.executemany("""
                    INSERT INTO dim_time (time_id, epoch_index, seconds_from_start)
                    VALUES (:1, :2, :3)
                """, time_data)

                # ---- insert fact ----
                cursor.executemany("""
                    INSERT INTO fact_eeg_features (
                        study_pat_id, time_id,
                        mean, variance, skewness, kurtosis,
                        power_delta, power_theta, power_alpha, power_beta,
                        theta_alpha_ratio, spectral_entropy, rolling_var,
                        delta_diff, theta_diff, alpha_diff, beta_diff,
                        delta_theta_ratio, alpha_beta_ratio,
                        rolling_mean_delta, rolling_mean_theta,
                        label
                    ) VALUES (
                        :1,:2,
                        :3,:4,:5,:6,
                        :7,:8,:9,:10,
                        :11,:12,:13,
                        :14,:15,:16,:17,
                        :18,:19,
                        :20,:21,
                        :22
                    )
                """, data)

                conn.commit()

            except Exception as e:
                print(f'Error processing {base_name}: {e}')

    cursor.close()
    conn.close()

    print('Processing complete! Data inserted into Oracle.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='Dataset')
    args = parser.parse_args()

    process_all_datasets(args.input)


if __name__ == '__main__':
    main()
