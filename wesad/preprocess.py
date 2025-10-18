import os
import json
import warnings
import numpy as np
import pandas as pd
import neurokit2 as nk

from fire import Fire
from glob import glob
from tqdm import tqdm
from datasets import Dataset
from multiprocessing import Pool
from scipy.signal import butter, filtfilt, welch, find_peaks
from scipy.fft import rfft, rfftfreq

from neurokit2.misc._warnings import NeuroKitWarning

warnings.simplefilter("ignore", NeuroKitWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

WESAD_PATH = "data/WESAD/raw"
OUT_DIR = "data/processed/wesad"
SLIDING_WINDOW = 10  # seconds
ACC_DURATION = 5  # seconds
EMG_DURATION = 5  # seconds
DURATION = 60  # seconds
SAMPLING_RATES = {
    "chest_ACC": 700,
    "chest_ECG": 700,
    "chest_EMG": 700,
    "chest_EDA": 700,
    "chest_TEMP": 700,
    "chest_RESP": 700,
    "wrist_ACC": 32,
    "wrist_BVP": 64,
    "wrist_EDA": 4,
    "wrist_TEMP": 4,
    "label": 700,
}
LABEL_DICT = {
    1: "baseline",
    2: "stress",
    3: "amusement",
}


def store_info(info_path):
    # include ranges (later)
    info = {
        "task": "Classify the user's emotional state into one of the three affective categories: [\"baseline\", \"stress\", \"amusement\"], based on physiological signals collected from wearable sensors.",
        "class": {
            "baseline": "The subject was seated or standing at a table, reading neutral materials such as magazines. This condition aimed to induce a calm, neutral affective state.",
            "stress": "The subject was engaged in the Trier Social Stress Test (TSST), involving public speaking and mental arithmetic under evaluation. This condition reliably induces cognitive and social stress.",
            "amusement": "The subject watched a series of humorous video clips designed to elicit positive emotions such as laughter and amusement."
        },
        "data": (
            "Physiological data was collected using two wearable devices:\n"
            "- chest device: Collected high-resolution (700 Hz) signals including accelerometer (ACC, in g), respiration (RESP, in %), electrocardiogram (ECG, in mV), electrodermal activity (EDA, in μS), electromyogram (EMG, in mV), and skin temperature (TEMP, in °C). The sensors were placed on the chest, abdomen, and upper back muscles to capture both cardiac and muscular activity.\n"
            "- wrist device: Recorded blood volume pulse (BVP, unitless) at 64 Hz, EDA (in μS) and TEMP (in °C) at 4 Hz, and accelerometer data (ACC, in 1/64g) at 32 Hz. It was worn on the non-dominant wrist to monitor peripheral physiological responses.\n"
            "All signals were time-synchronized, and features were computed from the most recent segment of data. A 60-second window was used for most modalities, while a shorter 5-second window was used for accelerometer and EMG frequency features.\n"
            "Each feature is named using the format 'position_modality_featurename', where position indicates the sensor location (e.g., chest, wrist), modality refers to the signal type (e.g., ECG, EDA), and featurename describes the extracted statistic or characteristic (e.g., hrv_rmssd, mag_std, scr_mean)."
        ),
        "modality_data": {
            "chest_ACC": (
                "Physiological data was collected from a chest-worn accelerometer (ACC, in g), recording acceleration across three axes (x, y, z) and the overall magnitude at 700 Hz. "
                "Features were computed from the most recent 5-second window. "
                "Each feature is named using the format 'position_modality_featurename', where 'featurename' describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
            "chest_ECG": (
                "Electrocardiogram (ECG, in mV) signals were recorded at 700 Hz using a chest-worn sensor. Heartbeats were detected via peak detection, and heart rate (hr) and heart rate variability (hrv) features were derived. "
                "Features include statistical measures (mean, std), time-domain HRV metrics (pnn50, tinn, rmssd), and frequency-domain components (ulf, lf, hf, uhf, total_power, lf_hf_ratio), along with their normalized and relative versions. "
                "Features were computed from the most recent 60-second window. "
                "Each feature is named using the format 'position_modality_featurename', where 'featurename' describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
            "chest_EMG": (
                "Electromyogram (EMG, in mV) signals were collected from upper back muscles at 700 Hz. "
                "Two processing chains were used: a high-pass filter followed by 5-second windows for statistical and frequency features (mean, std, range, integral, percentiles, mean/median/peak frequency, band energy from 0–350 Hz); and a low-pass filter over a 60-second window to extract peak-related features (count, amplitude stats, normalized sum). "
                "Each feature is named using the format 'position_modality_featurename', where 'featurename' describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
            "chest_EDA": (
                "Electrodermal activity (EDA, in μS) was recorded from the chest at 700 Hz and low-pass filtered at 5 Hz. "
                "Features were derived from both tonic (SCL) and phasic (SCR) components. "
                "Statistical features (mean, std, min, max, slope, dynamic range) were computed from the raw signal. "
                "From the decomposed signals, features include SCL-time correlation, SCR count, total magnitude and duration, area under the curve (AUC), and more. "
                "Each feature is named using the format 'position_modality_featurename', where 'featurename' describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
            "chest_TEMP": (
                "Skin temperature (TEMP, in °C) was recorded at 700 Hz from the chest. "
                "From the most recent 60-second window, statistical features were extracted including mean, std, min, max, slope, and dynamic range. "
                "Each feature is named using the format 'position_modality_featurename', where 'featurename' describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
            "chest_RESP": (
                "Respiration (RESP, in %) was measured using a plethysmograph at 700 Hz. A bandpass filter (0.1–0.35 Hz) was applied to identify inhale/exhale cycles. "
                "Features include durations and statistics (mean, std) of each phase, inhalation/exhalation ratio, stretch, inspiration volume, respiration rate, and total cycle duration. "
                "Features were extracted from the most recent 60-second window. "
                "Each feature is named using the format 'position_modality_featurename', where 'featurename' describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
            "wrist_ACC": (
                "Wrist-worn accelerometer (ACC, in 1/64 g) data was recorded at 32 Hz. "
                "Acceleration in three axes (x, y, z) and overall magnitude were used to compute features from the most recent 5-second window, including mean, std, absolute integral, and peak frequency per axis. "
                "Each feature is named using the format 'position_modality_featurename', where 'featurename' describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
            "wrist_BVP": (
                "Blood volume pulse (BVP, unitless) was collected at 64 Hz from the wrist. "
                "Peak detection was applied to extract heartbeats, and from this, heart rate (HR) and heart rate variability (HRV) features were derived. "
                "Features include HRV statistics (mean, std, rmssd, tinn, pnn50) and spectral energy across standard frequency bands (ulf, lf, hf, uhf), as well as total power and ratios (e.g., lf_hf_ratio). "
                "Features were extracted from a 60-second window. "
                "Each feature is named using the format 'position_modality_featurename', where 'featurename' describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
            "wrist_EDA": (
                "Electrodermal activity (EDA, in μS) was recorded at 4 Hz from the wrist. "
                "The signal was processed similarly to chest EDA, including low-pass filtering and decomposition into tonic (SCL) and phasic (SCR) components. "
                "Features were derived from both raw and decomposed signals, covering basic statistics, time correlation, SCR frequency, and area-based metrics. "
                "Each feature is named using the format 'position_modality_featurename', where 'featurename' describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
            "wrist_TEMP": (
                "Skin temperature (TEMP, in °C) was recorded at 4 Hz from the wrist. "
                "Statistical features were extracted from the most recent 60-second window, including mean, std, min, max, slope, and dynamic range. "
                "Each feature is named using the format 'position_modality_featurename', where 'featurename' describes the extracted statistic or characteristic (e.g., _mean, _std)."
            ),
        },
        "feature": {
            "ACC": (
                "For raw accelerometer signals, the mean, standard deviation (std), and absolute integral were computed "
                "for each axis (x, y, z) and for the overall magnitude (mag). In addition, the peak frequency was extracted for each axis."
            ),
            "ECG": (
                "Heartbeats were detected via peak detection. From the inter-beat intervals, heart rate (hr) features "
                "including mean and standard deviation were derived. Heart rate variability (hrv) features were also computed, "
                "including: rmssd (root mean square of successive differences), pnn50 (percentage of intervals differing by more than 50 ms), "
                "tinn (triangular interpolation of NN interval histogram), and standard deviation. Frequency-domain HRV features were extracted across the following bands: "
                "ulf (0.01–0.04 Hz), lf (0.04–0.15 Hz), hf (0.15–0.4 Hz), and uhf (0.4–1.0 Hz). Additional features include total power, lf_hf_ratio, "
                "relative power (rel_ulf, rel_lf, rel_hf, rel_uhf), and normalized LF and HF (normalized_lf, normalized_hf)."
            ),
            "BVP": (
                "Using peak detection on the raw PPG signal, heartbeat intervals were identified. The same set of hr and hrv features as for ECG "
                "were extracted, including: mean and standard deviation (std) of hr, rmssd, pnn50, tinn, frequency-domain energy in ulf, lf, hf, uhf bands, "
                "LF/HF ratio, total power, relative power per band, and normalized LF and HF components."
            ),
            "EDA": (
                "The raw EDA signal was first low-pass filtered at 5 Hz. From the filtered signal, basic statistical features were computed: "
                "mean, standard deviation, minimum, maximum, slope (first-to-last value divided by duration), and dynamic range (max − min). "
                "The signal was then decomposed into two components: tonic (skin conductance level, SCL) and phasic (skin conductance response, SCR). "
                "SCL reflects a slow-changing baseline, while SCR represents short-term stimulus-related responses. "
                "From each component, the mean and standard deviation were computed. Additional features include the correlation between SCL and time "
                "(scl_time_corr), number of SCR events (scr_num_segments), sum of SCR magnitudes (scr_sum_magnitudes), total SCR duration "
                "(scr_total_duration), and the area under the SCR curve (scr_auc)."
            ),
            "EMG": (
                "A high-pass filter was applied to remove the DC component. From the resulting signal, features were computed over 5-second windows: "
                "mean, standard deviation, dynamic range, absolute integral, median, 10th percentile, 90th percentile, and frequency-domain features including mean, median, and peak frequency. "
                "Additionally, spectral energy was calculated across seven equally spaced frequency bands from 0 to 350 Hz. "
                "In the second chain, a 50 Hz low-pass filter was applied to the raw EMG signal. From the filtered signal, using a 60-second window, the number of peaks, "
                "mean and standard deviation of peak amplitudes, sum of peak amplitudes, and normalized sum of peak amplitudes were computed."
            ),
            "RESP": (
                "A bandpass filter with cutoff frequencies of 0.1 Hz and 0.35 Hz was applied to the raw respiration signal. "
                "A peak detection algorithm was then used to identify minima and maxima, allowing estimation of inhalation and exhalation durations. "
                "From these, the mean and standard deviation of inhale and exhale durations were computed, along with their ratio (inhale-to-exhale). "
                "Additional features include stretch (difference between maximum and minimum respiratory signal), inspiration volume (area under the inhalation curve), "
                "respiration rate (mean number of breaths per minute), and average respiration cycle duration."
            ),
            "TEMP": (
                "The following statistical features were computed: mean, standard deviation, minimum, maximum, slope (rate of change over time), "
                "and dynamic range (max - min)."
            )
        },
    }
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)


def lowpass_filter(signal, cutoff=50, fs=1000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, signal)


def process_by_mod(mod_name, mod_data):
    sr = SAMPLING_RATES[mod_name]
    features = {}
    if "ACC" in mod_name:
        mean = np.mean(mod_data, axis=0)
        std = np.std(mod_data, axis=0)
        freqs = rfftfreq(mod_data.shape[0], d=1 / sr)
        fft_vals = np.abs(rfft(mod_data - mean, axis=0))
        peak_freq = freqs[np.argmax(fft_vals, axis=0)]
        for i, axis in enumerate(["x", "y", "z"]):
            features[f"{mod_name}_{axis}_mean"] = mean[i]
            features[f"{mod_name}_{axis}_std"] = std[i]
            features[f"{mod_name}_{axis}_peak_frequency"] = peak_freq[i]
            features[f"{mod_name}_{axis}_absolute_integral"] = (
                np.sum(np.abs(mod_data[:, i])) / sr
            )

        mag = np.sqrt(np.sum(np.square(mod_data), axis=1))
        mag_mean = np.mean(mag)
        mag_std = np.std(mag)
        features[f"{mod_name}_mag_mean"] = mag_mean
        features[f"{mod_name}_mag_std"] = mag_std
        features[f"{mod_name}_mag_absolute_integral"] = np.sum(np.abs(mag)) / sr

    elif "ECG" in mod_name:
        ecg_cleaned = nk.ecg_clean(mod_data, sampling_rate=sr)
        # peaks, info = nk.ecg_peaks(mod_data[:, 0], sampling_rate=sr)
        peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=sr)

        hr = nk.ecg_rate(peaks=info["ECG_R_Peaks"], sampling_rate=sr)
        hr_mean = hr.mean()
        hr_std = hr.std()
        features[f"{mod_name}_hr_mean"] = hr_mean
        features[f"{mod_name}_hr_std"] = hr_std

        hrv_time = nk.hrv_time(peaks, sampling_rate=sr, show=False)
        hrv_freq = nk.hrv_frequency(
            peaks,
            sampling_rate=sr,
            ulf=(0, 0.01),
            vlf=(0.01, 0.04),
            lf=(0.04, 0.15),
            hf=(0.15, 0.4),
            vhf=(0.4, 1.0),
            show=False,
        )
        features[f"{mod_name}_hrv_mean"] = hrv_time["HRV_MeanNN"].values[0]
        features[f"{mod_name}_hrv_std"] = hrv_time["HRV_SDNN"].values[0]
        features[f"{mod_name}_hrv_pnn50"] = hrv_time["HRV_pNN50"].values[0]
        features[f"{mod_name}_hrv_tinn"] = hrv_time["HRV_TINN"].values[0]
        features[f"{mod_name}_hrv_rmssd"] = hrv_time["HRV_RMSSD"].values[0]

        features[f"{mod_name}_hrv_ulf"] = hrv_freq["HRV_VLF"].values[0]
        features[f"{mod_name}_hrv_lf"] = hrv_freq["HRV_LF"].values[0]
        features[f"{mod_name}_hrv_hf"] = hrv_freq["HRV_HF"].values[0]
        features[f"{mod_name}_hrv_uhf"] = hrv_freq["HRV_VHF"].values[0]
        features[f"{mod_name}_hrv_lf_hf_ratio"] = hrv_freq["HRV_LFHF"].values[0]
        features[f"{mod_name}_hrv_total_power"] = hrv_freq["HRV_TP"].values[0]
        features[f"{mod_name}_hrv_rel_ulf"] = (
            hrv_freq["HRV_VLF"].values[0] / hrv_freq["HRV_TP"].values[0]
        )
        features[f"{mod_name}_hrv_rel_lf"] = (
            hrv_freq["HRV_LF"].values[0] / hrv_freq["HRV_TP"].values[0]
        )
        features[f"{mod_name}_hrv_rel_hf"] = (
            hrv_freq["HRV_HF"].values[0] / hrv_freq["HRV_TP"].values[0]
        )
        features[f"{mod_name}_hrv_rel_uhf"] = (
            hrv_freq["HRV_VHF"].values[0] / hrv_freq["HRV_TP"].values[0]
        )
        features[f"{mod_name}_hrv_normalized_lf"] = hrv_freq["HRV_LFn"].values[0]
        features[f"{mod_name}_hrv_normalized_hf"] = hrv_freq["HRV_HFn"].values[0]

    elif "BVP" in mod_name:
        bvp_cleaned = nk.ppg_clean(mod_data, sampling_rate=sr)
        _, info = nk.ppg_peaks(bvp_cleaned, sampling_rate=sr)

        hr = nk.ppg_rate(peaks=info["PPG_Peaks"], sampling_rate=sr)
        hr_mean = hr.mean()
        hr_std = hr.std()
        features[f"{mod_name}_hr_mean"] = hr_mean
        features[f"{mod_name}_hr_std"] = hr_std

        hrv_time = nk.hrv_time(info, sampling_rate=sr, show=False)
        hrv_freq = nk.hrv_frequency(
            info,
            sampling_rate=sr,
            ulf=(0, 0.01),
            vlf=(0.01, 0.04),
            lf=(0.04, 0.15),
            hf=(0.15, 0.4),
            vhf=(0.4, 1.0),
            show=False,
        )

        features[f"{mod_name}_hrv_mean"] = hrv_time["HRV_MeanNN"].values[0]
        features[f"{mod_name}_hrv_std"] = hrv_time["HRV_SDNN"].values[0]
        features[f"{mod_name}_hrv_pnn50"] = hrv_time["HRV_pNN50"].values[0]
        features[f"{mod_name}_hrv_tinn"] = hrv_time["HRV_TINN"].values[0]
        features[f"{mod_name}_hrv_rmssd"] = hrv_time["HRV_RMSSD"].values[0]

        features[f"{mod_name}_hrv_ulf"] = hrv_freq["HRV_VLF"].values[0]
        features[f"{mod_name}_hrv_lf"] = hrv_freq["HRV_LF"].values[0]
        features[f"{mod_name}_hrv_hf"] = hrv_freq["HRV_HF"].values[0]
        features[f"{mod_name}_hrv_uhf"] = hrv_freq["HRV_VHF"].values[0]
        features[f"{mod_name}_hrv_lf_hf_ratio"] = hrv_freq["HRV_LFHF"].values[0]
        features[f"{mod_name}_hrv_total_power"] = hrv_freq["HRV_TP"].values[0]
        features[f"{mod_name}_hrv_rel_ulf"] = (
            hrv_freq["HRV_VLF"].values[0] / hrv_freq["HRV_TP"].values[0]
        )
        features[f"{mod_name}_hrv_rel_lf"] = (
            hrv_freq["HRV_LF"].values[0] / hrv_freq["HRV_TP"].values[0]
        )
        features[f"{mod_name}_hrv_rel_hf"] = (
            hrv_freq["HRV_HF"].values[0] / hrv_freq["HRV_TP"].values[0]
        )
        features[f"{mod_name}_hrv_rel_uhf"] = (
            hrv_freq["HRV_VHF"].values[0] / hrv_freq["HRV_TP"].values[0]
        )
        features[f"{mod_name}_hrv_normalized_lf"] = hrv_freq["HRV_LFn"].values[0]
        features[f"{mod_name}_hrv_normalized_hf"] = hrv_freq["HRV_HFn"].values[0]

    elif "EDA" in mod_name:
        if "wrist" in mod_name:
            eda_cleaned = nk.eda_clean(mod_data, sampling_rate=sr)
        else:
            eda_cleaned = nk.eda_clean(mod_data, sampling_rate=sr, method="biosppy")
        if np.isnan(eda_cleaned).any():
            return None

        features[f"{mod_name}_mean"] = np.mean(eda_cleaned)
        features[f"{mod_name}_std"] = np.std(eda_cleaned)
        features[f"{mod_name}_min"] = np.min(eda_cleaned)
        features[f"{mod_name}_max"] = np.max(eda_cleaned)
        features[f"{mod_name}_slope"] = (eda_cleaned[-1] - eda_cleaned[0]) / len(
            eda_cleaned
        )
        features[f"{mod_name}_dynamic_range"] = np.max(eda_cleaned) - np.min(
            eda_cleaned
        )

        eda_phasic = nk.eda_phasic(eda_cleaned, sampling_rate=sr)
        scl = eda_phasic["EDA_Tonic"]
        scr = eda_phasic["EDA_Phasic"]
        features[f"{mod_name}_scl_mean"] = np.mean(scl)
        features[f"{mod_name}_scl_std"] = np.std(scl)
        features[f"{mod_name}_scr_mean"] = np.mean(scr)
        features[f"{mod_name}_scr_std"] = np.std(scr)

        time_vector = np.linspace(0, len(scl) / sr, len(scl))
        features[f"{mod_name}_scl_time_corr"] = np.corrcoef(scl, time_vector)[0, 1]
        _, scr_info = nk.eda_peaks(scr, sampling_rate=sr)
        peaks, props = find_peaks(scr, height=0.01, distance=int(1.0 * sr))
        scr_num_segments = len(peaks)
        scr_sum_magnitudes = props["peak_heights"].sum()
        features[f"{mod_name}_scr_num_segments"] = scr_num_segments
        features[f"{mod_name}_scr_sum_magnitudes"] = scr_sum_magnitudes
        features[f"{mod_name}_scr_total_duration"] = np.nansum(scr_info["SCR_RiseTime"])
        features[f"{mod_name}_scr_auc"] = np.trapezoid(np.maximum(scr, 0), dx=1 / sr)

    elif "EMG" in mod_name:
        emg_cleaned = nk.emg_clean(mod_data, sampling_rate=sr, method="biosppy")
        emg_window = emg_cleaned[len(emg_cleaned) - int(EMG_DURATION * sr) :]
        features[f"{mod_name}_mean"] = np.mean(emg_window)
        features[f"{mod_name}_std"] = np.std(emg_window)
        features[f"{mod_name}_dynamic_range"] = np.max(emg_window) - np.min(emg_window)
        features[f"{mod_name}_absolute_integral"] = np.sum(np.abs(emg_window)) / sr
        features[f"{mod_name}_median"] = np.median(emg_window)
        features[f"{mod_name}_10th_percentile"] = np.percentile(emg_window, 10)
        features[f"{mod_name}_90th_percentile"] = np.percentile(emg_window, 90)

        frequencies, power = welch(emg_window, fs=sr, nperseg=sr * 2)
        power_norm = power / np.sum(power)

        features[f"{mod_name}_mean_freq"] = np.sum(frequencies * power_norm)
        cumulative_power = np.cumsum(power_norm)
        features[f"{mod_name}_median_freq"] = frequencies[
            np.where(cumulative_power >= 0.5)[0][0]
        ]
        features[f"{mod_name}_peak_freq"] = frequencies[np.argmax(power)]

        band_edges = np.linspace(0, 350, 8)
        for i in range(7):
            mask = (frequencies >= band_edges[i]) & (frequencies < band_edges[i + 1])
            features[f"{mod_name}_band{i+1}_energy"] = np.sum(power[mask])

        emg_lowpassed = lowpass_filter(emg_cleaned, cutoff=50, fs=sr)
        peaks, _ = find_peaks(
            emg_lowpassed, height=np.std(emg_lowpassed)
        )  # or use a threshold
        peak_values = emg_lowpassed[peaks]
        features[f"{mod_name}_num_peaks"] = len(peaks)
        features[f"{mod_name}_peak_amplitude_mean"] = (
            np.mean(peak_values) if len(peak_values) > 0 else 0
        )
        features[f"{mod_name}_peak_amplitude_std"] = (
            np.std(peak_values) if len(peak_values) > 0 else 0
        )
        features[f"{mod_name}_peak_amplitude_sum"] = (
            np.sum(peak_values) if len(peak_values) > 0 else 0
        )
        features[f"{mod_name}_peak_amplitude_norm_sum"] = (
            np.sum(peak_values) / np.sum(np.abs(emg_lowpassed))
            if np.sum(np.abs(emg_lowpassed)) > 0
            else 0
        )

    elif "RESP" in mod_name:
        rsp_signals, _ = nk.rsp_process(
            mod_data, sampling_rate=sr, method="biosppy"
        )

        clean = rsp_signals["RSP_Clean"]
        phase = rsp_signals["RSP_Phase"]
        rate = rsp_signals["RSP_Rate"]
        amplitude = rsp_signals["RSP_Amplitude"]
        # rvt = rsp_signals["RSP_RVT"]
        peaks = np.where(rsp_signals["RSP_Peaks"] == 1)[0]
        troughs = np.where(rsp_signals["RSP_Troughs"] == 1)[0]

        inhale_durations = []
        for t in troughs:
            next_peaks = peaks[peaks > t]
            if len(next_peaks) == 0:
                continue
            inhale_durations.append((next_peaks[0] - t) / sr)
        inhale_durations = np.array(inhale_durations)

        exhale_durations = []
        for p in peaks:
            next_troughs = troughs[troughs > p]
            if len(next_troughs) == 0:
                continue
            exhale_durations.append((next_troughs[0] - p) / sr)
        exhale_durations = np.array(exhale_durations)

        features[f"{mod_name}_inhale_duration_mean"] = np.mean(inhale_durations)
        features[f"{mod_name}_inhale_duration_std"] = np.std(inhale_durations)
        features[f"{mod_name}_exhale_duration_mean"] = np.mean(exhale_durations)
        features[f"{mod_name}_exhale_duration_std"] = np.std(exhale_durations)
        features[f"{mod_name}_inhale_exhale_ratio"] = (
            np.mean(inhale_durations) / np.mean(exhale_durations)
            if np.mean(exhale_durations) > 0
            else np.nan
        )

        features[f"{mod_name}_stretch"] = np.max(clean) - np.min(clean)
        inhale_mask = phase == 1
        features[f"{mod_name}_inspiration_volume"] = np.trapezoid(
            amplitude[inhale_mask], dx=1 / sr
        )
        features[f"{mod_name}_respiration_rate"] = np.mean(rate)
        resp_durations = np.diff(troughs) / sr
        features[f"{mod_name}_respiration_duration"] = np.mean(resp_durations)

    elif "TEMP" in mod_name:
        temp_data = mod_data[:, 0]
        features[f"{mod_name}_mean"] = np.mean(temp_data)
        features[f"{mod_name}_std"] = np.std(temp_data)
        features[f"{mod_name}_min"] = np.min(temp_data)
        features[f"{mod_name}_max"] = np.max(temp_data)
        features[f"{mod_name}_slope"] = (temp_data[-1] - temp_data[0]) / len(temp_data)

    return features


def preprocess(user_path):
    user_id = os.path.basename(user_path)
    sync_data = pd.read_pickle(os.path.join(user_path, f"{user_id}.pkl"))

    signal = sync_data["signal"]
    label = sync_data["label"]

    data = []
    # windowing
    w_time = DURATION
    session_time = label.shape[0] / SAMPLING_RATES["label"]

    total_steps = int((session_time - DURATION) / SLIDING_WINDOW)
    pbar = tqdm(total=total_steps, desc=f"Processing {user_id}", leave=False)

    while w_time < session_time:
        skip_window = False
        label_end_idx = int(w_time * SAMPLING_RATES["label"])
        label_start_idx = int((w_time - DURATION) * SAMPLING_RATES["label"])
        w_label = label[label_start_idx:label_end_idx]

        # check if the label is unique
        if len(np.unique(w_label)) != 1:
            w_time += SLIDING_WINDOW
            continue
        w_label = w_label[0]

        # only consider labels that are in the LABEL_DICT
        if not w_label in LABEL_DICT.keys():
            w_time += SLIDING_WINDOW
            pbar.update(1)
            continue
        w_label = LABEL_DICT[w_label]

        w_features = {}
        # chest and wrist
        for pos in signal.keys():
            signal_pos = signal[pos]
            # different sensors
            for mod in signal_pos.keys():
                # extract exact slice of raw data
                mod_name = f"{pos.lower()}_{mod.upper()}"
                mod_data = signal_pos[mod]
                duration = DURATION
                if mod == "acc":
                    duration = ACC_DURATION
                mod_end_idx = int(w_time * SAMPLING_RATES[mod_name])
                mod_start_idx = int((w_time - duration) * SAMPLING_RATES[mod_name])
                mod_data = mod_data[mod_start_idx:mod_end_idx]

                mod_features = process_by_mod(mod_name, mod_data)
                if mod_features is None:
                    skip_window = True
                    break
                for k, v in mod_features.items():
                    w_features[k] = v      
            if skip_window:
                break
        if skip_window:
            w_time += SLIDING_WINDOW
            pbar.update(1)
            skip_window = False
            continue
        data.append(
            dict(
                user_id=user_id,
                label=w_label,
                features=w_features,
            )
        )

        w_time += SLIDING_WINDOW
        pbar.update(1)

    pbar.close()
    return data


def run(path=WESAD_PATH, out_dir=OUT_DIR, num_workers=32):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    #info_path = os.path.join(out_dir, "info.json")
    #store_info(info_path)
    #print(f"Saved info to {info_path}")
    users = glob(os.path.join(path, "S*"))
    all_data = []
    with Pool(processes=num_workers) as pool:
        for user_data in pool.imap_unordered(preprocess, users):
            all_data.extend(user_data)
    dataset = Dataset.from_list(all_data)
    hf_path = os.path.join(out_dir, "hf_dataset")
    dataset.save_to_disk(hf_path)
    print(f"Saved dataset to {hf_path}")


if __name__ == "__main__":
    Fire(run)
