
import mne
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import ttest_rel
import antropy as ant
import pandas as pd
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# Fixed seed for reproducibility across runs
np.random.seed(42)

# ─────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────

def bandpass(data, lowcut, highcut, fs, order=4):
    nyq = fs / 2
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def sci_score(signal, n_surrogates=40, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    w, L = 12, 12
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    envelope = np.convolve(envelope, np.ones(w)/w, mode='same')
    mu = np.mean(envelope)
    demeaned = envelope - mu
    var = np.sum(demeaned**2)
    if var == 0:
        return 0.5
    acf_vals = [np.sum(demeaned[:-k]*demeaned[k:])/var for k in range(1, L+1)]
    c_obs = np.mean(acf_vals)
    fft = np.fft.rfft(signal)
    mags = np.abs(fft)
    surrogate_scores = []
    for _ in range(n_surrogates):
        phases = rng.uniform(0, 2*np.pi, len(mags))
        phases[0] = 0
        fft_s = mags * np.exp(1j * phases)
        surrogate = np.fft.irfft(fft_s, n=len(signal))
        a_s = hilbert(surrogate)
        e_s = np.abs(a_s)
        e_s = np.convolve(e_s, np.ones(w)/w, mode='same')
        mu_s = np.mean(e_s)
        d_s = e_s - mu_s
        v_s = np.sum(d_s**2)
        if v_s == 0:
            surrogate_scores.append(0)
            continue
        acf_s = [np.sum(d_s[:-k]*d_s[k:])/v_s for k in range(1, L+1)]
        surrogate_scores.append(np.mean(acf_s))
    mu_sur = np.mean(surrogate_scores)
    sig_sur = np.std(surrogate_scores)
    if sig_sur == 0:
        return 0.5
    z = np.clip((c_obs - mu_sur) / sig_sur, -6, 6)
    return 1 / (1 + np.exp(-0.9 * z))

def band_power(signal, fs, lowcut, highcut):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    idx = np.logical_and(freqs >= lowcut, freqs <= highcut)
    if not np.any(idx):
        return np.nan
    return np.trapz(psd[idx], freqs[idx])

def compute_all_features(signal, fs, rng):
    features = {}
    filtered = bandpass(signal, 1, 40, fs)

    # SCI — broadband
    features['sci'] = sci_score(filtered, rng=rng)

    # SCI — delta only (spindle control: removes sigma/spindle content)
    try:
        delta_sig = bandpass(signal, 0.5, 4, fs)
        features['sci_delta'] = sci_score(delta_sig, rng=rng)
    except:
        features['sci_delta'] = np.nan

    # SCI — sigma removed (rough spindle suppression)
    # Note: reconstructing from two bands may introduce filter artifacts.
    # Treat as first-pass control only.
    try:
        low_sig  = bandpass(signal, 1, 12, fs)
        high_sig = bandpass(signal, 15, 40, fs)
        no_sigma = low_sig + high_sig
        features['sci_no_sigma'] = sci_score(no_sigma, rng=rng)
    except:
        features['sci_no_sigma'] = np.nan

    # Band powers
    features['delta_power'] = band_power(signal, fs, 0.5, 4)
    features['theta_power'] = band_power(signal, fs, 4, 8)
    features['alpha_power'] = band_power(signal, fs, 8, 12)
    features['sigma_power'] = band_power(signal, fs, 12, 15)  # spindle band
    features['beta_power']  = band_power(signal, fs, 15, 30)

    # Entropy / complexity baselines
    try:
        features['sample_entropy'] = ant.sample_entropy(filtered.astype(np.float64))
    except:
        features['sample_entropy'] = np.nan
    try:
        features['lempel_ziv'] = ant.lziv_complexity(
            filtered > np.median(filtered), normalize=True)
    except:
        features['lempel_ziv'] = np.nan
    try:
        features['spectral_entropy'] = ant.spectral_entropy(
            signal, sf=fs, method='welch', normalize=True)
    except:
        features['spectral_entropy'] = np.nan

    return features

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────

data_dir = '/Users/parkerlee/Desktop/SCI_Consciousness/mne_data/physionet-sleep-data/'
psg_files = sorted(glob.glob(os.path.join(data_dir, '*PSG.edf')))
print(f"Found {len(psg_files)} subject files\n")

# Known annotation labels to skip (unknown/artifact)
SKIP_LABELS = {'Sleep stage ?', 'Movement time'}

all_records = []
rng = np.random.default_rng(42)

for psg_file in psg_files:
    base = psg_file.replace('E0-PSG.edf', '')
    hypno_files = glob.glob(os.path.dirname(psg_file) + '/' + os.path.basename(psg_file)[:6] + '*Hypnogram.edf')
    if not hypno_files:
        print(f"No hypnogram for {psg_file}, skipping")
        continue
    hypno_file = hypno_files[0]
    subject = os.path.basename(psg_file)[:6]
    print(f"Processing {subject}...", end=' ', flush=True)

    try:
        raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
        annot = mne.read_annotations(hypno_file)
        raw.set_annotations(annot)
        fs = raw.info['sfreq']
        eeg_ch = [c for c in raw.ch_names if 'EEG' in c][0]
        data, _ = raw[eeg_ch]
        data = data[0]
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        epoch_len = int(30 * fs)
        n_epochs = 0

        for event in events:
            onset  = event[0]
            stage  = event[2]
            segment = data[onset:onset+epoch_len]
            if len(segment) < epoch_len:
                continue
            stage_name = [k for k,v in event_id.items() if v == stage][0]
            if stage_name in SKIP_LABELS:
                continue
            features = compute_all_features(segment, fs, rng)
            features['subject'] = subject
            features['stage']   = stage_name
            all_records.append(features)
            n_epochs += 1

        print(f"{n_epochs} epochs")

    except Exception as e:
        print(f"ERROR: {e}")
        continue

# ─────────────────────────────────────────
# BUILD DATAFRAME
# ─────────────────────────────────────────

df = pd.DataFrame(all_records)

# Save raw epoch data for paper trail
df.to_csv(os.path.expanduser("~/sci_sleepedf_epoch_features.csv"), index=False)
print(f"\nSaved epoch features: ~/sci_sleepedf_epoch_features.csv")

STAGE_ORDER = [
    'Sleep stage W',
    'Sleep stage R',
    'Sleep stage 1',
    'Sleep stage 2',
    'Sleep stage 3',
    'Sleep stage 4',
]

df = df[df['stage'].isin(STAGE_ORDER)].copy()
df['stage_code'] = df['stage'].str.replace('Sleep stage ', 'S')

# ─────────────────────────────────────────
# SECTION 1: SUBJECT COUNT PER STAGE
# ─────────────────────────────────────────

print("\n" + "="*70)
print("SUBJECTS CONTRIBUTING TO EACH STAGE")
print("="*70)
print(df.groupby('stage')['subject'].nunique().reindex(STAGE_ORDER).to_string())

# ─────────────────────────────────────────
# SECTION 2: POOLED EPOCH-LEVEL RESULTS
# ─────────────────────────────────────────

print("\n" + "="*70)
print("POOLED EPOCH-LEVEL RESULTS")
print("(Descriptive only — epochs not independent within subject)")
print("="*70)
print(f"{'Stage':<18} {'N':>5} {'SCI':>7} {'SCI_d':>7} "
      f"{'SCI_ns':>7} {'Delta':>9} {'Sigma':>9} "
      f"{'SampEn':>8} {'LZ':>7}")
print("-"*70)

for stage in STAGE_ORDER:
    sub = df[df['stage'] == stage]
    if len(sub) == 0:
        continue
    label = stage.replace('Sleep stage ', 'Stage ')
    print(f"{label:<18} {len(sub):>5} "
          f"{sub['sci'].mean():>7.3f} "
          f"{sub['sci_delta'].mean():>7.3f} "
          f"{sub['sci_no_sigma'].mean():>7.3f} "
          f"{sub['delta_power'].mean():>9.1f} "
          f"{sub['sigma_power'].mean():>9.1f} "
          f"{sub['sample_entropy'].mean():>8.3f} "
          f"{sub['lempel_ziv'].mean():>7.3f}")

# ─────────────────────────────────────────
# SECTION 3: PER-SUBJECT STAGE MEANS
# ─────────────────────────────────────────

print("\n" + "="*70)
print("PER-SUBJECT MEAN SCI BY STAGE")
print("(These are the units for statistical testing)")
print("="*70)

pivot = df.pivot_table(
    values='sci', index='subject',
    columns='stage', aggfunc='mean')
present_stages = [s for s in STAGE_ORDER if s in pivot.columns]
pivot = pivot[present_stages]
pivot.columns = [c.replace('Sleep stage ', 'S') for c in pivot.columns]
print(pivot.round(3).to_string())

pivot_full = df.pivot_table(
    values='sci', index='subject',
    columns='stage', aggfunc='mean')
pivot_full.to_csv(
    os.path.expanduser("~/sci_sleepedf_subject_stage_means.csv"))
print(f"\nSaved subject means: ~/sci_sleepedf_subject_stage_means.csv")

# ─────────────────────────────────────────
# SECTION 4: TRUE MIXED-EFFECTS MODEL
# ─────────────────────────────────────────

print("\n" + "="*70)
print("MIXED-EFFECTS MODEL: sci ~ C(stage_code) + (1|subject)")
print("(True random-intercept model via statsmodels mixedlm)")
print("="*70)

try:
    import statsmodels.formula.api as smf
    model = smf.mixedlm(
        "sci ~ C(stage_code, Treatment('SW'))",
        df, groups=df["subject"])
    res = model.fit(reml=True)
    print(res.summary())
except Exception as e:
    print(f"mixedlm error: {e}")
    print("Install with: pip3 install statsmodels")

# ─────────────────────────────────────────
# SECTION 5: SUBJECT-LEVEL CONTRASTS
# ─────────────────────────────────────────

print("\n" + "="*70)
print("SUBJECT-LEVEL CONTRASTS")
print("(Paired t-tests on per-subject means — correct unit of analysis)")
print("="*70)

contrasts = [
    ('Sleep stage W', 'Sleep stage 4'),
    ('Sleep stage R', 'Sleep stage 4'),
    ('Sleep stage W', 'Sleep stage R'),
    ('Sleep stage W', 'Sleep stage 3'),
    ('Sleep stage 3', 'Sleep stage 4'),
]

for s1, s2 in contrasts:
    col1 = pivot_full.get(s1)
    col2 = pivot_full.get(s2)
    if col1 is None or col2 is None:
        continue
    paired = pd.concat([col1, col2], axis=1).dropna()
    if len(paired) < 3:
        print(f"{s1.replace('Sleep stage ','')} vs "
              f"{s2.replace('Sleep stage ','')}: "
              f"insufficient subjects (n={len(paired)})")
        continue
    g1 = paired.iloc[:,0].values
    g2 = paired.iloc[:,1].values
    t, p = ttest_rel(g1, g2)
    diff = np.mean(g1 - g2)
    d = diff / np.std(g1 - g2)
    label1 = s1.replace('Sleep stage ', '')
    label2 = s2.replace('Sleep stage ', '')
    print(f"{label1} vs {label2}: "
          f"mean diff={diff:+.3f}, "
          f"t({len(paired)-1})={t:.2f}, "
          f"p={p:.4f}, "
          f"Cohen's d={d:.3f}, "
          f"n_subjects={len(paired)}")

# ─────────────────────────────────────────
# SECTION 6: SPINDLE CONTROL
# ─────────────────────────────────────────

print("\n" + "="*70)
print("SPINDLE CONTROL: SCI vs SCI-no-sigma vs SCI-delta")
print("Key question: does Stage 2 drop when sigma band removed?")
print("Note: sigma removal via band reconstruction — first-pass only,")
print("filter artifacts possible. Treat as exploratory.")
print("="*70)
print(f"{'Stage':<12} {'SCI':>7} {'SCI_ns':>8} {'SCI_d':>7} {'Sigma_pwr':>10}")
print("-"*50)

for stage in STAGE_ORDER:
    sub = df[df['stage'] == stage]
    if len(sub) == 0:
        continue
    label = stage.replace('Sleep stage ', 'Stage ')
    print(f"{label:<12} "
          f"{sub['sci'].mean():>7.3f} "
          f"{sub['sci_no_sigma'].mean():>8.3f} "
          f"{sub['sci_delta'].mean():>7.3f} "
          f"{sub['sigma_power'].mean():>10.1f}")

# ─────────────────────────────────────────
# SECTION 7: INCREMENTAL VALUE OF SCI
# ─────────────────────────────────────────

print("\n" + "="*70)
print("INCREMENTAL VALUE: Wake vs Stage 4 classification")
print("Leave-one-subject-out CV (correct — prevents subject leakage)")
print("="*70)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df_lr = df[df['stage'].isin(
        ['Sleep stage W', 'Sleep stage 4'])].copy()
    df_lr['conscious'] = (
        df_lr['stage'] == 'Sleep stage W').astype(int)
    df_lr = df_lr.dropna(
        subset=['delta_power','sci','sample_entropy','lempel_ziv'])

    subjects = df_lr['subject'].unique()

    feature_sets = {
        'delta_power only':        ['delta_power'],
        'delta + sigma':           ['delta_power', 'sigma_power'],
        'entropy only':            ['sample_entropy', 'lempel_ziv'],
        'delta + entropy':         ['delta_power', 'sample_entropy', 'lempel_ziv'],
        'SCI only':                ['sci'],
        'delta + entropy + SCI':   ['delta_power', 'sample_entropy', 'lempel_ziv', 'sci'],
    }

    from sklearn.metrics import roc_auc_score

    for label, feats in feature_sets.items():
        aucs = []
        for test_subj in subjects:
            train = df_lr[df_lr['subject'] != test_subj]
            test  = df_lr[df_lr['subject'] == test_subj]
            if test['conscious'].nunique() < 2:
                continue
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train[feats].values)
            X_test  = scaler.transform(test[feats].values)
            y_train = train['conscious'].values
            y_test  = test['conscious'].values
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train, y_train)
            if len(np.unique(y_test)) < 2:
                continue
            auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
            aucs.append(auc)
        if aucs:
            print(f"{label:<30} AUC = {np.mean(aucs):.3f} "
                  f"(+/- {np.std(aucs):.3f}, n={len(aucs)} subjects)")

except Exception as e:
    print(f"sklearn not available: {e}")
    print("Install: pip3 install scikit-learn")

print("\n" + "="*70)
print("DONE")
print("="*70)
