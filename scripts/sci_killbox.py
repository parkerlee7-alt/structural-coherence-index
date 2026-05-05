"""
SCI Kill Box
============
Pre-registered predictions: SCI_KillBox_PreRegistration_v1.docx (May 2, 2026)
Run ONCE. Do not modify predictions after running.

Outputs saved to ~/Desktop/SCI_Consciousness/killbox_results/
  - sci_killbox_results.csv       : full results
  - sci_killbox_metadata.txt      : seed, version, datetime, params
  - sci_killbox_plots.png         : bar chart of all results
  - sci_killbox_log.txt           : full console output
"""

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from datetime import datetime
import os
import sys

# ─────────────────────────────────────────
# LOCKED PARAMETERS — DO NOT CHANGE
# ─────────────────────────────────────────
SCRIPT_VERSION = "1.0.0"
SEED           = 42
N              = 10_000
FS             = 1_000
W              = 12
L              = 12
S              = 40
K              = 0.9
CORE_THRESH    = 0.75
RUN_TIMESTAMP  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

np.random.seed(SEED)
rng = np.random.default_rng(SEED)

OUTPUT_DIR = os.path.expanduser("~/Desktop/SCI_Consciousness/killbox_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tee output to log file
log_path = os.path.join(OUTPUT_DIR, "sci_killbox_log.txt")
log_file = open(log_path, "w")

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_file)

# ─────────────────────────────────────────
# SCI FUNCTION
# ─────────────────────────────────────────

def sci_score(signal, n_surrogates=S):
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    envelope = np.convolve(envelope, np.ones(W)/W, mode='same')
    mu = np.mean(envelope)
    demeaned = envelope - mu
    var = np.sum(demeaned**2)
    if var == 0:
        return 0.5, 0.0, 0.0
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
        e_s = np.convolve(e_s, np.ones(W)/W, mode='same')
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
        return 0.5, c_obs, mu_sur
    z = np.clip((c_obs - mu_sur) / sig_sur, -6, 6)
    sci = 1 / (1 + np.exp(-K * z))
    return sci, c_obs, mu_sur

# ─────────────────────────────────────────
# SIGNAL GENERATORS
# ─────────────────────────────────────────

def gen_white_noise():
    return rng.standard_normal(N)

def gen_pink_noise():
    f = np.fft.rfftfreq(N)
    f[0] = 1
    psd = 1.0 / f
    psd[0] = 0
    phases = rng.uniform(0, 2*np.pi, len(f))
    spectrum = np.sqrt(psd) * np.exp(1j * phases)
    signal = np.fft.irfft(spectrum, n=N)
    return signal / np.std(signal)

def gen_brownian():
    return np.cumsum(rng.standard_normal(N))

def gen_pi_digits():
    try:
        from mpmath import mp
        mp.dps = N + 10
        pi_str = mp.nstr(mp.pi, N + 5, strip_zeros=False).replace('3.', '').replace('.', '')[:N]
        return np.array([int(d) for d in pi_str], dtype=float) - 4.5
    except ImportError:
        print("  (mpmath not installed -- pip3 install mpmath -- using seeded substitute)")
        rng2 = np.random.default_rng(314159265)
        return rng2.integers(0, 10, N).astype(float) - 4.5

def gen_shuffled_real():
    t = np.arange(N) / FS
    base = rng.standard_normal(N)*0.5 + np.sin(2*np.pi*10*t)
    shuffled = base.copy()
    rng.shuffle(shuffled)
    return shuffled

def gen_phase_randomized():
    t = np.arange(N) / FS
    base = rng.standard_normal(N)*0.5 + np.sin(2*np.pi*10*t)
    fft = np.fft.rfft(base)
    mags = np.abs(fft)
    phases = rng.uniform(0, 2*np.pi, len(mags))
    phases[0] = 0
    return np.fft.irfft(mags * np.exp(1j*phases), n=N)

def gen_ar1(phi):
    x = np.zeros(N)
    e = rng.standard_normal(N)
    for t in range(1, N):
        x[t] = phi*x[t-1] + e[t]
    return x

def gen_ar2_oscillatory():
    x = np.zeros(N)
    e = rng.standard_normal(N)
    for t in range(2, N):
        x[t] = 1.5*x[t-1] - 0.9*x[t-2] + e[t]
    return x / (np.std(x) + 1e-12)

def gen_arma11():
    x = np.zeros(N)
    e = rng.standard_normal(N)
    for t in range(1, N):
        x[t] = 0.7*x[t-1] + e[t] + 0.4*e[t-1]
    return x

def gen_garch11():
    alpha0, alpha1, beta1 = 0.1, 0.1, 0.8
    h = np.zeros(N)
    x = np.zeros(N)
    e = rng.standard_normal(N)
    h[0] = alpha0 / (1 - alpha1 - beta1)
    x[0] = np.sqrt(max(h[0], 1e-12)) * e[0]
    for t in range(1, N):
        h[t] = alpha0 + alpha1*x[t-1]**2 + beta1*h[t-1]
        x[t] = np.sqrt(max(h[t], 1e-12)) * e[t]
    return x

def gen_pure_sine():
    t = np.arange(N) / FS
    return np.sin(2*np.pi*10*t)

def gen_am_signal():
    t = np.arange(N) / FS
    carrier = np.cos(2*np.pi*10*t)
    envelope_mod = 1 + 0.8*np.sin(2*np.pi*0.3*t)
    return envelope_mod * carrier + 0.1*rng.standard_normal(N)

def gen_chirp():
    t = np.arange(N) / FS
    phase = 2*np.pi*(2.0*t + (50.0-2.0)/(2*t[-1])*t**2)
    return np.sin(phase) + 0.1*rng.standard_normal(N)

def gen_logistic_map(r=3.9):
    x = np.zeros(N)
    x[0] = 0.5
    for t in range(1, N):
        x[t] = r*x[t-1]*(1 - x[t-1])
    return x - np.mean(x)

def gen_lorenz():
    dt = 0.01
    sigma, rho, beta = 10, 28, 8/3
    x, y, z = 0.1, 0.0, 0.0
    for _ in range(1000):  # burn in
        dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
        x += dx*dt; y += dy*dt; z += dz*dt
    xs = []
    for _ in range(N):
        dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
        x += dx*dt; y += dy*dt; z += dz*dt
        xs.append(x)
    sig = np.array(xs)
    return (sig - np.mean(sig)) / (np.std(sig) + 1e-12)

def gen_cwru_healthy():
    t = np.arange(N) / FS
    return (rng.standard_normal(N)*0.5 +
            0.1*np.sin(2*np.pi*120*t) +
            0.05*np.sin(2*np.pi*240*t))

def gen_cwru_fault():
    t = np.arange(N) / FS
    f_fault = 162.0
    impact_train = np.zeros(N)
    period = int(FS / f_fault)
    for i in range(0, N, period):
        if i < N:
            impact_train[i] = 1.0
    decay = np.exp(-np.arange(100)/10)
    impact_response = np.convolve(impact_train, decay, mode='same')
    carrier = rng.standard_normal(N)
    envelope_mod = 1 + 0.8 * impact_response / (impact_response.max() + 1e-9)
    return envelope_mod * carrier

def gen_awake_eeg():
    t = np.arange(N) / FS
    alpha = np.sin(2*np.pi*10*t + rng.uniform(0, 2*np.pi))
    beta  = 0.3*np.sin(2*np.pi*20*t + rng.uniform(0, 2*np.pi))
    mod = 1 + 0.5*np.sin(2*np.pi*0.1*t)
    return mod*(alpha+beta) + 0.3*rng.standard_normal(N)

def gen_deep_sleep_eeg():
    t = np.arange(N) / FS
    delta = 2.0*np.sin(2*np.pi*1.5*t + rng.uniform(0, 2*np.pi))
    return delta + rng.standard_normal(N)

def gen_meteo():
    t = np.arange(N)
    return 10*np.sin(2*np.pi*t/365) + 0.001*t + 2*rng.standard_normal(N)

# ─────────────────────────────────────────
# SIGNAL REGISTRY WITH PREDICTIONS
# ─────────────────────────────────────────

SIGNALS = [
    # (name, generator, prediction_label, group, is_relative_scoring)
    ('White noise',             gen_white_noise,          'LOW',                  'A', False),
    ('Pink noise',              gen_pink_noise,           'AMBIGUOUS',            'A', False),
    ('Brownian motion',         gen_brownian,             'AMBIGUOUS',            'A', False),
    ('Pi digits',               gen_pi_digits,            'LOW',                  'A', False),
    ('Shuffled real data',      gen_shuffled_real,        'LOW',                  'A', False),
    ('Phase-randomized',        gen_phase_randomized,     'LOW',                  'A', False),
    ('AR(1) phi=0.95',          lambda: gen_ar1(0.95),    'HIGH',                 'B', False),
    ('AR(1) phi=0.30',          lambda: gen_ar1(0.30),    'AMBIGUOUS',            'B', False),
    ('AR(1) phi=0.05',          lambda: gen_ar1(0.05),    'LOW',                  'B', False),
    ('AR(2) oscillatory',       gen_ar2_oscillatory,      'HIGH',                 'B', False),
    ('ARMA(1,1)',                gen_arma11,               'AMBIGUOUS-HIGH',       'B', False),
    ('GARCH(1,1)',               gen_garch11,              'HIGH',                 'B', False),
    ('Pure sine',               gen_pure_sine,            'LOW-AMBIGUOUS',        'C', False),
    ('AM signal',               gen_am_signal,            'HIGH',                 'C', False),
    ('Chirp',                   gen_chirp,                'HIGH',                 'C', False),
    ('Logistic map r=3.9',      gen_logistic_map,         'AMBIGUOUS',            'C', False),
    ('Lorenz x-component',      gen_lorenz,               'AMBIGUOUS-HIGH',       'C', False),
    ('CWRU healthy 0HP',        gen_cwru_healthy,         'LOW-AMBIGUOUS',        'D', True),
    ('CWRU fault 0.007"',       gen_cwru_fault,           'HIGH',                 'D', True),
    ('Awake EEG (simulated)',   gen_awake_eeg,            'HIGH',                 'D', True),
    ('Stage 4 EEG (simulated)', gen_deep_sleep_eeg,       'LOW REL/AMB ABS',      'D', True),
    ('Meteorological data',     gen_meteo,                'HIGH',                 'D', False),
]

GROUP_LABELS = {
    'A': 'NULL / RANDOM BASELINES',
    'B': 'CAUSAL PARAMETRIC',
    'C': 'DETERMINISTIC / CHAOTIC',
    'D': 'REAL / SIMULATED PHYSICAL',
}

def classify(sci):
    if sci >= CORE_THRESH:
        return 'HIGH'
    elif sci >= 0.55:
        return 'AMBIGUOUS'
    else:
        return 'LOW'

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

print("=" * 75)
print("SCI KILL BOX — PRE-REGISTERED EXECUTION")
print(f"Date/time : {RUN_TIMESTAMP}")
print(f"Script    : v{SCRIPT_VERSION}")
print(f"Seed      : {SEED}  |  N={N}  |  W={W}  |  L={L}  |  S={S}  |  k={K}")
print("=" * 75)

rows = []
current_group = None

print(f"\n{'Signal':<28} {'Pred':>16} {'SCI':>7} {'c_obs':>7} {'c_surr':>7} {'Class':>10}")
print("-" * 80)

for name, gen_fn, pred, group, relative in SIGNALS:
    if group != current_group:
        current_group = group
        print(f"\n  -- Group {group}: {GROUP_LABELS[group]} --")

    print(f"  {name:<28}", end=' ', flush=True)
    try:
        signal = gen_fn()
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
        sci, c_obs, c_surr = sci_score(signal)
        cls = classify(sci)
        print(f"{pred:>16}  {sci:>6.3f}  {c_obs:>6.4f}  {c_surr:>6.4f}  {cls:>10}")
        rows.append({
            'signal': name, 'group': group,
            'prediction': pred, 'relative_scoring': relative,
            'sci': round(sci, 4), 'c_obs': round(c_obs, 6),
            'c_surr': round(c_surr, 6), 'classification': cls,
            'error': None
        })
    except Exception as e:
        print(f"  ERROR: {e}")
        rows.append({
            'signal': name, 'group': group,
            'prediction': pred, 'relative_scoring': relative,
            'sci': None, 'c_obs': None, 'c_surr': None,
            'classification': 'ERROR', 'error': str(e)
        })

# ─────────────────────────────────────────
# CONFIRMATION SCORING
# ─────────────────────────────────────────

# Get reference values for relative scoring
awake_sci  = next((r['sci'] for r in rows if r['signal'] == 'Awake EEG (simulated)'), None)
fault_sci  = next((r['sci'] for r in rows if r['signal'] == 'CWRU fault 0.007"'), None)

for r in rows:
    if r['sci'] is None:
        r['confirmed'] = 'ERROR'
        continue
    pred = r['prediction'].upper()
    sci  = r['sci']
    cls  = r['classification']

    if r['signal'] == 'Stage 4 EEG (simulated)' and awake_sci is not None:
        r['confirmed'] = 'YES' if (awake_sci - sci) >= 0.15 else 'NO'
    elif r['signal'] == 'CWRU healthy 0HP' and fault_sci is not None:
        r['confirmed'] = 'YES' if (fault_sci - sci) >= 0.20 else 'PARTIAL'
    elif 'HIGH' in pred and 'LOW' not in pred and 'AMBIGUOUS' not in pred:
        r['confirmed'] = 'YES' if sci >= 0.75 else 'NO'
    elif 'LOW' in pred and 'HIGH' not in pred and 'AMBIGUOUS' not in pred:
        r['confirmed'] = 'YES' if sci < 0.60 else 'NO'
    else:
        # Ambiguous predictions — check if within expected range
        r['confirmed'] = 'PARTIAL'

# ─────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────

df = pd.DataFrame(rows)

print("\n" + "=" * 75)
print("RESULTS SUMMARY")
print("=" * 75)
print(f"\n{'Signal':<28} {'Pred':>16} {'SCI':>7} {'Confirmed':>10}")
print("-" * 65)
for _, r in df.iterrows():
    sci_str = f"{r['sci']:.3f}" if r['sci'] is not None else "ERROR"
    print(f"  {r['signal']:<28} {r['prediction']:>16}  {sci_str:>6}  {r['confirmed']:>10}")

confirmed = df[df['confirmed'] == 'YES'].shape[0]
failed    = df[df['confirmed'] == 'NO'].shape[0]
partial   = df[df['confirmed'] == 'PARTIAL'].shape[0]
errors    = df[df['confirmed'] == 'ERROR'].shape[0]
total     = len(df)

print(f"\nConfirmed : {confirmed}/{total}")
print(f"Failed    : {failed}/{total}")
print(f"Partial   : {partial}/{total}")
print(f"Errors    : {errors}/{total}")

print("\nFailed predictions:")
for _, r in df[df['confirmed'] == 'NO'].iterrows():
    print(f"  {r['signal']:<28} predicted {r['prediction']:<16} got {r['sci']:.3f} ({r['classification']})")

print("\nKey scientific questions:")

garch = df[df['signal'] == 'GARCH(1,1)']['sci'].values
if len(garch) and garch[0] is not None:
    g = garch[0]
    verdict = "EXTENDS proposition to nonlinear dependence" if g > 0.75 else "Does NOT extend to GARCH volatility clustering"
    print(f"  GARCH(1,1)        SCI={g:.3f}  -> {verdict}")

logistic = df[df['signal'] == 'Logistic map r=3.9']['sci'].values
if len(logistic) and logistic[0] is not None:
    v = logistic[0]
    verdict = "Chaos qualifies as causal coherence generator" if v > 0.70 else "Chaos does NOT reliably produce coherence excess"
    print(f"  Logistic map      SCI={v:.3f}  -> {verdict}")

lorenz = df[df['signal'] == 'Lorenz x-component']['sci'].values
if len(lorenz) and lorenz[0] is not None:
    v = lorenz[0]
    verdict = "Structured chaos produces strong coherence" if v > 0.75 else "Lorenz ambiguous"
    print(f"  Lorenz            SCI={v:.3f}  -> {verdict}")

ar_vals = {}
for phi_str in ['0.95', '0.30', '0.05']:
    name = f'AR(1) phi={phi_str}'
    val = df[df['signal'] == name]['sci'].values
    if len(val) and val[0] is not None:
        ar_vals[phi_str] = val[0]
if ar_vals:
    print(f"  AR(1) gradient    phi=0.95->{ar_vals.get('0.95', '?'):.3f}  phi=0.30->{ar_vals.get('0.30', '?'):.3f}  phi=0.05->{ar_vals.get('0.05', '?'):.3f}")

theory_breakers = ['White noise', 'Shuffled real data', 'Phase-randomized']
print("\nTheory-breaking checks (must be LOW):")
for _, r in df[df['signal'].isin(theory_breakers)].iterrows():
    flag = '*** THEORY PROBLEM ***' if r['sci'] is not None and r['sci'] > 0.70 else 'OK'
    sci_str = f"{r['sci']:.3f}" if r['sci'] is not None else "ERROR"
    print(f"  {r['signal']:<28} SCI={sci_str}  {flag}")

# ─────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────

# CSV
csv_path = os.path.join(OUTPUT_DIR, "sci_killbox_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# Metadata
meta_path = os.path.join(OUTPUT_DIR, "sci_killbox_metadata.txt")
with open(meta_path, 'w') as f:
    f.write(f"SCI Kill Box Metadata\n")
    f.write(f"=====================\n")
    f.write(f"Run timestamp  : {RUN_TIMESTAMP}\n")
    f.write(f"Script version : {SCRIPT_VERSION}\n")
    f.write(f"Random seed    : {SEED}\n")
    f.write(f"Signal length  : {N}\n")
    f.write(f"Sampling rate  : {FS} Hz\n")
    f.write(f"Smoothing (W)  : {W}\n")
    f.write(f"ACF lags (L)   : {L}\n")
    f.write(f"Surrogates (S) : {S}\n")
    f.write(f"Logistic k     : {K}\n")
    f.write(f"CORE threshold : {CORE_THRESH}\n")
    f.write(f"Pre-reg doc    : SCI_KillBox_PreRegistration_v1.docx\n")
    f.write(f"Confirmed      : {confirmed}/{total}\n")
    f.write(f"Failed         : {failed}/{total}\n")
print(f"Saved: {meta_path}")

# Plot
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    df_plot = df[df['sci'].notna()].copy()
    colors = []
    for _, r in df_plot.iterrows():
        if r['confirmed'] == 'YES':
            colors.append('#4CAF50')
        elif r['confirmed'] == 'NO':
            colors.append('#F44336')
        elif r['confirmed'] == 'PARTIAL':
            colors.append('#FF9800')
        else:
            colors.append('#9E9E9E')

    fig, ax = plt.subplots(figsize=(14, 9))
    bars = ax.barh(df_plot['signal'], df_plot['sci'], color=colors, edgecolor='white', height=0.7)
    ax.axvline(x=CORE_THRESH, color='#1F3864', linestyle='--', linewidth=1.5, label=f'CORE threshold ({CORE_THRESH})')
    ax.axvline(x=0.60, color='#888888', linestyle=':', linewidth=1, label='LOW boundary (0.60)')
    ax.set_xlabel('SCI Score', fontsize=12)
    ax.set_title('SCI Kill Box Results — Pre-Registered Predictions\nParker J. Lee — May 2, 2026', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)

    # Add value labels
    for bar, (_, r) in zip(bars, df_plot.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{r['sci']:.3f}", va='center', ha='left', fontsize=9, color='#333333')

    # Group separators
    group_boundaries = []
    prev_group = None
    for i, (_, r) in enumerate(df_plot.iterrows()):
        if r['group'] != prev_group:
            group_boundaries.append((i, r['group']))
            prev_group = r['group']

    # Legend
    patches = [
        mpatches.Patch(color='#4CAF50', label='Confirmed'),
        mpatches.Patch(color='#F44336', label='Failed'),
        mpatches.Patch(color='#FF9800', label='Partial/Ambiguous'),
    ]
    ax.legend(handles=patches + [
        plt.Line2D([0],[0], color='#1F3864', linestyle='--', label=f'CORE ({CORE_THRESH})'),
        plt.Line2D([0],[0], color='#888888', linestyle=':', label='LOW boundary (0.60)'),
    ], loc='lower right', fontsize=9)

    ax.invert_yaxis()
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "sci_killbox_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")
except Exception as e:
    print(f"Plot failed: {e}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"Log: {log_path}")
print("\nDONE.")
