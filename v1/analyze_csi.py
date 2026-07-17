import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("csi_data.csv", on_bad_lines='warn')
print(f"Loaded {len(df)} rows")
print("\nRows per phase:")
print(df['label'].value_counts().to_string())


def parse_iq(row):
    vals = np.array(json.loads(row['data']), dtype=np.int8)
    declared_len = int(row['len'])
    if len(vals) >= declared_len:
        vals = vals[:declared_len]
    else:
        vals = np.concatenate([vals, np.zeros(declared_len - len(vals), dtype=np.int8)])
    vals = vals[4:]
    if len(vals) % 2 == 1:
        vals = vals[:-1]
    iq = vals.astype(np.float32)
    return iq[0::2] + 1j * iq[1::2]


parsed = [parse_iq(r) for _, r in df.iterrows()]
n_subcarriers = min(len(p) for p in parsed)
print(f"\nSubcarriers per packet: {n_subcarriers}")
csi = np.stack([p[:n_subcarriers] for p in parsed])
amp = np.abs(csi)
labels = df['label'].values

print(f"Amplitude range: [{amp.min():.1f}, {amp.max():.1f}]")
if amp.max() > 200:
    print(f"WARNING: {(amp > 200).sum()} samples have impossible amplitudes — filtering")
    bad = (amp > 200).any(axis=1)
    csi = csi[~bad]
    amp = amp[~bad]
    labels = labels[~bad]
    df = df[~bad].reset_index(drop=True)
    print(f"After filter: {len(df)} rows")

mean_amp = amp.mean(axis=0)
active = mean_amp > 1.0
print(f"Active subcarriers: {active.sum()} / {n_subcarriers}")

phase_colors = {'empty': 'gray', 'walking': 'tab:red',
                'waving': 'tab:orange', 'breathing': 'tab:green'}
phases_in_order = list(dict.fromkeys(labels))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

ax = axes[0, 0]
im = ax.imshow(amp.T, aspect='auto', origin='lower', cmap='viridis')
for i in range(1, len(df)):
    if labels[i] != labels[i - 1]:
        ax.axvline(i, color='white', linestyle='--', alpha=0.8)
ax.set_xlabel("Packet index")
ax.set_ylabel("Subcarrier")
ax.set_title("CSI amplitude heatmap")
plt.colorbar(im, ax=ax, label='|H|')

ax = axes[0, 1]
for phase in phases_in_order:
    mask = labels == phase
    if mask.sum() > 0:
        ax.plot(amp[mask].mean(axis=0), label=f"{phase} (n={mask.sum()})",
                color=phase_colors.get(phase, 'k'))
ax.set_xlabel("Subcarrier")
ax.set_ylabel("Mean |H|")
ax.set_title("Mean amplitude profile by phase")
ax.legend()

ax = axes[1, 0]
for phase in phases_in_order:
    mask = labels == phase
    if mask.sum() > 1:
        ax.plot(amp[mask].std(axis=0), label=phase,
                color=phase_colors.get(phase, 'k'))
ax.set_xlabel("Subcarrier")
ax.set_ylabel("Std of |H|")
ax.set_title("Motion signature: variability per subcarrier")
ax.legend()

ax = axes[1, 1]
amp_active = amp[:, active]
inst_var = amp_active.std(axis=1)
window = max(5, len(df) // 100)
smoothed = pd.Series(inst_var).rolling(window, center=True).mean()
ax.plot(smoothed, color='black', linewidth=1.0)
for phase in phases_in_order:
    mask = labels == phase
    if mask.sum() > 0:
        idx = np.where(mask)[0]
        ax.axvspan(idx[0], idx[-1], alpha=0.2,
                   color=phase_colors.get(phase, 'k'), label=phase)
ax.set_xlabel("Packet index")
ax.set_ylabel("Cross-subcarrier std")
ax.set_title(f"Motion intensity over time (window={window})")
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('csi_analysis.png', dpi=120)
plt.show()
print("\nSaved csi_analysis.png")

print("\nPer-phase summary (active subcarriers only):")
print(f"{'phase':12s}{'n':>6s}{'mean':>10s}{'within_std':>14s}{'temporal_var':>16s}")
for phase in phases_in_order:
    mask = labels == phase
    if mask.sum() > 0:
        a = amp[mask][:, active]
        print(f"{phase:12s}{mask.sum():>6d}{a.mean():>10.2f}"
              f"{a.std():>14.2f}{a.std(axis=0).mean():>16.2f}")
