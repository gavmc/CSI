import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("csi_data.csv", on_bad_lines='skip')
print(f"Loaded {len(df)} rows")

empty_idx = df[df['label'] == 'empty'].index
if len(empty_idx) > 500:
    df = df.drop(empty_idx[:500]).reset_index(drop=True)
    print(f"Dropped 500 empty-phase warmup rows. Now {len(df)} rows.")


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


print("Parsing CSI...")
parsed = [parse_iq(r) for _, r in df.iterrows()]
n_sub = min(len(p) for p in parsed)
csi = np.stack([p[:n_sub] for p in parsed])
amp = np.abs(csi)
labels = df['label'].values

bad = (amp > 200).any(axis=1)
if bad.any():
    amp = amp[~bad]
    labels = labels[~bad]

active = amp.mean(axis=0) > 1.0
amp_active = amp[:, active]
print(f"Using {active.sum()} active subcarriers (out of {n_sub})")

WINDOW = 50
STRIDE = 25


def make_windows(amp_data, labels_data, window, stride):
    X, y = [], []
    for start in range(0, len(amp_data) - window + 1, stride):
        end = start + window
        win_labels = labels_data[start:end]
        if len(set(win_labels)) != 1:
            continue
        win = amp_data[start:end]
        feat = np.concatenate([
            win.mean(axis=0),
            win.std(axis=0),
            win.max(axis=0) - win.min(axis=0),
        ])
        X.append(feat)
        y.append(win_labels[0])
    return np.array(X), np.array(y)


X, y = make_windows(amp_active, labels, WINDOW, STRIDE)
print(f"\nFeature matrix: {X.shape}")
print("Windows per class:")
for cls, n in zip(*np.unique(y, return_counts=True)):
    print(f"  {cls:10s}: {n}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} windows  Test: {len(X_test)} windows")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("\nTraining Random Forest...")
clf = RandomForestClassifier(
    n_estimators=200, max_depth=20,
    class_weight='balanced',
    n_jobs=-1, random_state=42
)
clf.fit(X_train_s, y_train)

print(f"\nTrain accuracy: {clf.score(X_train_s, y_train):.3f}")
print(f"Test accuracy:  {clf.score(X_test_s, y_test):.3f}")

y_pred = clf.predict(X_test_s)
print("\n4-CLASS REPORT (test set):")
print(classification_report(y_test, y_pred, digits=3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)\
    .plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title('4-class confusion matrix (test)')

n_active = active.sum()
imps = clf.feature_importances_
axes[1].plot(imps[:n_active], label='mean')
axes[1].plot(imps[n_active:2*n_active], label='std')
axes[1].plot(imps[2*n_active:], label='peak-to-peak')
axes[1].set_xlabel('Active subcarrier index')
axes[1].set_ylabel('Importance')
axes[1].set_title('Feature importance')
axes[1].legend()

plt.tight_layout()
plt.savefig('classifier_results.png', dpi=120)
plt.show()
print("\nSaved classifier_results.png")

print("\nBinary (motion vs no-motion):")
y_bin = np.where(np.isin(y, ['walking', 'waving']), 'motion', 'no_motion')
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    X, y_bin, test_size=0.25, random_state=42, stratify=y_bin
)
sc_b = StandardScaler().fit(Xb_tr)
clf_b = RandomForestClassifier(
    n_estimators=200, max_depth=20, class_weight='balanced',
    n_jobs=-1, random_state=42
)
clf_b.fit(sc_b.transform(Xb_tr), yb_tr)
print(f"Test accuracy: {clf_b.score(sc_b.transform(Xb_te), yb_te):.3f}")
print(classification_report(yb_te, clf_b.predict(sc_b.transform(Xb_te)), digits=3))

joblib.dump({
    'classifier': clf,
    'scaler': scaler,
    'active_mask': active,
    'window': WINDOW,
    'classes': clf.classes_.tolist(),
}, 'csi_model.joblib')
print("\nSaved csi_model.joblib")
