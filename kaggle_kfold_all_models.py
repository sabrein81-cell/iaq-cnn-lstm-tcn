# ===== Kaggle version: data prep (STEP 1 to 8b) + ablation =====
import os

SEED = 42

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import random
import numpy as np
import tensorflow as tf

random.seed(SEED)
np.random.seed(SEED)

tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

print("Deterministic seed =", SEED)

import glob, pandas as pd
CSV = glob.glob('/kaggle/input/**/Raw*measurements*.csv', recursive=True)[0]
print("Using:", CSV)
df = pd.read_csv(CSV)
print("Rows:", len(df), "Cols:", len(df.columns))

# ---- original cell 3 ----
# ==============================
# STEP 1: Create Timestamp
# ==============================

import pandas as pd
import numpy as np

df['Timestamp'] = pd.to_datetime(
    df['Date'].astype(str) + ' ' + df['Time'].astype(str),
    errors='coerce'
)

# Drop rows without a valid timestamp
df = df.dropna(subset=['Timestamp']).copy()

# Sort each device chronologically within each room
df = df.sort_values(
    ['roomcode', 'devcode', 'Timestamp']
).reset_index(drop=True)

print("✅ Timestamp created")
print("Rows:", len(df))
print("Rooms:", df['roomcode'].nunique())
print("Devices:", df['devcode'].nunique())

print("\nFirst rows:")
print(df[['roomcode', 'devcode', 'Timestamp']].head(10))

# ---- original cell 4 ----
# ==========================================
# STEP 2: Check sampling intervals per device
# ==========================================

temp = df[['roomcode', 'devcode', 'Timestamp']].copy()

temp['gap_min'] = (
    temp.groupby(['roomcode', 'devcode'])['Timestamp']
        .diff()
        .dt.total_seconds() / 60
)

# Exclude implausible values and long gaps from the distribution plot
gaps = temp.loc[
    (temp['gap_min'] > 0) &
    (temp['gap_min'] <= 60),
    'gap_min'
]

print("Median interval:", round(gaps.median(), 2), "minutes")

print("\nMost common intervals:")
print(
    gaps.round(2)
        .value_counts()
        .head(20)
)

# ---- original cell 5 ----
# ============================================================
# STEP 3 FINAL: CONTINUOUS 3-MIN RESAMPLING
# No device mixing + No interpolation across gaps > 15 minutes
# ============================================================

import pandas as pd
import numpy as np

SENSOR_COLS = ['CO2', 'PM2.5', 'PM10', 'T', 'RH']
MAX_GAP_MIN = 15
RESAMPLE_RULE = '3min'

# Work from the original data
work = df[
    ['roomcode', 'devcode', 'Timestamp'] + SENSOR_COLS
].copy()

work = work.dropna(subset=['roomcode', 'devcode', 'Timestamp'])
work = work.sort_values(
    ['roomcode', 'devcode', 'Timestamp']
).reset_index(drop=True)

# ------------------------------------------------------------
# 1. Identify interruptions within each device
# ------------------------------------------------------------
gap = (
    work.groupby(['roomcode', 'devcode'])['Timestamp']
        .diff()
        .dt.total_seconds()
        .div(60)
)

new_segment = (
    gap.isna() |
    (gap > MAX_GAP_MIN)
)

work['segment_id'] = (
    new_segment.groupby(
        [work['roomcode'], work['devcode']]
    ).cumsum()
)

print("Original rows :", len(work))
print("Segments      :", work[
    ['roomcode','devcode','segment_id']
].drop_duplicates().shape[0])

# ------------------------------------------------------------
# 2. Resampling within each continuous segment only
# ------------------------------------------------------------
parts = []

for (room, dev, seg), g in work.groupby(
    ['roomcode', 'devcode', 'segment_id'],
    sort=False
):

    if len(g) < 2:
        continue

    g = (
        g.set_index('Timestamp')[SENSOR_COLS]
        .resample(RESAMPLE_RULE)
        .mean()
    )

    # Short gaps only: at most 4 missing points
    MAX_MISSING_BINS = 4

    for col in SENSOR_COLS:

        s = g[col].copy()

        interp = s.interpolate(
            method='linear',
            limit_area='inside'
        )

        is_nan = s.isna()

        run_id = (is_nan != is_nan.shift()).cumsum()

        run_sizes = (
            is_nan.groupby(run_id)
            .transform('sum')
        )

        short_gap = is_nan & (run_sizes <= MAX_MISSING_BINS)

        s.loc[short_gap] = interp.loc[short_gap]

        g[col] = s

    # Drop rows that still have a missing sensor value
    g = g.dropna(subset=SENSOR_COLS)

    if len(g) == 0:
        continue

    # Restore Timestamp from the index
    g = g.rename_axis('Timestamp').reset_index()

    g['roomcode'] = room
    g['devcode'] = dev
    g['segment_id'] = seg

    parts.append(g)

# ------------------------------------------------------------
# 3. Final resampled dataset
# ------------------------------------------------------------
df_resampled = pd.concat(parts, ignore_index=True)

df_resampled = df_resampled[
    ['roomcode', 'devcode', 'segment_id',
     'Timestamp'] + SENSOR_COLS
]

df_resampled = df_resampled.sort_values(
    ['roomcode', 'devcode', 'segment_id', 'Timestamp']
).reset_index(drop=True)
# Re-segment after removing long missing runs
df_resampled = df_resampled.sort_values(
    ['roomcode', 'devcode', 'segment_id', 'Timestamp']
).reset_index(drop=True)

gap_after = (
    df_resampled
    .groupby(['roomcode', 'devcode'])['Timestamp']
    .diff()
    .dt.total_seconds()
    .div(60)
)

new_break = gap_after.isna() | (gap_after > 15)

df_resampled['segment_id'] = (
    new_break
    .groupby([
        df_resampled['roomcode'],
        df_resampled['devcode']
    ])
    .cumsum()
    .astype('int32')
)
# ------------------------------------------------------------
# 4. Verification
# ------------------------------------------------------------
check_gap = (
    df_resampled
    .groupby(['roomcode','devcode','segment_id'])['Timestamp']
    .diff()
    .dt.total_seconds()
    .div(60)
    .dropna()
)

print("\n✅ Corrected resampling completed")
print("Resampled rows :", len(df_resampled))
print("Rooms          :", df_resampled['roomcode'].nunique())
print("Devices        :", df_resampled['devcode'].nunique())

print("\nIntervals inside segments:")
print(check_gap.value_counts().head())

print("\nMissing values:")
print(df_resampled[SENSOR_COLS].isna().sum())

# ---- original cell 6 ----
# ============================================================
# STEP 4: 8-HOUR PREDICTED MOVING AVERAGES
# Within each room + device + continuous segment only
# ============================================================

import pandas as pd
import numpy as np

data = df_resampled.copy()

GROUP = ['roomcode', 'devcode', 'segment_id']

data = data.sort_values(
    GROUP + ['Timestamp']
).reset_index(drop=True)

parts_step4 = []

for _, g in data.groupby(GROUP, sort=False):

    g = g.sort_values('Timestamp').copy()
    g = g.set_index('Timestamp')

    for col in ['CO2', 'PM2.5', 'PM10']:

        # Past 4-hour average
        avg4 = g[col].rolling(
            '4h',
            min_periods=2
        ).mean()

        # Past 1-hour average
        avg1 = g[col].rolling(
            '1h',
            min_periods=2
        ).mean()

        # IAQI-S predicted 8-hour average
        g[f'{col}_8h'] = (4 * avg4 + 4 * avg1) / 8

    g = g.reset_index()
    parts_step4.append(g)

data = pd.concat(parts_step4, ignore_index=True)

# Drop rows without complete averages
before = len(data)

data = data.dropna(
    subset=['CO2_8h', 'PM2.5_8h', 'PM10_8h']
).reset_index(drop=True)

print("✅ STEP 4 completed")
print("Before :", before)
print("After  :", len(data))

print(
    data[
        ['roomcode', 'devcode', 'segment_id', 'Timestamp',
         'CO2_8h', 'PM2.5_8h', 'PM10_8h']
    ].head()
)

# ---- original cell 7 ----
# ============================================================
# STEP 5 FINAL: IAQI-S Ground Truth
# Continuous linear interpolation — no gaps, no rounding
# ============================================================

import numpy as np
import pandas as pd

N_BEFORE = len(data)
print("Rows before IAQI-S:", N_BEFORE)


# ------------------------------------------------------------
# 1. Continuous IAQI interpolation
# ------------------------------------------------------------

def iaqi_continuous(x, concentration_points):
    """
    Continuous linear interpolation from pollutant concentration
    to IAQI-S index.
    """
    x = np.asarray(x, dtype=np.float32)

    iaqi_points = np.array(
        [0, 50, 100, 200, 300, 400, 500],
        dtype=np.float32
    )

    concentration_points = np.asarray(
        concentration_points,
        dtype=np.float32
    )

    out = np.interp(
        x,
        concentration_points,
        iaqi_points
    )

    return out.astype(np.float32)


# ------------------------------------------------------------
# 2. IAQI-S concentration breakpoints
# ------------------------------------------------------------

PM10_POINTS = [
    0, 20, 75, 150, 220, 300, 380
]

PM25_POINTS = [
    0, 10, 35, 70, 100, 140, 180
]

CO2_POINTS = [
    300, 700, 1000, 2000, 3000, 4000, 5000
]


# ------------------------------------------------------------
# 3. Individual IAQI
# ------------------------------------------------------------

data["IAQI_PM10"] = iaqi_continuous(
    data["PM10_8h"].to_numpy(),
    PM10_POINTS
)

data["IAQI_PM25"] = iaqi_continuous(
    data["PM2.5_8h"].to_numpy(),
    PM25_POINTS
)

data["IAQI_CO2"] = iaqi_continuous(
    data["CO2_8h"].to_numpy(),
    CO2_POINTS
)


# ------------------------------------------------------------
# 4. Composite IAQI-S
# ------------------------------------------------------------

idx = data[
    ["IAQI_PM10", "IAQI_PM25", "IAQI_CO2"]
].to_numpy(dtype=np.float32)

# Maximum individual IAQI
base = np.max(idx, axis=1)

# Number of pollutants above Caution threshold
n_caution = np.sum(idx > 100, axis=1)

# EP1
ep1 = np.zeros(len(data), dtype=np.float32)
ep1[n_caution == 2] = 50
ep1[n_caution == 3] = 75

# EP2
ep2 = (
    np.sum((idx > 100) & (idx <= 200), axis=1) * 5
    + np.sum((idx > 200) & (idx <= 300), axis=1) * 10
    + np.sum((idx > 300) & (idx <= 400), axis=1) * 20
    + np.sum(idx > 400, axis=1) * 40
)

# Final composite
data["IAQI_Composite"] = np.clip(
    base + ep1 + ep2,
    0,
    500
)


# ------------------------------------------------------------
# 5. Six IAQI-S classes
# ------------------------------------------------------------

labels = np.array([
    "Good",
    "Moderate",
    "Caution",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous"
])

score = data["IAQI_Composite"].to_numpy()

# <=50, >50-100, >100-200, >200-300, >300-400, >400
class_id = np.digitize(
    score,
    bins=[50, 100, 200, 300, 400],
    right=True
)

data["IAQI_Class"] = labels[class_id]


# ------------------------------------------------------------
# 6. Final verification
# ------------------------------------------------------------

N_AFTER = len(data)

print("\n✅ IAQI-S Ground Truth created")
print("Rows before:", N_BEFORE)
print("Rows after :", N_AFTER)

assert N_AFTER == N_BEFORE, "ERROR: number of rows changed!"


print("\nClass distribution:")

counts = data["IAQI_Class"].value_counts()

for cls in labels:
    n = counts.get(cls, 0)
    p = n / N_AFTER * 100

    print(f"{cls:16s}: {n:,} ({p:.2f}%)")

# ---- original cell 8 ----
# ============================================================
# STEP 6 FINAL:
# Fixed + deterministic stratified ROOM-BASED split
# Uses only labels that will actually survive WINDOW=10
# No room overlap
# ============================================================

import numpy as np
import pandas as pd

SEED = 42
WINDOW = 10
N_VAL = 5
N_TEST = 5

LABELS = [
    'Good',
    'Moderate',
    'Caution',
    'Unhealthy',
    'Very Unhealthy',
    'Hazardous'
]

GROUP = ['roomcode', 'devcode', 'segment_id']

# Physical room = room code without the retrofit-period suffix (-a/-b/-c)
data['room_group'] = (
    data['roomcode'].astype(str).str.replace(r'[abc]$', '', regex=True)
)

# ------------------------------------------------------------
# 1. Find labels that are actually eligible after WINDOW=10
# ------------------------------------------------------------
eligible_idx = []

for _, g in data.groupby(GROUP, sort=False):

    g = g.sort_values('Timestamp')

    if len(g) < WINDOW:
        continue

    eligible_idx.extend(
        g.index[WINDOW - 1:].tolist()
    )

eligible = data.loc[
    eligible_idx,
    ['room_group', 'IAQI_Class']
].copy()

all_rooms = sorted(
    data['room_group'].dropna().unique().tolist(),
    key=lambda x: str(x)
)

room_counts = (
    eligible
    .groupby(['room_group', 'IAQI_Class'])
    .size()
    .unstack(fill_value=0)
    .reindex(index=all_rooms, fill_value=0)
    .reindex(columns=LABELS, fill_value=0)
)

# ------------------------------------------------------------
# 2. Scientific check:
# each class must exist in at least 3 different rooms
# if we want Train + Validation + Test all to contain it
# ------------------------------------------------------------
room_support = (room_counts > 0).sum(axis=0)

print("Rooms containing each eligible class:")
print(room_support)

if (room_support < 3).any():
    bad = room_support[room_support < 3]

    raise ValueError(
        "\nCannot create a leakage-free room-based "
        "Train/Validation/Test split containing every class.\n"
        f"Classes with fewer than 3 supporting rooms:\n{bad}"
    )

# ------------------------------------------------------------
# 3. Deterministic room selection
#    Not thousands of random trials
# ------------------------------------------------------------
rng = np.random.default_rng(SEED)

random_order = rng.permutation(all_rooms)

tie_rank = {
    room: i
    for i, room in enumerate(random_order)
}

global_counts = room_counts.sum(axis=0).to_numpy(dtype=float)
global_prop = global_counts / global_counts.sum()

def choose_rooms(candidates, n_pick):

    candidates = list(candidates)

    selected = []
    selected_counts = np.zeros(len(LABELS), dtype=float)

    support = (
        (room_counts.loc[candidates] > 0)
        .sum(axis=0)
        .to_numpy(dtype=float)
    )

    # Start with rarest classes
    class_order = np.argsort(support)

    for j in class_order:

        if selected_counts[j] > 0:
            continue

        available = [
            r for r in candidates
            if r not in selected
            and room_counts.loc[r, LABELS[j]] > 0
        ]

        if not available:
            raise ValueError(
                f"No available room contains class {LABELS[j]}"
            )

        def coverage_score(room):

            row = (
                room_counts
                .loc[room, LABELS]
                .to_numpy(dtype=float)
            )

            present = row > 0

            return np.sum(
                present / (support + 1e-9)
            )

        best_room = max(
            available,
            key=lambda r: (
                coverage_score(r),
                -tie_rank[r]
            )
        )

        selected.append(best_room)

        selected_counts += (
            room_counts
            .loc[best_room, LABELS]
            .to_numpy(dtype=float)
        )

        if len(selected) == n_pick:
            return np.array(selected, dtype=object)

    # Fill remaining rooms while keeping distribution close
    while len(selected) < n_pick:

        best_room = None
        best_score = np.inf

        for room in candidates:

            if room in selected:
                continue
            new_counts = (
                selected_counts
                + room_counts
                  .loc[room, LABELS]
                  .to_numpy(dtype=float)
            )

            if new_counts.sum() == 0:
                score = np.inf
            else:
                new_prop = new_counts / new_counts.sum()

                score = np.mean(
                    np.abs(new_prop - global_prop)
                    / np.sqrt(global_prop + 1e-8)
                )

            if (
                score < best_score
                or (
                    np.isclose(score, best_score)
                    and (
                        best_room is None
                        or tie_rank[room] < tie_rank[best_room]
                    )
                )
            ):
                best_score = score
                best_room = room

        selected.append(best_room)

        selected_counts += (
            room_counts
            .loc[best_room, LABELS]
            .to_numpy(dtype=float)
        )

    return np.array(selected, dtype=object)

# Test first
test_rooms = choose_rooms(
    all_rooms,
    N_TEST
)

remaining = [
    r for r in all_rooms
    if r not in set(test_rooms)
]

# Validation from remaining rooms
val_rooms = choose_rooms(
    remaining,
    N_VAL
)

train_rooms = np.array(
    [
        r for r in remaining
        if r not in set(val_rooms)
    ],
    dtype=object
)

# ------------------------------------------------------------
# 4. Create final data partitions
# ------------------------------------------------------------
train_part = data[
    data['room_group'].isin(train_rooms)
].copy()

val_part = data[
    data['room_group'].isin(val_rooms)
].copy()

test_part = data[
    data['room_group'].isin(test_rooms)
].copy()

# ------------------------------------------------------------
# 5. Final verification using WINDOW-eligible labels
# ------------------------------------------------------------
print("\n✅ FINAL FIXED PHYSICAL-ROOM SPLIT")

print("Train rooms      :", len(train_rooms))
print("Validation rooms :", len(val_rooms))
print("Test rooms       :", len(test_rooms))

print("\nRows:")
print("Train      :", len(train_part))
print("Validation :", len(val_part))
print("Test       :", len(test_part))

print("\nRoom overlap:")
print(
    "Train-Val  :",
    len(set(train_rooms) & set(val_rooms))
)
print(
    "Train-Test :",
    len(set(train_rooms) & set(test_rooms))
)
print(
    "Val-Test   :",
    len(set(val_rooms) & set(test_rooms))
)

for name, rooms_part in [
    ("TRAIN", train_rooms),
    ("VAL", val_rooms),
    ("TEST", test_rooms)
]:

    counts = (
        room_counts
        .loc[list(rooms_part), LABELS]
        .sum(axis=0)
    )

    print(f"\n{name} eligible labels after WINDOW={WINDOW}:")

    for cls in LABELS:
        print(f"{cls:16s}: {int(counts[cls]):,}")

    assert (counts > 0).all(), (
        f"{name} is missing one or more classes"
    )

print("\n✅ All 6 classes exist in Train, Validation and Test")

# =====================================================================
# EXTRA EXPERIMENT (M3): leave-rooms-out 3-fold cross-validation
# Models: all nine architectures. Seed 42. Same recipe as the main experiments.
# Needs GPU (T4). Results are appended to /kaggle/working/kfold_results.csv after every model.
# =====================================================================
import os, time, gc
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

features = ['CO2','PM2.5','PM10','T','RH']
LABELS = ['Good','Moderate','Caution','Unhealthy','Very Unhealthy','Hazardous']
CLS_ID = {n:i for i,n in enumerate(LABELS)}
K = 3
OUT = '/kaggle/working/kfold_results.csv'

# ---------- 1) assign the physical rooms to K folds (rare classes spread evenly) ----------
rc = room_counts[LABELS].copy()                      # eligible windows per room and class (from STEP 6)
rc = rc[rc.sum(axis=1) > 0]
tot = rc.sum()
order = rc.sort_values(['Hazardous','Very Unhealthy','Unhealthy','Caution'], ascending=False).index.tolist()
fold_of = {}; load = np.zeros((K, 6))
def cost(f, r):
    v = load[f] + rc.loc[r].values
    w = np.array([0.2, 0.2, 0.5, 1, 1, 1.5])         # weight rare classes more
    return float((w * v / tot.values).sum())
for r in order:
    f = int(np.argmin([cost(f, r) for f in range(K)]))
    fold_of[r] = f; load[f] += rc.loc[r].values
print("rooms per fold:", [sum(1 for v in fold_of.values() if v == f) for f in range(K)])
print("windows per class and fold (rows = fold):")
print(pd.DataFrame(load.astype(int), columns=LABELS))
assert (load > 0).all(), "a fold misses a class"

# ---------- 2) helpers ----------
def make_windows(df):
    Xs, ys = [], []
    for _, g in df.groupby(['roomcode','devcode','segment_id'], sort=False):
        g = g.sort_values('Timestamp')
        if len(g) < 10: continue
        X = g[features].to_numpy(dtype=np.float32)
        y = g['IAQI_Class'].map(CLS_ID).to_numpy()
        Xs.append(np.lib.stride_tricks.sliding_window_view(X, 10, axis=0).transpose(0, 2, 1))
        ys.append(y[9:])
    return np.concatenate(Xs), np.concatenate(ys)

def tcn_block(x, filters=64, kernel_size=3, dilation_rate=2, dropout=0.2):
    res = x
    x = L.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)(x)
    x = L.BatchNormalization()(x); x = L.Activation("relu")(x); x = L.Dropout(dropout)(x)
    x = L.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)(x)
    x = L.BatchNormalization()(x)
    if res.shape[-1] != filters: res = L.Conv1D(filters, 1, padding="same")(res)
    return L.Activation("relu")(L.Add()([x, res]))
def head(x):
    x = L.Dropout(0.2)(L.Dense(64, activation="relu")(x)); return L.Dense(6, activation="softmax")(x)
def build_hybrid(shape):
    i = L.Input(shape=shape)
    x = L.Conv1D(64, 3, padding="same")(i)
    x = L.Dropout(0.2)(L.Activation("relu")(L.BatchNormalization()(x)))
    x = L.Dropout(0.2)(L.LSTM(64, return_sequences=True)(x))
    x = tcn_block(x); x = L.GlobalAveragePooling1D()(x)
    return Model(i, head(x))
def build_tcn(shape):
    i = L.Input(shape=shape); x = i
    for d in (1, 2, 4): x = tcn_block(x, dilation_rate=d)
    return Model(i, head(L.GlobalAveragePooling1D()(x)))
def cbr(x):
    return L.Dropout(0.2)(L.Activation("relu")(L.BatchNormalization()(L.Conv1D(64, 3, padding="same")(x))))
def b_lstm(shape):
    i = L.Input(shape=shape); x = L.Dropout(0.2)(L.LSTM(64, return_sequences=True)(i))
    x = L.Dropout(0.2)(L.LSTM(32)(x)); return Model(i, head(x))
def b_cnn(shape):
    i = L.Input(shape=shape)
    x = L.Activation("relu")(L.BatchNormalization()(L.Conv1D(64, 3, padding="same")(i)))
    x = L.Activation("relu")(L.BatchNormalization()(L.Conv1D(64, 3, padding="same")(x)))
    x = L.Dropout(0.2)(L.MaxPooling1D(2)(x))
    x = L.Activation("relu")(L.BatchNormalization()(L.Conv1D(128, 3, padding="same")(x)))
    return Model(i, head(L.GlobalAveragePooling1D()(x)))
def b_cnn_lstm(shape):
    i = L.Input(shape=shape); x = L.Dropout(0.2)(L.LSTM(64)(cbr(i))); return Model(i, head(x))
def b_cnn_tcn(shape):
    i = L.Input(shape=shape); x = tcn_block(cbr(i)); return Model(i, head(L.GlobalAveragePooling1D()(x)))
def b_lstm_tcn(shape):
    i = L.Input(shape=shape); x = L.Dropout(0.2)(L.LSTM(64, return_sequences=True)(i))
    x = tcn_block(x); return Model(i, head(L.GlobalAveragePooling1D()(x)))
def b_parallel(shape):
    i = L.Input(shape=shape)
    c = L.GlobalAveragePooling1D()(cbr(i)); l = L.Dropout(0.2)(L.LSTM(64)(i)); t = L.GlobalAveragePooling1D()(tcn_block(i))
    return Model(i, head(L.Concatenate()([c, l, t])))
def b_mobile(shape):
    i = L.Input(shape=shape)
    x = L.BatchNormalization()(L.Conv1D(64, 3, padding="same", activation="relu")(i))
    for f in (64, 128, 128):
        r = x
        x = L.BatchNormalization()(L.SeparableConv1D(f, 3, padding="same", activation="relu")(x))
        x = L.BatchNormalization()(L.SeparableConv1D(f, 3, padding="same", activation="relu")(x))
        if r.shape[-1] != f: r = L.Conv1D(f, 1, padding="same")(r)
        x = L.Add()([x, r])
    return Model(i, head(L.GlobalAveragePooling1D()(x)))
MODELS = {"CNN_LSTM_TCN": build_hybrid, "TCN": build_tcn, "CNN_LSTM": b_cnn_lstm, "CNN_TCN": b_cnn_tcn,
          "LSTM_TCN": b_lstm_tcn, "Parallel": b_parallel, "LSTM": b_lstm, "CNN1D": b_cnn, "MobileNet1D": b_mobile}

# ---------- 3) the folds ----------
for k in range(K):
    test_rooms = [r for r, f in fold_of.items() if f == k]
    rest = [r for r, f in fold_of.items() if f != k]
    rng = np.random.default_rng(42 + k)
    val_rooms = None
    for _ in range(5000):                              # 5 validation rooms that contain all six classes
        cand = list(rng.choice(rest, size=5, replace=False))
        cv = rc.loc[cand].sum()
        ct = rc.loc[[r for r in rest if r not in cand]].sum()
        if (cv >= [1,1,1,1,1,20]).all() and (ct >= 1).all():
            val_rooms = cand; break
    assert val_rooms is not None, "no valid validation set"
    train_rooms_k = [r for r in rest if r not in val_rooms]
    print(f"\n######## FOLD {k+1}/{K}: train rooms {len(train_rooms_k)}, val {len(val_rooms)}, test {len(test_rooms)}")

    parts = {n: data[data['room_group'].isin(rr)].copy()
             for n, rr in [('train', train_rooms_k), ('val', val_rooms), ('test', test_rooms)]}
    sc = StandardScaler().fit(parts['train'][features])
    for n in parts: parts[n][features] = sc.transform(parts[n][features])
    X_tr, y_tr = make_windows(parts['train']); X_va, y_va = make_windows(parts['val']); X_te, y_te = make_windows(parts['test'])
    del parts; gc.collect()
    # oversample Very Unhealthy and Hazardous in TRAIN only, same recipe as the main experiments
    rg = np.random.default_rng(42); Xl, yl = [X_tr], [y_tr]
    for c in (4, 5):
        idx = np.where(y_tr == c)[0]; need = 5000 - len(idx)
        if len(idx) and need > 0:
            ch = rg.choice(idx, size=need, replace=True)
            Xl.append(X_tr[ch] + rg.normal(0, 0.01, (need,) + X_tr.shape[1:]).astype(np.float32)); yl.append(np.full(need, c))
    X_tr = np.concatenate(Xl); y_tr = np.concatenate(yl)
    print("windows  train/val/test:", len(y_tr), len(y_va), len(y_te))
    print("test class counts:", np.bincount(y_te, minlength=6))

    for name, build in MODELS.items():
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(42)
        m = build(X_tr.shape[1:])
        m.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cb = [EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True),
              ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6)]
        t0 = time.time()
        h = m.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=15, batch_size=256, shuffle=True, callbacks=cb, verbose=2)
        pt = m.predict(X_te, batch_size=1024, verbose=0).argmax(1)
        pv = m.predict(X_va, batch_size=1024, verbose=0).argmax(1)
        f1c = f1_score(y_te, pt, average=None, labels=range(6), zero_division=0)
        row = {"fold": k+1, "model": name, "train_rooms": len(train_rooms_k), "test_rooms": ",".join(map(str, test_rooms)),
               "val_rooms": ",".join(map(str, val_rooms)),
               "val_macro_f1": f1_score(y_va, pv, average="macro", labels=range(6), zero_division=0),
               "test_acc": accuracy_score(y_te, pt)*100,
               "test_macro_f1": f1_score(y_te, pt, average="macro", labels=range(6), zero_division=0),
               "epochs_run": len(h.history["loss"]), "minutes": (time.time()-t0)/60}
        for i, n in enumerate(LABELS): row["f1_"+n.replace(" ","")] = f1c[i]
        print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items() if a not in ("test_rooms","val_rooms")})
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    del X_tr, y_tr, X_va, y_va, X_te, y_te; gc.collect()

d = pd.read_csv(OUT)
print("\n===== mean ± SD over folds =====")
print(d.groupby("model")[["test_acc","test_macro_f1","f1_Hazardous"]].agg(["mean","std"]).round(4).to_string())
print("saved:", OUT)
