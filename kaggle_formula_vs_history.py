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
# EXTRA ANALYSIS (M2): IAQI-S formula applied to different history lengths
# Runs on test_part BEFORE scaling (real units). No training. No GPU.
# =====================================================================
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PM10_P=[0,20,75,150,220,300,380]; PM25_P=[0,10,35,70,100,140,180]
CO2_P=[300,700,1000,2000,3000,4000,5000]; I_P=[0,50,100,200,300,400,500]
def formula_class(co2,pm25,pm10):
    idx=np.stack([np.interp(pm10,PM10_P,I_P),np.interp(pm25,PM25_P,I_P),np.interp(co2,CO2_P,I_P)],axis=1)
    base=idx.max(axis=1); n_c=(idx>100).sum(axis=1)
    ep1=np.zeros(len(idx)); ep1[n_c==2]=50; ep1[n_c==3]=75
    ep2=(((idx>100)&(idx<=200)).sum(1)*5+((idx>200)&(idx<=300)).sum(1)*10
         +((idx>300)&(idx<=400)).sum(1)*20+(idx>400).sum(1)*40)
    return np.digitize(np.clip(base+ep1+ep2,0,500),bins=[50,100,200,300,400],right=True)

CLS={'Good':0,'Moderate':1,'Caution':2,'Unhealthy':3,'Very Unhealthy':4,'Hazardous':5}
H_LIST=[5,10,20,40,60,80]      # readings (3 min each): 15 min, 30 min, 1 h, 2 h, 3 h, 4 h
HMAX=max(H_LIST)

rows_all=[]   # (position in segment, label, per-H predictions)
means={h:[] for h in H_LIST}; short={h:[] for h in H_LIST}; labels=[]; pos_in_seg=[]
for _,g in test_part.groupby(['roomcode','devcode','segment_id'],sort=False):
    g=g.sort_values('Timestamp')
    if len(g)<10: continue
    y=g['IAQI_Class'].map(CLS).to_numpy()
    X=g[['CO2','PM2.5','PM10']].astype(float)
    m20=X.rolling(20,min_periods=20).mean().to_numpy()
    pos=np.arange(len(g))
    labels.append(y); pos_in_seg.append(pos)
    for h in H_LIST:
        mh=X.rolling(h,min_periods=h).mean().to_numpy()
        means[h].append(mh)
        short[h].append(X.rolling(min(h,20),min_periods=min(h,20)).mean().to_numpy())
labels=np.concatenate(labels); pos_in_seg=np.concatenate(pos_in_seg)
for h in H_LIST:
    means[h]=np.concatenate(means[h]); short[h]=np.concatenate(short[h])

def evaluate(mask,h):
    mh=means[h][mask]; ms=short[h][mask]
    c8=(4*mh+4*ms)/8          # same combination as the label: 4*avg4h + 4*avg1h over 8
    pred=formula_class(c8[:,0],c8[:,1],c8[:,2])
    y=labels[mask]
    P,R,F,_=precision_recall_fscore_support(y,pred,labels=range(6),zero_division=0)
    return accuracy_score(y,pred)*100, F.mean(), F[5]

# A) common subset: rows with a full 4-hour history (position >= 79) so every H is possible
maskA=pos_in_seg>=HMAX-1
print("\n=== A) Common subset: rows with 80 readings of history, n =",maskA.sum(),"===")
print("history(readings)  history(min)  accuracy%  macroF1  F1_Hazardous")
for h in H_LIST:
    a,f,fh=evaluate(maskA,h); print(f"{h:>8d} {h*3:>12d} {a:>12.2f} {f:>9.4f} {fh:>10.4f}")

# B) check: 10-reading history on every test row that can finish a 10-reading window (should be ~91.23% / 0.656)
maskB=(pos_in_seg>=9)&~np.isnan(means[10][:,0])
a,f,fh=evaluate(maskB,10)
print("\n=== B) Check, all test windows of 10 readings, n =",maskB.sum(),"===")
print(f"accuracy={a:.2f}%  macroF1={f:.4f}  F1_Hazardous={fh:.4f}   (paper: 91.23% / 0.656)")

# C) Save what is needed to evaluate the NETWORKS on the same rows (same order as make_windows)
win = pos_in_seg >= 9                      # rows that finish a 10-reading window
y_w = labels[win]
mask_w = pos_in_seg[win] >= HMAX - 1       # windows whose last reading has a full 4-hour history
print("\n=== C) Saved for exact comparison ===")
print("windows:", len(y_w), "| with full 4-h history:", int(mask_w.sum()))
print("class counts (should be 260,276 / 97,325 / 27,614 / 2,004 / 576 / 113):", np.bincount(y_w, minlength=6))
np.save('/kaggle/working/y_test_w10.npy', y_w)
np.save('/kaggle/working/subset_mask_w10.npy', mask_w)
print("saved: /kaggle/working/y_test_w10.npy and /kaggle/working/subset_mask_w10.npy")
