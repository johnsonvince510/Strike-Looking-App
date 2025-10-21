import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Pitch Strike-Look App", layout="wide")

# ------------------ Constants ------------------
REQUIRED_COLS = [
    "x0","y0","z0","vx0","vy0","vz0","ax0","ay0","az0",
    "Pitcher","PitchNo","TaggedPitchType","RelSpeed","PitchCall"
]
DEFAULT_SZ_BOT, DEFAULT_SZ_TOP = 1.6, 3.5
PLATE_HALF_FT = 0.708
BALL_RADIUS_FT = 0.120
EPS_EDGE, EPS_VERT = 0.03, 0.03
X_LIM_PLATE = PLATE_HALF_FT + BALL_RADIUS_FT
Y_FRONT_PLATE = 0.0
PLATE_THICKNESS_FT = 17.0/12.0
Y_BACK_PLATE = Y_FRONT_PLATE - PLATE_THICKNESS_FT
DT = 0.001
T_MAX = 0.9

GREEN = "#00A000"
RED = "#CC0000"
BLACK = "#000000"

# ------------------ Helpers ------------------
def add_count_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Balls" in df.columns and "Strikes" in df.columns:
        b = pd.to_numeric(df["Balls"], errors="coerce").fillna(-1).astype(int)
        s = pd.to_numeric(df["Strikes"], errors="coerce").fillna(-1).astype(int)
        df["Count"] = b.astype(str) + "-" + s.astype(str)
        df["CountState"] = np.select([b > s, b == s, b < s], ["Behind","Even","Ahead"], default="Unknown")
    else:
        df["Count"] = "Unknown"
        df["CountState"] = "Unknown"
    return df

# ------------------ Physics / Geometry ------------------
def kinematics(row, t):
    x0, y0, z0 = float(row["x0"]), float(row["y0"]), float(row["z0"])
    vx0, vy0, vz0 = float(row["vx0"]), float(row["vy0"]), float(row["vz0"])
    ax0, ay0, az0 = float(row["ax0"]), float(row["ay0"]), float(row["az0"])
    x = x0 + vx0*t + 0.5*ax0*(t**2)
    y = y0 + vy0*t + 0.5*ay0*(t**2)
    z = z0 + vz0*t + 0.5*az0*(t**2)
    return x, y, z

def time_vector_to_back_of_plate(row):
    y0, vy0, ay0 = float(row["y0"]), float(row["vy0"]), float(row["ay0"])
    A, B, C = 0.5*ay0, vy0, y0 - Y_BACK_PLATE
    t_end = T_MAX
    if abs(A) > 1e-12:
        disc = B*B - 4*A*C
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            r1 = (-B - sqrt_disc) / (2*A)
            r2 = (-B + sqrt_disc) / (2*A)
            roots = [r for r in (r1, r2) if r >= 0]
            if roots:
                t_end = min(roots) + 0.02
    elif abs(B) > 1e-12:
        t_lin = (Y_BACK_PLATE - y0) / B
        if t_lin >= 0:
            t_end = min(t_lin + 0.02, T_MAX)
    t = np.arange(0.0, min(t_end, T_MAX) + 1e-12, DT)
    return t

def moving_corridor_inside(row, x, y, z):
    sz_bot = float(row.get("sz_bot", DEFAULT_SZ_BOT))
    sz_top = float(row.get("sz_top", DEFAULT_SZ_TOP))
    z_mid = 0.5 * (sz_bot + sz_top)
    zone_half = 0.5 * (sz_top - sz_bot)
    lateral_tol = PLATE_HALF_FT + BALL_RADIUS_FT + EPS_EDGE
    vertical_tol = zone_half + BALL_RADIUS_FT + EPS_VERT
    x0, y0, z0 = float(row["x0"]), float(row["y0"]), float(row["z0"])
    denom = (y0 - Y_FRONT_PLATE) if abs(y0 - Y_FRONT_PLATE) > 1e-9 else 1.0
    alpha = np.clip((y0 - y) / denom, 0.0, 1.0)
    x_c = x0 + (0.0 - x0) * alpha
    z_c = z0 + (z_mid - z0) * alpha
    inside = (np.abs(x - x_c) <= lateral_tol) & (np.abs(z - z_c) <= vertical_tol)
    return inside

def ends_as_strike(row, x, y, z):
    sz_bot = float(row.get("sz_bot", DEFAULT_SZ_BOT))
    sz_top = float(row.get("sz_top", DEFAULT_SZ_TOP))
    in_plate_zone = (
        (np.abs(x) <= X_LIM_PLATE) &
        (z >= (sz_bot - BALL_RADIUS_FT)) & (z <= (sz_top + BALL_RADIUS_FT)) &
        (y <= Y_FRONT_PLATE) & (y >= Y_BACK_PLATE)
    )
    return bool(in_plate_zone.any())

def ensure_ends_as_strike(df):
    if "EndsAsStrike" in df.columns:
        return df
    df = df.copy()
    def _eas(row):
        t = time_vector_to_back_of_plate(row)
        x, y, z = kinematics(row, t)
        return ends_as_strike(row, x, y, z)
    df["EndsAsStrike"] = df.apply(_eas, axis=1)
    return df

def arc_length(x, y, z):
    if len(x) <= 1:
        return np.array([0.0])
    seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    return np.concatenate([[0.0], np.cumsum(seg)])

# ------------------ Strike-look segmentation ------------------
def simulate_pitch_segments(row):
    t = time_vector_to_back_of_plate(row)
    x, y, z = kinematics(row, t)
    if ends_as_strike(row, x, y, z):
        return dict(t=t, x=x, y=y, z=z, green_idx=(0, len(t)), red_idx=None, diamond=None, strike=True)
    inside = moving_corridor_inside(row, x, y, z)
    outside_idx = np.where(~inside)[0]
    if len(outside_idx) > 0:
        i = int(outside_idx[0])
        diamond = dict(t_ms=float(t[i] * 1000.0), x=float(x[i]), y=float(y[i]), z=float(z[i]))
        return dict(t=t, x=x, y=y, z=z, green_idx=(0, i), red_idx=(i, len(t)), diamond=diamond, strike=False)
    else:
        return dict(t=t, x=x, y=y, z=z, green_idx=(0, len(t)), red_idx=None, diamond=None, strike=False)

def compute_metrics(df):
    rows = []
    for _, row in df.iterrows():
        sim = simulate_pitch_segments(row)
        t = sim["t"]
        if sim["diamond"] is None:
            strike_look = 1.0
            inf_time_ms = np.nan
            inf_dist_ft = np.nan
        else:
            i = sim["green_idx"][1]
            cumdist = arc_length(sim["x"], sim["y"], sim["z"])
            strike_look = (t[i] - t[0]) / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0.0
            inf_time_ms = float(t[i] * 1000.0)
            inf_dist_ft = float(cumdist[i])
        rows.append(dict(
            Pitcher=row["Pitcher"],
            TaggedPitchType=row["TaggedPitchType"],
            PitchNo=row["PitchNo"],
            RelSpeed=row["RelSpeed"],
            EndsAsStrike=sim["strike"],
            StrikeLookPct=strike_look,
            InferenceTime_ms=inf_time_ms,
            InferenceDistance_ft=inf_dist_ft,
            PitchCall=row["PitchCall"],
            Balls=row["Balls"] if "Balls" in row.index else np.nan,
            Strikes=row["Strikes"] if "Strikes" in row.index else np.nan
        ))
    return pd.DataFrame(rows)

def add_outcome_columns(met_df):
    met_df = ensure_ends_as_strike(met_df)
    pc = met_df["PitchCall"].astype(str).str.strip()
    swing_labels = {"StrikeSwinging", "SwingingStrike", "Foul", "FoulBall", "InPlay", "InPlayNoOut", "InPlayOut", "FoulTip"}
    miss_labels = {"StrikeSwinging", "SwingingStrike"}
    met_df["Swing"] = pc.isin(swing_labels)
    met_df["Miss"] = pc.isin(miss_labels)
    met_df["Chase"] = met_df["Swing"] & (~met_df["EndsAsStrike"])
    # Count
    met_df = add_count_fields(met_df)
    return met_df

# ------------------ Tunneling utilities ------------------
def time_grid_local():
    return np.arange(0.0, 0.6 + 1e-12, DT)

def simulate_to_grid(row, t_grid):
    t = time_vector_to_back_of_plate(row)
    x, y, z = kinematics(row, t)
    X = np.full_like(t_grid, np.nan, dtype=float)
    Y = np.full_like(t_grid, np.nan, dtype=float)
    Z = np.full_like(t_grid, np.nan, dtype=float)
    if len(t) >= 2:
        mask = t_grid <= t[-1]
        X[mask] = np.interp(t_grid[mask], t, x)
        Y[mask] = np.interp(t_grid[mask], t, y)
        Z[mask] = np.interp(t_grid[mask], t, z)
    return X, Y, Z

def _stack_and_trim(arrs, min_frac=0.5):
    A = np.vstack(arrs)
    counts = np.sum(~np.isnan(A), axis=0)
    min_count = max(3, int(np.ceil(min_frac * A.shape[0])))
    valid_idx = np.where(counts >= min_count)[0]
    if len(valid_idx) == 0:
        return A[:, :1], 0
    last = valid_idx[-1]
    return A[:, :last + 1], last

def _smooth_running_mean(v, w=7):
    if v.size < 3 or w <= 1 or v.size < w:
        return v
    kernel = np.ones(w) / w
    out = np.convolve(v, kernel, mode="same")
    out[:w // 2] = v[:w // 2]
    out[-(w // 2):] = v[-(w // 2):]
    return out

def avg_trajectory_for_type(df_p, pitch_type, t_grid, min_pitches=3):
    sub = df_p[df_p["TaggedPitchType"] == pitch_type]
    if len(sub) < min_pitches:
        return None
    Xs, Ys, Zs = [], [], []
    for _, row in sub.iterrows():
        X, Y, Z = simulate_to_grid(row, t_grid)
        Xs.append(X); Ys.append(Y); Zs.append(Z)
    Xmat, _ = _stack_and_trim(Xs, min_frac=0.5)
    Ymat, _ = _stack_and_trim(Ys, min_frac=0.5)
    Zmat, _ = _stack_and_trim(Zs, min_frac=0.5)
    with np.errstate(invalid="ignore"):
        X_mean = np.nanmean(Xmat, axis=0)
        Y_mean = np.nanmean(Ymat, axis=0)
        Z_mean = np.nanmean(Zmat, axis=0)
    X_mean = _smooth_running_mean(X_mean, w=7)
    Y_mean = _smooth_running_mean(Y_mean, w=7)
    Z_mean = _smooth_running_mean(Z_mean, w=7)
    pad = len(t_grid) - len(X_mean)
    if pad > 0:
        X_mean = np.concatenate([X_mean, np.full(pad, np.nan)])
        Y_mean = np.concatenate([Y_mean, np.full(pad, np.nan)])
        Z_mean = np.concatenate([Z_mean, np.full(pad, np.nan)])
    return X_mean, Y_mean, Z_mean, len(sub)

def tunneling_between_types(avgA, avgB, t_grid, sep_thresh_ft=0.25):
    XA, YA, ZA, nA = avgA
    XB, YB, ZB, nB = avgB
    valid = ~np.isnan(XA) & ~np.isnan(XB)
    if valid.sum() < 2:
        return None
    d = np.sqrt((XA - XB) ** 2 + (YA - YB) ** 2 + (ZA - ZB) ** 2)
    d[~valid] = np.nan
    idx = np.where((d > sep_thresh_ft) & valid)[0]
    if len(idx) > 0:
        i_sep = int(idx[0])
        t_sep = t_grid[i_sep]
        segA = np.sqrt(np.diff(XA[:i_sep + 1]) ** 2 + np.diff(YA[:i_sep + 1]) ** 2 + np.diff(ZA[:i_sep + 1]) ** 2)
        distA = float(np.nansum(segA))
    else:
        i_valid = np.where(valid)[0]
        i_sep = int(i_valid[-1])
        t_sep = t_grid[i_sep]
        segA = np.sqrt(np.diff(XA[:i_sep + 1]) ** 2 + np.diff(YA[:i_sep + 1]) ** 2 + np.diff(ZA[:i_sep + 1]) ** 2)
        distA = float(np.nansum(segA))
    i_last = int(np.where(valid)[0][-1])
    entry_gap = float(d[i_last]) if not np.isnan(d[i_last]) else float("nan")
    return dict(
        TunnelingTime_ms=float(t_sep * 1000.0),
        TunnelingDist_ft=distA,
        EntryGap_ft=entry_gap,
        i_sep=i_sep,
        i_last=i_last,
        nA=int(nA),
        nB=int(nB),
    )

def build_tunneling_table_directed_from_pool(df_pool, pitcher, sep_thresh_in=3.0, min_pitches=3):
    df_p = df_pool[df_pool["Pitcher"] == pitcher].copy()
    if df_p.empty:
        return pd.DataFrame(), None
    t_grid = time_grid_local()
    types = sorted(df_p["TaggedPitchType"].dropna().unique().tolist())
    avg_map = {}
    for pt in types:
        avg = avg_trajectory_for_type(df_p, pt, t_grid, min_pitches=min_pitches)
        if avg is not None:
            avg_map[pt] = avg

    type_mean_speed = df_p.groupby("TaggedPitchType")["RelSpeed"].mean().to_dict()
    type_flight_time = {}
    for pt, (X, Y, Z, n) in avg_map.items():
        valid = ~np.isnan(X)
        if valid.any():
            i_last = int(np.where(valid)[0][-1])
            type_flight_time[pt] = float(t_grid[i_last])
        else:
            type_flight_time[pt] = np.nan

    results = []
    sep_thresh_ft = sep_thresh_in / 12.0
    for a in types:
        for b in types:
            if a == b:
                continue
            if a not in avg_map or b not in avg_map:
                continue
            met = tunneling_between_types(avg_map[a], avg_map[b], t_grid, sep_thresh_ft=sep_thresh_ft)
            if met is None:
                continue
            row = dict(Pitcher=pitcher, TypeA=a, TypeB=b, **met,
                       MeanVeloA=float(type_mean_speed.get(a, np.nan)),
                       MeanVeloB=float(type_mean_speed.get(b, np.nan)),
                       TflightA=float(type_flight_time.get(a, np.nan)),
                       TflightB=float(type_flight_time.get(b, np.nan)))
            results.append(row)
    tbl = pd.DataFrame(results)
    return tbl, dict(t_grid=t_grid, avg_map=avg_map, df_p=df_p)

def plot_tunneling_pair(avg_map, t_grid, typeA, typeB, i_sep):
    XA, YA, ZA, _ = avg_map[typeA]
    XB, YB, ZB, _ = avg_map[typeB]
    valid = (~np.isnan(XA)) & (~np.isnan(XB))
    i_last = int(np.where(valid)[0][-1]) if valid.any() else min(len(XA), len(XB)) - 1
    i_sep = min(i_sep, i_last)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=XA[:i_sep + 1], y=YA[:i_sep + 1], z=ZA[:i_sep + 1], mode="lines",
                               line=dict(width=6, color=GREEN), name=str(typeA) + " (tunnel)"))
    fig.add_trace(go.Scatter3d(x=XB[:i_sep + 1], y=YB[:i_sep + 1], z=ZB[:i_sep + 1], mode="lines",
                               line=dict(width=6, color=GREEN), name=str(typeB) + " (tunnel)"))
    fig.add_trace(go.Scatter3d(x=XA[i_sep:i_last + 1], y=YA[i_sep:i_last + 1], z=ZA[i_sep:i_last + 1], mode="lines",
                               line=dict(width=6, color=RED), name=str(typeA) + " (diverge)"))
    fig.add_trace(go.Scatter3d(x=XB[i_sep:i_last + 1], y=YB[i_sep:i_last + 1], z=ZB[i_sep:i_last + 1], mode="lines",
                               line=dict(width=6, color=RED), name=str(typeB) + " (diverge)"))
    fig.update_layout(height=650, scene=dict(
        xaxis_title="X (ft) — plate L/R",
        yaxis_title="Y (ft) — pitcher → catcher",
        zaxis_title="Z (ft) — height",
        aspectmode="data"
    ), legend=dict(bgcolor="rgba(255,255,255,0.85)"))
    return fig

def release_consistency_tables(df, pitcher):
    sub = df[df["Pitcher"] == pitcher].copy()
    if sub.empty:
        return pd.DataFrame(), None
    g = sub.groupby("TaggedPitchType").agg(
        N=("PitchNo", "count"),
        x0_mean=("x0", "mean"), x0_std=("x0", "std"),
        z0_mean=("z0", "mean"), z0_std=("z0", "std")
    ).reset_index()
    g["xz_std"] = np.sqrt((g["x0_std"].fillna(0) ** 2 + g["z0_std"].fillna(0) ** 2))
    g["ConsistencyScore"] = 1.0 / (1.0 + g["xz_std"])
    return g, sub

def add_tunneling_score(tbl, rel_tbl, weights):
    if tbl.empty:
        return tbl
    w_time = float(weights.get("time", 0.4))
    w_dist = float(weights.get("dist", 0.3))
    w_gap  = float(weights.get("gap", 0.2))
    w_rel  = float(weights.get("rel", 0.1))
    w_sum = w_time + w_dist + w_gap + w_rel
    if w_sum == 0:
        w_time = 1.0; w_dist = w_gap = w_rel = 0.0; w_sum = 1.0
    w_time /= w_sum; w_dist /= w_sum; w_gap /= w_sum; w_rel /= w_sum

    out = tbl.copy()
    tmax = float(np.nanmax(out["TunnelingTime_ms"])) if "TunnelingTime_ms" in out.columns and len(out) > 0 else np.nan
    abs_norm = out["TunnelingTime_ms"] / tmax if tmax > 0 else 0.0
    shared_t = np.minimum(out.get("TflightA", np.nan), out.get("TflightB", np.nan))
    pct_of_flight = out["TunnelingTime_ms"] / (shared_t * 1000.0)
    pct_of_flight = pct_of_flight.clip(0, 1).fillna(0.0)
    alpha = 0.60
    t_blend = alpha * abs_norm.fillna(0.0) + (1 - alpha) * pct_of_flight
    dv = (out.get("MeanVeloA", np.nan) - out.get("MeanVeloB", np.nan)).abs()
    k = 0.35
    damp = 1.0 / (1.0 + k * (dv / 10.0))
    damp = damp.fillna(1.0)
    out["time_norm"] = (t_blend * damp).astype(float)

    dmax = float(np.nanmax(out["TunnelingDist_ft"])) if "TunnelingDist_ft" in out.columns and len(out) > 0 else np.nan
    out["dist_norm"] = out["TunnelingDist_ft"] / dmax if dmax > 0 else 0.0
    cap_ft = 12.0 / 12.0
    out["gap_norm"]  = 1.0 - np.clip(out["EntryGap_ft"], 0, cap_ft) / cap_ft

    rel_map = {}
    if rel_tbl is not None and not rel_tbl.empty:
        for _, r in rel_tbl.iterrows():
            rel_map[str(r["TaggedPitchType"])] = float(r["ConsistencyScore"])
    def pair_rel(a, b):
        ra = rel_map.get(str(a), np.nan)
        rb = rel_map.get(str(b), np.nan)
        return float(np.nanmean([ra, rb]))
    out["rel_norm"] = [pair_rel(a, b) for a, b in zip(out["TypeA"], out["TypeB"])]
    if not np.isnan(out["rel_norm"]).all():
        out["rel_norm"] = out["rel_norm"].fillna(float(np.nanmean(out["rel_norm"])))
    else:
        out["rel_norm"] = 0.5

    out["TunnelingScore"] = 100.0 * (
        w_time * out["time_norm"].fillna(0) +
        w_dist * out["dist_norm"].fillna(0) +
        w_gap  * out["gap_norm"].fillna(0) +
        w_rel  * out["rel_norm"].fillna(0)
    )
    return out

# ------------------ Sequences & outcomes ------------------
def add_sequence_keys(df):
    df = df.copy()
    has_all = all([c in df.columns for c in ["Inning", "Top/Bottom", "PAofInning", "PitchofPA"]])
    if has_all:
        df["_seq_group"] = (df["Pitcher"].astype(str) + "|" +
                            df["Inning"].astype(str) + "|" +
                            df["Top/Bottom"].astype(str) + "|" +
                            df["PAofInning"].astype(str))
        df["_seq_order"] = df["PitchofPA"]
    else:
        if "PitchNo" in df.columns:
            df["_seq_group"] = df["Pitcher"].astype(str)
            df["_seq_order"] = df["PitchNo"]
        else:
            df["_seq_group"] = df["Pitcher"].astype(str)
            df["_seq_order"] = np.arange(len(df))
    return df

def compute_outcome_flags(df):
    df = ensure_ends_as_strike(df)
    pc = df["PitchCall"].astype(str).str.strip()
    swing_labels = {"StrikeSwinging", "SwingingStrike", "Foul", "FoulBall", "InPlay", "InPlayNoOut", "InPlayOut", "FoulTip"}
    miss_labels = {"StrikeSwinging", "SwingingStrike"}
    df["Swing"] = pc.isin(swing_labels)
    df["Miss"] = pc.isin(miss_labels)
    df["Chase"] = df["Swing"] & (~df["EndsAsStrike"])
    return df

def outcome_lift_next_pitch(df, pitcher, pairs_tbl, count_mode, sel_counts, sel_states, scope):
    sub = add_sequence_keys(df[df["Pitcher"] == pitcher].copy())
    sub = compute_outcome_flags(add_count_fields(sub)).sort_values(["_seq_group", "_seq_order"])
    sub["PrevPitchType"] = sub.groupby("_seq_group")["TaggedPitchType"].shift(1)
    sub["Count_A"] = sub.groupby("_seq_group")["Count"].shift(1)
    sub["CountState_A"] = sub.groupby("_seq_group")["CountState"].shift(1)

    def mask_for(frame, at="B"):
        if count_mode == "Exact counts (B-S)":
            if not sel_counts: return np.ones(len(frame), dtype=bool)
            col = "Count_A" if at=="A" else "Count"
            return frame[col].isin(sel_counts)
        elif count_mode == "Count state (Ahead/Even/Behind)":
            if not sel_states: return np.ones(len(frame), dtype=bool)
            col = "CountState_A" if at=="A" else "CountState"
            return frame[col].isin(sel_states)
        return np.ones(len(frame), dtype=bool)

    if scope == "Count at A (prev pitch)":
        sub = sub[mask_for(sub, "A")]
    elif scope == "Count at B (current pitch)":
        sub = sub[mask_for(sub, "B")]
    else:
        sub = sub[mask_for(sub, "A") & mask_for(sub, "B")]

    base_pitcher_whiff = float(sub["Miss"].mean()) if len(sub) else np.nan
    base_pitcher_chase = float(sub["Chase"].mean()) if len(sub) else np.nan
    base_type = sub.groupby("TaggedPitchType").agg(
        base_Whiff=("Miss", "mean"),
        base_Chase=("Chase", "mean")
    )

    rows = []
    for _, r in pairs_tbl.iterrows():
        A, B = r["TypeA"], r["TypeB"]
        mask = (sub["PrevPitchType"] == A) & (sub["TaggedPitchType"] == B)
        cur = sub[mask]
        if len(cur) == 0:
            rows.append(dict(TypeA=A, TypeB=B, N=0, Whiff=np.nan, Chase=np.nan,
                             Lift_vsType_Whiff=np.nan, Lift_vsType_Chase=np.nan,
                             Lift_vsPitcher_Whiff=np.nan, Lift_vsPitcher_Chase=np.nan))
            continue
        whiff = float(cur["Miss"].mean())
        chase = float(cur["Chase"].mean())
        if B in base_type.index:
            base_w = float(base_type.loc[B, "base_Whiff"]); base_c = float(base_type.loc[B, "base_Chase"])
        else:
            base_w = np.nan; base_c = np.nan
        rows.append(dict(
            TypeA=A, TypeB=B, N=len(cur), Whiff=whiff, Chase=chase,
            Lift_vsType_Whiff=whiff - base_w if pd.notna(base_w) else np.nan,
            Lift_vsType_Chase=chase - base_c if pd.notna(base_c) else np.nan,
            Lift_vsPitcher_Whiff=whiff - base_pitcher_whiff if pd.notna(base_pitcher_whiff) else np.nan,
            Lift_vsPitcher_Chase=chase - base_pitcher_chase if pd.notna(base_pitcher_chase) else np.nan
        ))
    return pd.DataFrame(rows)

# ------------------ Correlations helpers ------------------
def smart_quantile_bins(series, max_bins=6, min_bins=4, min_count=12):
    s = pd.to_numeric(series, errors="coerce").dropna().values
    if s.size == 0:
        return None
    for k in range(max_bins, min_bins-1, -1):
        qs = np.linspace(0, 1, k+1)
        edges = np.unique(np.quantile(s, qs))
        if len(edges) - 1 < min_bins:
            continue
        cats = pd.cut(series, bins=edges, include_lowest=True, right=True, duplicates="drop")
        counts = cats.value_counts(dropna=False).sort_index()
        if (counts >= min_count).all():
            return edges
    qs = np.linspace(0, 1, min_bins+1)
    edges = np.unique(np.quantile(s, qs))
    if len(edges) < 2:
        lo = float(np.nanmin(s)); hi = float(np.nanmax(s))
        edges = np.array([lo, hi])
    return edges

def rates_by_bin(df_local, value_col, edges):
    cats = pd.cut(df_local[value_col], bins=edges, include_lowest=True, right=True, duplicates="drop")
    g = df_local.groupby(cats).agg(
        Pitches=("PitchNo","count"),
        SwingPct=("Swing","mean"),
        WhiffPct=("Miss","mean"),
        ChasePct=("Chase","mean"),
        StrikePct=("EndsAsStrike","mean")
    ).reset_index().rename(columns={value_col:"Bin"})
    for c in ["SwingPct","WhiffPct","ChasePct","StrikePct"]:
        g[c] = (g[c]*100).round(1)
    g["Bin"] = g["Bin"].astype(str)
    return g

def corr_pair(df_local, x, y):
    d = df_local[[x,y]].dropna()
    if len(d) < 3: 
        return np.nan, np.nan, len(d)
    pear = stats.pearsonr(d[x], d[y]).statistic
    spear = stats.spearmanr(d[x], d[y]).correlation
    return pear, spear, len(d)

# ------------------ App Layout ------------------
st.title("Pitch Strike‑Look App — v5g5 (All‑in‑one)")

uploaded = st.file_uploader("Upload a CSV (TrackMan-style columns)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin. Required: " + ", ".join(REQUIRED_COLS))
    st.stop()

df = pd.read_csv(uploaded)
miss = [c for c in REQUIRED_COLS if c not in df.columns]
if miss:
    st.error("Missing required columns: " + ", ".join(miss))
    st.stop()

if "sz_bot" not in df.columns:
    df["sz_bot"] = DEFAULT_SZ_BOT
if "sz_top" not in df.columns:
    df["sz_top"] = DEFAULT_SZ_TOP
df = df.dropna(subset=REQUIRED_COLS).copy()
df = add_count_fields(df)

sim_results = [simulate_pitch_segments(row) for _, row in df.iterrows()]
metrics = compute_metrics(df)
metrics = add_outcome_columns(metrics)

tab1, tab2, tab3, tab4 = st.tabs(["🧭 3D Visualizer", "📊 Coach Dashboard", "📈 Correlations", "🧵 Tunneling"])

# ---------- 3D Visualizer ----------
with tab1:
    st.subheader("3D Visualizer")
    pitchers = ["All"] + sorted(df["Pitcher"].dropna().unique().tolist())
    types = ["All"] + sorted(df["TaggedPitchType"].dropna().unique().tolist())
    colA, colB = st.columns(2)
    sel_pitcher = colA.selectbox("Pitcher", pitchers, index=0)
    sel_type = colB.selectbox("Pitch Type", types, index=0)

    mask = np.ones(len(df), dtype=bool)
    if sel_pitcher != "All":
        mask &= (df["Pitcher"].values == sel_pitcher)
    if sel_type != "All":
        mask &= (df["TaggedPitchType"].values == sel_type)

    fig = go.Figure()
    for ((_, row), sim, keep) in zip(df.iterrows(), sim_results, mask):
        if not keep: continue
        name_base = f"{row['Pitcher']} — #{int(row['PitchNo'])} — {row['TaggedPitchType']} — {row['RelSpeed']} mph"
        g0, g1 = sim["green_idx"]
        gx, gy, gz = sim["x"][g0:g1], sim["y"][g0:g1], sim["z"][g0:g1]
        fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", line=dict(width=6, color=GREEN), name=name_base + " [GREEN]"))
        if sim["red_idx"] is not None:
            r0, r1 = sim["red_idx"]
            rx, ry, rz = sim["x"][r0:r1], sim["y"][r0:r1], sim["z"][r0:r1]
            fig.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines", line=dict(width=6, color=RED), name=name_base + " [RED]"))
        if sim["diamond"] is not None:
            dmd = sim["diamond"]
            fig.add_trace(go.Scatter3d(x=[dmd["x"]], y=[dmd["y"]], z=[dmd["z"]], mode="markers", marker=dict(size=7, color=BLACK, symbol="diamond"), name=name_base + " [INFERENCE]"))
    fig.update_layout(height=700, scene=dict(xaxis_title="X (ft) — plate left/right", yaxis_title="Y (ft) — pitcher → catcher", zaxis_title="Z (ft) — height", aspectmode="data"), legend=dict(itemsizing="trace", bgcolor="rgba(255,255,255,0.85)"))
    st.plotly_chart(fig, use_container_width=True)

# ---------- Coach Dashboard ----------
with tab2:
    st.subheader("Coach Dashboard")
    met = metrics.copy()
    met_display = met.copy()
    met_display["StrikeLookPct"] = (met_display["StrikeLookPct"] * 100).round(1)
    st.write("Per‑Pitch Metrics")
    st.dataframe(met_display)

    by_pitcher = met.groupby("Pitcher").agg(
        Pitches=("PitchNo", "count"),
        StrikePct=("EndsAsStrike", "mean"),
        AvgStrikeLookPct=("StrikeLookPct", "mean"),
        MedianInferenceDist_ft=("InferenceDistance_ft", "median"),
        MeanInferenceDist_ft=("InferenceDistance_ft", "mean")
    ).reset_index()
    by_pitcher["StrikePct"] = (by_pitcher["StrikePct"] * 100).round(1)
    by_pitcher["AvgStrikeLookPct"] = (by_pitcher["AvgStrikeLookPct"] * 100).round(1)
    st.write("By Pitcher")
    st.dataframe(by_pitcher)

# ---------- Correlations ----------
with tab3:
    st.subheader("Correlations")

    # Dataset‑wide buckets
    st.markdown("### Strike‑Look % buckets — dataset‑wide")
    edges_sl = smart_quantile_bins(metrics["StrikeLookPct"], max_bins=6, min_bins=4, min_count=8)
    if edges_sl is None:
        st.info("Not enough data to bin Strike‑Look %.")
    else:
        tbl_sl = rates_by_bin(metrics, "StrikeLookPct", edges_sl)
        st.dataframe(tbl_sl)

    st.markdown("### Inference Time (ms) buckets — dataset‑wide")
    met_it = metrics[metrics["InferenceTime_ms"].notna()].copy()
    edges_it = smart_quantile_bins(met_it["InferenceTime_ms"], max_bins=6, min_bins=4, min_count=8)
    if edges_it is None:
        st.info("Not enough data to bin Inference Time.")
    else:
        tbl_it = rates_by_bin(met_it, "InferenceTime_ms", edges_it)
        st.dataframe(tbl_it)

    # Balls‑only correlation summary and scatterplots
    st.markdown("---")
    st.markdown("### Balls‑only: Correlation summary (pitcher‑level) & scatterplots")
    balls_only = metrics[~metrics["EndsAsStrike"]].copy()

    pitcher_agg = balls_only.groupby("Pitcher").agg(
        Pitches=("PitchNo","count"),
        StrikeLookPct=("StrikeLookPct","mean"),
        SwingPct=("Swing","mean"),
        WhiffPct=("Miss","mean"),
        ChasePct=("Chase","mean"),
        InferenceTime_ms_mean=("InferenceTime_ms","mean")
    ).reset_index()

    rows = []
    for x,y in [("StrikeLookPct","SwingPct"), ("StrikeLookPct","WhiffPct"), ("StrikeLookPct","ChasePct")]:
        p,s,n = corr_pair(pitcher_agg, x, y)
        rows.append(dict(Metric=f"{x} vs {y}", Pearson=round(p,3) if pd.notna(p) else np.nan, Spearman=round(s,3) if pd.notna(s) else np.nan, N_pitchers=n))
    st.dataframe(pd.DataFrame(rows))

    st.markdown("#### Scatterplots (balls‑only, pitcher‑level)")
    def scatter_with_fit_local(st_df, xcol, ycol, title):
        d = st_df[[xcol, ycol, "Pitches"]].dropna().copy()
        if len(d) < 2:
            st.info("Not enough data to draw: " + title); return
        x = d[xcol] * 100.0; y = d[ycol] * 100.0
        sizes = 20 + 2 * np.sqrt(d["Pitches"].values)
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=sizes)
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, m * xs + b)
        ax.set_xlabel(xcol + " (%)"); ax.set_ylabel(ycol + " (%)"); ax.set_title(title)
        fig.tight_layout(); st.pyplot(fig)

    scatter_with_fit_local(pitcher_agg, "StrikeLookPct", "SwingPct", "Strike‑Look % vs Swing % (balls‑only)")
    scatter_with_fit_local(pitcher_agg, "StrikeLookPct", "WhiffPct", "Strike‑Look % vs Whiff % (balls‑only)")
    scatter_with_fit_local(pitcher_agg, "StrikeLookPct", "ChasePct", "Strike‑Look % vs Chase % (balls‑only)")

# ---------- Tunneling with COUNT FILTERS ----------
with tab4:
    st.subheader("Tunneling — within‑pitcher, with count filters")
    pitchers_all = sorted(df["Pitcher"].dropna().unique().tolist())
    sel_pitcher = st.selectbox("Pitcher", pitchers_all, index=0, key="tun_pitcher")
    sep_thresh_in = st.slider("Separation threshold (inches)", 1.0, 6.0, 3.0, 0.5)
    min_pitches = st.slider("Min pitches per type", 2, 10, 3, 1)

    # COUNT FILTER UI
    st.markdown("**Count context for tunneling pools**")
    col_cf1, col_cf2 = st.columns(2)
    count_mode = col_cf1.selectbox(
        "Filter mode", ["All counts","Exact counts (B‑S)","Count state (Ahead/Even/Behind)"], index=0
    )
    counts_available = sorted(df["Count"].dropna().unique().tolist())
    states_available = ["Ahead","Even","Behind"]
    sel_counts, sel_states = [], []
    if count_mode == "Exact counts (B‑S)":
        sel_counts = col_cf2.multiselect("Counts", counts_available, default=counts_available)
    elif count_mode == "Count state (Ahead/Even/Behind)":
        sel_states = col_cf2.multiselect("States", states_available, default=states_available)

    # Apply count filters to the tunneling POOL
    df_pool = df[df["Pitcher"] == sel_pitcher].copy()
    if count_mode == "Exact counts (B‑S)" and sel_counts:
        df_pool = df_pool[df_pool["Count"].isin(sel_counts)]
    elif count_mode == "Count state (Ahead/Even/Behind)" and sel_states:
        df_pool = df_pool[df_pool["CountState"].isin(sel_states)]

    tbl, aux = build_tunneling_table_directed_from_pool(df_pool, sel_pitcher, sep_thresh_in=sep_thresh_in, min_pitches=min_pitches)
    if tbl.empty:
        st.info("Not enough data for tunneling with current count filters."); st.stop()

    rel_tbl, df_p = release_consistency_tables(df_pool, sel_pitcher)
    colw1, colw2, colw3, colw4 = st.columns(4)
    w_time = colw1.slider("Weight: Time", 0.0, 1.0, 0.4, 0.05)
    w_dist = colw2.slider("Weight: Distance", 0.0, 1.0, 0.3, 0.05)
    w_gap  = colw3.slider("Weight: EntryGap", 0.0, 1.0, 0.2, 0.05)
    w_rel  = colw4.slider("Weight: Release", 0.0, 1.0, 0.1, 0.05)

    tbl_scored = add_tunneling_score(tbl, rel_tbl, {"time": w_time, "dist": w_dist, "gap": w_gap, "rel": w_rel})
    tbl_sorted = tbl_scored.sort_values(["TunnelingScore", "TunnelingTime_ms", "TunnelingDist_ft"], ascending=[False, False, False]).reset_index(drop=True)

    subtab1, subtab2, subtab3 = st.tabs(["Tunneling + Explanations", "Release Consistency", "Outcome Lift (count‑aware)"])

    with subtab1:
        st.write("Directed tunneling rankings (A→B).")
        disp = tbl_sorted.copy()
        for c in ["TunnelingScore", "TunnelingTime_ms", "TunnelingDist_ft", "EntryGap_ft","MeanVeloA","MeanVeloB"]:
            disp[c] = disp[c].round(2)
        disp["CountContext"] = (
            ("All counts") if count_mode == "All counts" else
            ("Counts: " + ", ".join(sel_counts) if count_mode.startswith("Exact") else "States: " + ", ".join(sel_states))
        )
        st.dataframe(disp)

        labels = [f"{r.TypeA} → {r.TypeB} | Score {r.TunnelingScore:.1f}" for _, r in disp.iterrows()]
        if labels:
            sel = st.selectbox("Explain pair", labels, index=0, key="explain_pair_counts")
            r = disp.iloc[labels.index(sel)]
            if aux is not None:
                avg_map = aux["avg_map"]; t_grid = aux["t_grid"]
                fig_t = plot_tunneling_pair(avg_map, t_grid, r.TypeA, r.TypeB, int(r.i_sep))
                st.plotly_chart(fig_t, use_container_width=True)
            # Explanation bullets
            shared = np.nanmin([r.get("TflightA", np.nan), r.get("TflightB", np.nan)])
            pct = (r["TunnelingTime_ms"] / (shared * 1000.0)) if pd.notna(shared) and shared>0 else np.nan
            dv = abs(r["MeanVeloA"] - r["MeanVeloB"])
            # compute norms for explanation
            tnorm = float(tbl_scored.loc[(tbl_scored["TypeA"]==r["TypeA"]) & (tbl_scored["TypeB"]==r["TypeB"]),"time_norm"].values[0])
            dnorm = float(tbl_scored.loc[(tbl_scored["TypeA"]==r["TypeA"]) & (tbl_scored["TypeB"]==r["TypeB"]),"dist_norm"].values[0])
            gnorm = float(tbl_scored.loc[(tbl_scored["TypeA"]==r["TypeA"]) & (tbl_scored["TypeB"]==r["TypeB"]),"gap_norm"].values[0])
            rnorm = float(tbl_scored.loc[(tbl_scored["TypeA"]==r["TypeA"]) & (tbl_scored["TypeB"]==r["TypeB"]),"rel_norm"].values[0])
            ctx = r["CountContext"]
            pct_txt = f"{pct*100:.1f}%" if not np.isnan(pct) else "n/a"
            expl = (
                f"**Why Score {r['TunnelingScore']:.1f} ({ctx}):**\n"
                f"- **Time until split:** {r['TunnelingTime_ms']:.1f} ms (≈ {pct_txt} of shared flight).\n"
                f"- **Overlap distance while tunneling:** {r['TunnelingDist_ft']:.2f} ft.\n"
                f"- **Entry gap near zone:** {r['EntryGap_ft']:.2f} ft (smaller is better).\n"
                f"- **Release repeatability (avg of types):** {rnorm:.2f} (normalized).\n"
                f"- **Velo gap dampener:** ΔV = {dv:.1f} mph (applies to time component).\n"
                f"- **Weighted parts → final score:** Time {tnorm:.2f} × {w_time:.2f}, Dist {dnorm:.2f} × {w_dist:.2f}, Gap {gnorm:.2f} × {w_gap:.2f}, Release {rnorm:.2f} × {w_rel:.2f}."
            )
            st.markdown(expl)

    with subtab2:
        st.write("Release consistency by pitch type (count‑filtered pool)")
        if rel_tbl is not None and not rel_tbl.empty:
            disp_rel = rel_tbl.copy()
            for c in ["x0_mean", "z0_mean", "x0_std", "z0_std", "xz_std", "ConsistencyScore"]:
                disp_rel[c] = disp_rel[c].round(3)
            st.dataframe(disp_rel.sort_values("ConsistencyScore", ascending=False))
            types = sorted(df_p["TaggedPitchType"].dropna().unique().tolist())
            fig, ax = plt.subplots()
            for pt in types:
                d = df_p[df_p["TaggedPitchType"] == pt]
                ax.scatter(d["x0"], d["z0"], s=16, label=str(pt))
            ax.set_xlabel("Release side x0 (ft)"); ax.set_ylabel("Release height z0 (ft)")
            ax.set_title("Release clusters by pitch type"); ax.legend(loc="best", fontsize=8)
            fig.tight_layout(); st.pyplot(fig)
        else:
            st.info("No pitches available for release analysis in this count context.")

    with subtab3:
        st.write("Next Pitch Outcome — A→B (count‑aware)")
        lift_scope = st.radio("Apply count filter to:", ["Count at A (prev pitch)", "Count at B (current pitch)", "Both A and B"], index=1, horizontal=True)
        lift_next = outcome_lift_next_pitch(df, sel_pitcher, tbl_sorted, count_mode, sel_counts, sel_states, lift_scope)
        if lift_next.empty:
            st.info("No sequences found with current count filters.")
        else:
            def make_hover(r):
                whiff = r["Whiff"] * 100 if pd.notna(r["Whiff"]) else np.nan
                chase = r["Chase"] * 100 if pd.notna(r["Chase"]) else np.nan
                ltw = r["Lift_vsType_Whiff"] * 100 if pd.notna(r["Lift_vsType_Whiff"]) else np.nan
                lpw = r["Lift_vsPitcher_Whiff"] * 100 if pd.notna(r["Lift_vsPitcher_Whiff"]) else np.nan
                ltc = r["Lift_vsType_Chase"] * 100 if pd.notna(r["Lift_vsType_Chase"]) else np.nan
                lpc = r["Lift_vsPitcher_Chase"] * 100 if pd.notna(r["Lift_vsPitcher_Chase"]) else np.nan
                return (
                    f"<b>{r['TypeA']} → {r['TypeB']}</b><br>"
                    f"Whiff% after sequence: {whiff:.1f}%<br>"
                    f"Chase% after sequence: {chase:.1f}%<br><br>"
                    f"<b>Lift_vsType_Whiff</b>: {ltw:+.1f} pts vs {r['TypeB']} avg<br>"
                    f"<b>Lift_vsPitcher_Whiff</b>: {lpw:+.1f} pts vs pitcher overall<br>"
                    f"<b>Lift_vsType_Chase</b>: {ltc:+.1f} pts vs {r['TypeB']} avg<br>"
                    f"<b>Lift_vsPitcher_Chase</b>: {lpc:+.1f} pts vs pitcher overall"
                )
            lift_plot = lift_next.copy().fillna(0.0)
            lift_plot["label"] = lift_plot["TypeA"].astype(str) + "→" + lift_plot["TypeB"].astype(str)
            hover_texts = [make_hover(r) for _, r in lift_next.iterrows()]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=lift_plot["label"], y=lift_plot["Lift_vsType_Chase"] * 100.0, name="Lift vs Type (Chase)", marker_color=RED, hovertext=hover_texts, hovertemplate="%{hovertext}<extra></extra>"))
            fig_bar.add_trace(go.Bar(x=lift_plot["label"], y=lift_plot["Lift_vsPitcher_Chase"] * 100.0, name="Lift vs Pitcher (Chase)", marker_color="#666666", hovertext=hover_texts, hovertemplate="%{hovertext}<extra></extra>"))
            fig_bar.update_layout(barmode="group", xaxis_title="Sequence A→B", yaxis_title="Chase Lift (percentage points)", height=500, legend=dict(bgcolor="rgba(255,255,255,0.85)"))
            st.plotly_chart(fig_bar, use_container_width=True)
