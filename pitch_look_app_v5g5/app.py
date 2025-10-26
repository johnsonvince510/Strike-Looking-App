
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Pitch Strike-Look App", layout="wide")

REQUIRED_COLS = ["x0","y0","z0","vx0","vy0","vz0","ax0","ay0","az0","Pitcher","PitchNo","TaggedPitchType","RelSpeed","PitchCall"]
DEFAULT_SZ_BOT, DEFAULT_SZ_TOP = 1.6, 3.5
PLATE_HALF_FT = 0.708
BALL_RADIUS_FT = 0.120
EPS_EDGE, EPS_VERT = 0.03, 0.03
Y_FRONT_PLATE = 0.0
PLATE_THICKNESS_FT = 17.0/12.0
Y_BACK_PLATE = Y_FRONT_PLATE - PLATE_THICKNESS_FT
DT = 0.001
T_MAX = 0.9

# Fixed tunneling parameters (per request)
SEP_THRESH_IN = 6.0  # inches
W_TIME = 1.0
W_DIST = 1.0
W_GAP  = 0.0
W_REL  = 1.0

GREEN = "#00A000"; RED = "#CC0000"; BLACK = "#000000"; GRAY = "#666666"

def add_count_fields(df):
    df = df.copy()
    if "Balls" in df.columns and "Strikes" in df.columns:
        b = pd.to_numeric(df["Balls"], errors="coerce").fillna(-1).astype(int)
        s = pd.to_numeric(df["Strikes"], errors="coerce").fillna(-1).astype(int)
        df["Count"] = b.astype(str) + "-" + s.astype(str)
        df["CountState"] = np.select([b > s, b == s, b < s], ["Behind","Even","Ahead"], default="Unknown")
    else:
        df["Count"] = "Unknown"; df["CountState"] = "Unknown"
    return df

def kinematics(row, t):
    x0, y0, z0 = float(row["x0"]), float(row["y0"]), float(row["z0"])
    vx0, vy0, vz0 = float(row["vx0"]), float(row["vy0"]), float(row["vz0"])
    ax0, ay0, az0 = float(row["ax0"]), float(row["ay0"]), float(row["az0"])
    x = x0 + vx0*t + 0.5*ax0*(t**2); y = y0 + vy0*t + 0.5*ay0*(t**2); z = z0 + vz0*t + 0.5*az0*(t**2)
    return x, y, z

def time_vector_to_back_of_plate(row):
    y0, vy0, ay0 = float(row["y0"]), float(row["vy0"]), float(row["ay0"])
    A, B, C = 0.5*ay0, vy0, y0 - Y_BACK_PLATE
    t_end = T_MAX
    if abs(A) > 1e-12:
        disc = B*B - 4*A*C
        if disc >= 0:
            sqrt_disc = np.sqrt(disc)
            r1 = (-B - sqrt_disc) / (2*A); r2 = (-B + sqrt_disc) / (2*A)
            roots = [r for r in (r1, r2) if r >= 0]
            if roots: t_end = min(roots) + 0.02
    elif abs(B) > 1e-12:
        t_lin = (Y_BACK_PLATE - y0) / B
        if t_lin >= 0: t_end = min(t_lin + 0.02, T_MAX)
    return np.arange(0.0, min(t_end, T_MAX) + 1e-12, DT)

def moving_corridor_inside(row, x, y, z):
    sz_bot = float(row.get("sz_bot", DEFAULT_SZ_BOT)); sz_top = float(row.get("sz_top", DEFAULT_SZ_TOP))
    z_mid = 0.5 * (sz_bot + sz_top); zone_half = 0.5 * (sz_top - sz_bot)
    lateral_tol = PLATE_HALF_FT + BALL_RADIUS_FT + EPS_EDGE; vertical_tol = zone_half + BALL_RADIUS_FT + EPS_VERT
    x0, y0, z0 = float(row["x0"]), float(row["y0"]), float(row["z0"])
    denom = (y0 - Y_FRONT_PLATE) if abs(y0 - Y_FRONT_PLATE) > 1e-9 else 1.0
    alpha = np.clip((y0 - y) / denom, 0.0, 1.0)
    x_c = x0 + (0.0 - x0) * alpha; z_c = z0 + (z_mid - z0) * alpha
    return (np.abs(x - x_c) <= lateral_tol) & (np.abs(z - z_c) <= vertical_tol)

def ends_as_strike(row, x, y, z):
    sz_bot = float(row.get("sz_bot", DEFAULT_SZ_BOT)); sz_top = float(row.get("sz_top", DEFAULT_SZ_TOP))
    in_plate = (
        (np.abs(x) <= (PLATE_HALF_FT + BALL_RADIUS_FT)) &
        (z >= (sz_bot - BALL_RADIUS_FT)) & (z <= (sz_top + BALL_RADIUS_FT)) &
        (y <= Y_FRONT_PLATE) & (y >= Y_BACK_PLATE)
    )
    return bool(in_plate.any())

def ensure_ends_as_strike(df):
    if "EndsAsStrike" in df.columns: return df
    df = df.copy()
    df["EndsAsStrike"] = df.apply(lambda r: ends_as_strike(r, *kinematics(r, time_vector_to_back_of_plate(r))), axis=1)
    return df

def arc_length(x, y, z):
    if len(x) <= 1: return np.array([0.0])
    seg = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2); return np.concatenate([[0.0], np.cumsum(seg)])

def simulate_pitch_segments(row):
    t = time_vector_to_back_of_plate(row); x, y, z = kinematics(row, t)
    if ends_as_strike(row, x, y, z):
        return dict(t=t, x=x, y=y, z=z, green_idx=(0, len(t)), red_idx=None, diamond=None, strike=True)
    inside = moving_corridor_inside(row, x, y, z); outside_idx = np.where(~inside)[0]
    if len(outside_idx) > 0:
        i = int(outside_idx[0]); d = dict(t_ms=float(t[i]*1000.0), x=float(x[i]), y=float(y[i]), z=float(z[i]))
        return dict(t=t, x=x, y=y, z=z, green_idx=(0,i), red_idx=(i,len(t)), diamond=d, strike=False)
    else:
        return dict(t=t, x=x, y=y, z=z, green_idx=(0,len(t)), red_idx=None, diamond=None, strike=False)

def compute_metrics(df):
    out = []
    for _, row in df.iterrows():
        sim = simulate_pitch_segments(row); t = sim["t"]
        if sim["diamond"] is None:
            strike_look = 1.0; inf_time_ms = np.nan; inf_dist_ft = np.nan
        else:
            i = sim["green_idx"][1]; cum = arc_length(sim["x"], sim["y"], sim["z"])
            strike_look = (t[i]-t[0])/(t[-1]-t[0]) if (t[-1]-t[0])>0 else 0.0
            inf_time_ms = float(t[i]*1000.0); inf_dist_ft = float(cum[i])
        out.append(dict(Pitcher=row["Pitcher"], TaggedPitchType=row["TaggedPitchType"], PitchNo=row["PitchNo"], RelSpeed=row["RelSpeed"],
                        EndsAsStrike=sim["strike"], StrikeLookPct=strike_look, InferenceTime_ms=inf_time_ms, InferenceDistance_ft=inf_dist_ft,
                        PitchCall=row["PitchCall"], Balls=row.get("Balls", np.nan), Strikes=row.get("Strikes", np.nan)))
    return pd.DataFrame(out)

def add_outcome_columns(met_df):
    met_df = ensure_ends_as_strike(met_df)
    pc = met_df["PitchCall"].astype(str).str.strip()
    swing_labels = {"StrikeSwinging","SwingingStrike","Foul","FoulBall","InPlay","InPlayNoOut","InPlayOut","FoulTip"}
    miss_labels = {"StrikeSwinging","SwingingStrike"}
    met_df["Swing"] = pc.isin(swing_labels); met_df["Miss"] = pc.isin(miss_labels)
    met_df["Chase"] = met_df["Swing"] & (~met_df["EndsAsStrike"])
    return add_count_fields(met_df)

def add_sequence_keys(df):
    df = df.copy()
    if all(c in df.columns for c in ["Inning","Top/Bottom","PAofInning","PitchofPA"]):
        df["_seq_group"] = (df["Pitcher"].astype(str)+"|"+df["Inning"].astype(str)+"|"+df["Top/Bottom"].astype(str)+"|"+df["PAofInning"].astype(str))
        df["_seq_order"] = df["PitchofPA"]
    else:
        df["_seq_group"] = df["Pitcher"].astype(str); df["_seq_order"] = df.get("PitchNo", pd.Series(range(len(df))))
    return df

def compute_outcome_flags(df):
    df = ensure_ends_as_strike(df); pc = df["PitchCall"].astype(str).str.strip()
    swing_labels = {"StrikeSwinging","SwingingStrike","Foul","FoulBall","InPlay","InPlayNoOut","InPlayOut","FoulTip"}
    miss_labels = {"StrikeSwinging","SwingingStrike"}
    df["Swing"] = pc.isin(swing_labels); df["Miss"] = pc.isin(miss_labels); df["Chase"] = df["Swing"] & (~df["EndsAsStrike"])
    return df

def outcome_lift_same_pitch(df, pitcher, count_mode, sel_counts, sel_states, scope):
    sub = add_sequence_keys(df[df["Pitcher"] == pitcher].copy())
    sub = compute_outcome_flags(add_count_fields(sub)).sort_values(["_seq_group","_seq_order"])
    sub["PrevPitchType"] = sub.groupby("_seq_group")["TaggedPitchType"].shift(1)
    sub["Count_A"] = sub.groupby("_seq_group")["Count"].shift(1); sub["CountState_A"] = sub.groupby("_seq_group")["CountState"].shift(1)

    def mask_for(frame, at="B"):
        if count_mode == "Exact counts (B-S)":
            if not sel_counts: return np.ones(len(frame), dtype=bool)
            col = "Count_A" if at=="A" else "Count"; return frame[col].isin(sel_counts)
        elif count_mode == "Count state (Ahead/Even/Behind)":
            if not sel_states: return np.ones(len(frame), dtype=bool)
            col = "CountState_A" if at=="A" else "CountState"; return frame[col].isin(sel_states)
        return np.ones(len(frame), dtype=bool)

    if scope == "Count at A (prev pitch)":
        sub = sub[mask_for(sub, "A")]
    elif scope == "Count at B (current pitch)":
        sub = sub[mask_for(sub, "B")]
    else:
        sub = sub[mask_for(sub, "A") & mask_for(sub, "B")]

    base_pitcher_whiff = float(sub["Miss"].mean()) if len(sub) else np.nan
    base_pitcher_chase = float(sub["Chase"].mean()) if len(sub) else np.nan

    results = []
    for pt in sub["TaggedPitchType"].dropna().unique().tolist():
        cur = sub[(sub["PrevPitchType"] == pt) & (sub["TaggedPitchType"] == pt)]
        if len(cur) == 0:
            results.append(dict(Type=pt,N=0,Whiff=np.nan,Chase=np.nan,Lift_vsPitcher_Whiff=np.nan,Lift_vsPitcher_Chase=np.nan)); continue
        whiff = float(cur["Miss"].mean()); chase = float(cur["Chase"].mean())
        results.append(dict(Type=pt,N=len(cur),Whiff=whiff,Chase=chase,
                            Lift_vsPitcher_Whiff=whiff-base_pitcher_whiff if pd.notna(base_pitcher_whiff) else np.nan,
                            Lift_vsPitcher_Chase=chase-base_pitcher_chase if pd.notna(base_pitcher_chase) else np.nan))
    return pd.DataFrame(results)

def outcome_lift_next_pitch(df, pitcher, pairs_tbl, count_mode, sel_counts, sel_states, scope):
    sub = add_sequence_keys(df[df["Pitcher"] == pitcher].copy())
    sub = compute_outcome_flags(add_count_fields(sub)).sort_values(["_seq_group","_seq_order"])
    sub["PrevPitchType"] = sub.groupby("_seq_group")["TaggedPitchType"].shift(1)
    sub["Count_A"] = sub.groupby("_seq_group")["Count"].shift(1); sub["CountState_A"] = sub.groupby("_seq_group")["CountState"].shift(1)

    def mask_for(frame, at="B"):
        if count_mode == "Exact counts (B-S)":
            if not sel_counts: return np.ones(len(frame), dtype=bool)
            col = "Count_A" if at=="A" else "Count"; return frame[col].isin(sel_counts)
        elif count_mode == "Count state (Ahead/Even/Behind)":
            if not sel_states: return np.ones(len(frame), dtype=bool)
            col = "CountState_A" if at=="A" else "CountState"; return frame[col].isin(sel_states)
        return np.ones(len(frame), dtype=bool)

    if scope == "Count at A (prev pitch)":
        sub = sub[mask_for(sub, "A")]
    elif scope == "Count at B (current pitch)":
        sub = sub[mask_for(sub, "B")]
    else:
        sub = sub[mask_for(sub, "A") & mask_for(sub, "B")]

    base_pitcher_whiff = float(sub["Miss"].mean()) if len(sub) else np.nan
    base_pitcher_chase = float(sub["Chase"].mean()) if len(sub) else np.nan
    base_type = sub.groupby("TaggedPitchType").agg(base_Whiff=("Miss","mean"), base_Chase=("Chase","mean"))

    rows = []
    for _, r in pairs_tbl.iterrows():
        A, B = r["TypeA"], r["TypeB"]; cur = sub[(sub["PrevPitchType"] == A) & (sub["TaggedPitchType"] == B)]
        if len(cur) == 0:
            rows.append(dict(TypeA=A,TypeB=B,N=0,Whiff=np.nan,Chase=np.nan,Lift_vsType_Whiff=np.nan,Lift_vsType_Chase=np.nan,Lift_vsPitcher_Whiff=np.nan,Lift_vsPitcher_Chase=np.nan)); continue
        whiff = float(cur["Miss"].mean()); chase = float(cur["Chase"].mean())
        base_w = float(base_type.loc[B,"base_Whiff"]) if B in base_type.index else np.nan
        base_c = float(base_type.loc[B,"base_Chase"]) if B in base_type.index else np.nan
        rows.append(dict(TypeA=A,TypeB=B,N=len(cur),Whiff=whiff,Chase=chase,
                         Lift_vsType_Whiff=whiff-base_w if pd.notna(base_w) else np.nan,
                         Lift_vsType_Chase=chase-base_c if pd.notna(base_c) else np.nan,
                         Lift_vsPitcher_Whiff=whiff-base_pitcher_whiff if pd.notna(base_pitcher_whiff) else np.nan,
                         Lift_vsPitcher_Chase=chase-base_pitcher_chase if pd.notna(base_pitcher_chase) else np.nan))
    return pd.DataFrame(rows)

def smart_quantile_bins(series, max_bins=6, min_bins=4, min_count=12):
    s = pd.to_numeric(series, errors="coerce").dropna().values
    if s.size == 0: return None
    for k in range(max_bins, min_bins-1, -1):
        qs = np.linspace(0, 1, k+1); edges = np.unique(np.quantile(s, qs))
        if len(edges) - 1 < min_bins: continue
        cats = pd.cut(series, bins=edges, include_lowest=True, right=True, duplicates="drop")
        counts = cats.value_counts(dropna=False).sort_index()
        if (counts >= min_count).all(): return edges
    qs = np.linspace(0, 1, min_bins+1); edges = np.unique(np.quantile(s, qs))
    if len(edges) < 2:
        lo = float(np.nanmin(s)); hi = float(np.nanmax(s)); edges = np.array([lo, hi])
    return edges

def rates_by_bin(df_local, value_col, edges):
    cats = pd.cut(df_local[value_col], bins=edges, include_lowest=True, right=True, duplicates="drop")
    g = df_local.groupby(cats, observed=True).agg(Pitches=("PitchNo","count"), SwingPct=("Swing","mean"), WhiffPct=("Miss","mean"), ChasePct=("Chase","mean"), StrikePct=("EndsAsStrike","mean")).reset_index().rename(columns={value_col:"Bin"})
    for c in ["SwingPct","WhiffPct","ChasePct","StrikePct"]: g[c] = (g[c]*100).round(1)
    g["Bin"] = g["Bin"].astype(str); return g

st.title("Pitch Strike‑Look App — v5g10a")
uploaded = st.file_uploader("Upload a CSV (TrackMan-style columns)", type=["csv"])
if uploaded is None: st.info("Upload a CSV to begin. Required: " + ", ".join(REQUIRED_COLS)); st.stop()

df = pd.read_csv(uploaded); miss = [c for c in REQUIRED_COLS if c not in df.columns]
if miss: st.error("Missing required columns: " + ", ".join(miss)); st.stop()
if "sz_bot" not in df.columns: df["sz_bot"] = DEFAULT_SZ_BOT
if "sz_top" not in df.columns: df["sz_top"] = DEFAULT_SZ_TOP
df = df.dropna(subset=REQUIRED_COLS).copy(); df = add_count_fields(df)

sim_results = [simulate_pitch_segments(row) for _, row in df.iterrows()]
metrics = compute_metrics(df); metrics = add_outcome_columns(metrics)

tab1, tab2, tab3, tab4 = st.tabs(["🧭 3D Visualizer", "📊 Coach Dashboard", "📈 Correlations", "🧵 Tunneling"])

with tab1:
    st.subheader("3D Visualizer")
    pitchers = ["All"] + sorted(df["Pitcher"].dropna().unique().tolist())
    types = ["All"] + sorted(df["TaggedPitchType"].dropna().unique().tolist())
    colA, colB = st.columns(2)
    sel_pitcher = colA.selectbox("Pitcher", pitchers, index=0)
    sel_type = colB.selectbox("Pitch Type", types, index=0)
    mask = np.ones(len(df), dtype=bool)
    if sel_pitcher != "All": mask &= (df["Pitcher"].values == sel_pitcher)
    if sel_type != "All": mask &= (df["TaggedPitchType"].values == sel_type)
    fig = go.Figure()
    for ((_, row), sim, keep) in zip(df.iterrows(), sim_results, mask):
        if not keep: continue
        name_base = f"{row['Pitcher']} — #{int(row['PitchNo'])} — {row['TaggedPitchType']} — {row['RelSpeed']} mph"
        g0, g1 = sim["green_idx"]; gx, gy, gz = sim["x"][g0:g1], sim["y"][g0:g1], sim["z"][g0:g1]
        fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", line=dict(width=6, color=GREEN), name=name_base + " [GREEN]"))
        if sim["red_idx"] is not None:
            r0, r1 = sim["red_idx"]; rx, ry, rz = sim["x"][r0:r1], sim["y"][r0:r1], sim["z"][r0:r1]
            fig.add_trace(go.Scatter3d(x=rx, y=ry, z=rz, mode="lines", line=dict(width=6, color=RED), name=name_base + " [RED]"))
        if sim["diamond"] is not None:
            dmd = sim["diamond"]
            fig.add_trace(go.Scatter3d(x=[dmd["x"]], y=[dmd["y"]], z=[dmd["z"]], mode="markers", marker=dict(size=7, color=BLACK, symbol="diamond"), name=name_base + " [INFERENCE]"))
    fig.update_layout(height=700, scene=dict(xaxis_title="X (ft)", yaxis_title="Y (ft)", zaxis_title="Z (ft)", aspectmode="data"), legend=dict(itemsizing="trace", bgcolor="rgba(255,255,255,0.85)"))
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("Coach Dashboard")
    met_display = metrics.copy(); met_display["StrikeLookPct"] = (met_display["StrikeLookPct"] * 100).round(1)
    st.write("Per‑Pitch Metrics"); st.dataframe(met_display, width="stretch")

    st.markdown("### Strike‑Look Time — By Pitcher (All pitches)")
    by_pitcher_all = metrics.groupby("Pitcher").agg(
        Pitches=("PitchNo", "count"),
        BallsPct=("EndsAsStrike", lambda s: (1.0 - s.mean())*100.0),
        StrikePct=("EndsAsStrike", lambda s: (s.mean())*100.0),
        AvgStrikeLookPct=("StrikeLookPct", lambda s: s.mean()*100.0),
        InferenceTime_ms_mean=("InferenceTime_ms", "mean"),
        InferenceTime_ms_median=("InferenceTime_ms", "median"),
        BallsOnly_N=("InferenceTime_ms", lambda s: s.notna().sum())
    ).reset_index()
    st.dataframe(by_pitcher_all, width="stretch")

    st.markdown("### Strike‑Look Time — By Pitcher & Pitch Type (All pitches)")
    by_ptype_all = metrics.groupby(["Pitcher","TaggedPitchType"]).agg(
        N=("PitchNo","count"),
        BallsPct=("EndsAsStrike", lambda s: (1.0 - s.mean())*100.0),
        StrikePct=("EndsAsStrike", lambda s: (s.mean())*100.0),
        AvgStrikeLookPct=("StrikeLookPct", lambda s: s.mean()*100.0),
        InferenceTime_ms_mean=("InferenceTime_ms", "mean"),
        InferenceTime_ms_median=("InferenceTime_ms", "median"),
        BallsOnly_N=("InferenceTime_ms", lambda s: s.notna().sum())
    ).reset_index()
    st.dataframe(by_ptype_all, width="stretch")

    balls_only = metrics[~metrics["EndsAsStrike"]].copy()
    st.markdown("### Strike‑Look Time — By Pitcher (Balls only)")
    by_pitcher_balls = balls_only.groupby("Pitcher").agg(
        Balls_N=("PitchNo", "count"),
        InferenceTime_ms_mean=("InferenceTime_ms", "mean"),
        InferenceTime_ms_median=("InferenceTime_ms", "median"),
        StrikeLookPct_mean=("StrikeLookPct", lambda s: s.mean()*100.0)
    ).reset_index()
    st.dataframe(by_pitcher_balls, width="stretch")

    st.markdown("### Strike‑Look Time — By Pitcher & Pitch Type (Balls only)")
    by_ptype_balls = balls_only.groupby(["Pitcher","TaggedPitchType"]).agg(
        Balls_N=("PitchNo", "count"),
        InferenceTime_ms_mean=("InferenceTime_ms", "mean"),
        InferenceTime_ms_median=("InferenceTime_ms", "median"),
        StrikeLookPct_mean=("StrikeLookPct", lambda s: s.mean()*100.0)
    ).reset_index()
    st.dataframe(by_ptype_balls, width="stretch")

with tab3:
    st.subheader("Correlations")
    edges_sl = smart_quantile_bins(metrics["StrikeLookPct"], max_bins=6, min_bins=4, min_count=8)
    st.markdown("### Strike‑Look % buckets — dataset‑wide")
    if edges_sl is None: st.info("Not enough data to bin Strike‑Look %.")
    else: st.dataframe(rates_by_bin(metrics, "StrikeLookPct", edges_sl), width="stretch")

    st.markdown("### Inference Time (ms) buckets — dataset‑wide")
    met_it = metrics[metrics["InferenceTime_ms"].notna()].copy()
    edges_it = smart_quantile_bins(met_it["InferenceTime_ms"], max_bins=6, min_bins=4, min_count=8)
    if edges_it is None: st.info("Not enough data to bin Inference Time.")
    else: st.dataframe(rates_by_bin(met_it, "InferenceTime_ms", edges_it), width="stretch")

    st.markdown("---"); st.markdown("### Balls‑only: Correlation summary (pitcher‑level) & scatterplots")
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
        d = pitcher_agg[[x,y]].dropna()
        if len(d) < 3: pear = spear = np.nan
        else:
            pear = stats.pearsonr(d[x], d[y]).statistic; spear = stats.spearmanr(d[x], d[y]).correlation
        rows.append(dict(Metric=f"{x} vs {y}", Pearson=round(pear,3) if pd.notna(pear) else np.nan, Spearman=round(spear,3) if pd.notna(spear) else np.nan, N_pitchers=len(d)))
    st.dataframe(pd.DataFrame(rows), width="stretch")

    def scatter_with_fit_local(st_df, xcol, ycol, title):
        d = st_df[[xcol, ycol, "Pitches"]].dropna().copy()
        if len(d) < 2: st.info("Not enough data to draw: " + title); return
        x = d[xcol] * 100.0; y = d[ycol] * 100.0; sizes = 20 + 2 * np.sqrt(d["Pitches"].values)
        fig, ax = plt.subplots(); ax.scatter(x, y, s=sizes); m, b = np.polyfit(x, y, 1); xs = np.linspace(np.nanmin(x), np.nanmax(x), 100); ax.plot(xs, m*xs + b)
        ax.set_xlabel(xcol + " (%)"); ax.set_ylabel(ycol + " (%)"); ax.set_title(title); fig.tight_layout(); st.pyplot(fig)

    st.markdown("#### Scatterplots (balls‑only, pitcher‑level)")
    scatter_with_fit_local(pitcher_agg, "StrikeLookPct", "SwingPct", "Strike‑Look % vs Swing % (balls‑only)")
    scatter_with_fit_local(pitcher_agg, "StrikeLookPct", "WhiffPct", "Strike‑Look % vs Whiff % (balls‑only)")
    scatter_with_fit_local(pitcher_agg, "StrikeLookPct", "ChasePct", "Strike‑Look % vs Chase % (balls‑only)")

with tab4:
    st.subheader("Tunneling — within‑pitcher (fixed settings)")
    st.caption("Separation threshold fixed at 6 inches. Weights: Time=1.0, Distance=1.0, EntryGap=0.0, Release=1.0")

    pitchers_all = sorted(df["Pitcher"].dropna().unique().tolist())
    sel_pitcher = st.selectbox("Pitcher", pitchers_all, index=0, key="tun_pitcher")
    min_pitches = st.slider("Min pitches per type", 2, 10, 3, 1)

    # Count filters
    col_cf1, col_cf2 = st.columns(2)
    count_mode = col_cf1.selectbox("Filter mode", ["All counts","Exact counts (B‑S)","Count state (Ahead/Even/Behind)"], index=0)
    counts_available = sorted(metrics["Count"].dropna().unique().tolist())
    states_available = ["Ahead","Even","Behind"]
    sel_counts, sel_states = [], []
    if count_mode == "Exact counts (B‑S)":
        sel_counts = col_cf2.multiselect("Counts", counts_available, default=counts_available)
    elif count_mode == "Count state (Ahead/Even/Behind)":
        sel_states = col_cf2.multiselect("States", states_available, default=states_available)

    df_pool = df[df["Pitcher"] == sel_pitcher].copy()
    if count_mode == "Exact counts (B‑S)" and sel_counts: df_pool = df_pool[df_pool["Count"].isin(sel_counts)]
    elif count_mode == "Count state (Ahead/Even/Behind)" and sel_states: df_pool = df_pool[df_pool["CountState"].isin(sel_states)]

    def time_grid_local(): return np.arange(0.0, 0.6 + 1e-12, DT)
    def simulate_to_grid(row, t_grid):
        t = time_vector_to_back_of_plate(row); x, y, z = kinematics(row, t)
        X = np.full_like(t_grid, np.nan); Y = np.full_like(t_grid, np.nan); Z = np.full_like(t_grid, np.nan)
        if len(t) >= 2:
            mask = t_grid <= t[-1]; X[mask] = np.interp(t_grid[mask], t, x); Y[mask] = np.interp(t_grid[mask], t, y); Z[mask] = np.interp(t_grid[mask], t, z)
        return X, Y, Z

    def _stack_and_trim(arrs, min_frac=0.5):
        A = np.vstack(arrs); counts = np.sum(~np.isnan(A), axis=0); min_count = max(3, int(np.ceil(min_frac * A.shape[0])))
        valid_idx = np.where(counts >= min_count)[0]
        if len(valid_idx) == 0: return A[:, :1], 0
        last = valid_idx[-1]; return A[:, :last + 1], last

    def avg_trajectory_for_type(df_p, pitch_type, t_grid, min_pitches=3):
        sub = df_p[df_p["TaggedPitchType"] == pitch_type]
        if len(sub) < min_pitches: return None
        Xs, Ys, Zs = [], [], []
        for _, row in sub.iterrows():
            X, Y, Z = simulate_to_grid(row, t_grid); Xs.append(X); Ys.append(Y); Zs.append(Z)
        Xmat, _ = _stack_and_trim(Xs, min_frac=0.5); Ymat, _ = _stack_and_trim(Ys, min_frac=0.5); Zmat, _ = _stack_and_trim(Zs, min_frac=0.5)
        with np.errstate(invalid="ignore"):
            X_mean = np.nanmean(Xmat, axis=0); Y_mean = np.nanmean(Ymat, axis=0); Z_mean = np.nanmean(Zmat, axis=0)
        def sm(v):
            if np.isnan(v).all() or v.size < 7: return v
            k = np.ones(7)/7; out = np.convolve(v, k, mode="same"); out[:3] = v[:3]; out[-3:] = v[-3:]; return out
        X_mean, Y_mean, Z_mean = sm(X_mean), sm(Y_mean), sm(Z_mean)
        pad = len(time_grid_local()) - len(X_mean)
        if pad > 0:
            X_mean = np.concatenate([X_mean, np.full(pad, np.nan)])
            Y_mean = np.concatenate([Y_mean, np.full(pad, np.nan)])
            Z_mean = np.concatenate([Z_mean, np.full(pad, np.nan)])
        return X_mean, Y_mean, Z_mean, len(sub)

    def tunneling_between_types(avgA, avgB, t_grid, sep_thresh_ft=0.25):
        XA, YA, ZA, nA = avgA; XB, YB, ZB, nB = avgB
        valid = ~np.isnan(XA) & ~np.isnan(XB)
        if valid.sum() < 2: return None
        d = np.sqrt((XA - XB) ** 2 + (YA - YB) ** 2 + (ZA - ZB) ** 2); d[~valid] = np.nan
        idx = np.where((d > sep_thresh_ft) & valid)[0]
        if len(idx) > 0:
            i_sep = int(idx[0])
            distA = float(np.nansum(np.sqrt(np.diff(XA[:i_sep + 1])**2 + np.diff(YA[:i_sep + 1])**2 + np.diff(ZA[:i_sep + 1])**2)))
        else:
            i_sep = int(np.where(valid)[0][-1])
            distA = float(np.nansum(np.sqrt(np.diff(XA[:i_sep + 1])**2 + np.diff(YA[:i_sep + 1])**2 + np.diff(ZA[:i_sep + 1])**2)))
        i_last = int(np.where(valid)[0][-1])
        entry_gap = float(d[i_last]) if not np.isnan(d[i_last]) else float("nan")
        return dict(TunnelingTime_ms=float(t_grid[i_sep]*1000.0), TunnelingDist_ft=distA, EntryGap_ft=entry_gap, i_sep=i_sep, i_last=i_last, nA=int(nA), nB=int(nB))

    t_grid = time_grid_local(); df_pool = df_pool.copy()
    types_avail = sorted(df_pool["TaggedPitchType"].dropna().unique().tolist())
    avg_map = {pt: avg_trajectory_for_type(df_pool, pt, t_grid, min_pitches=min_pitches) for pt in types_avail}
    avg_map = {k:v for k,v in avg_map.items() if v is not None}
    type_mean_speed = df_pool.groupby("TaggedPitchType")["RelSpeed"].mean().to_dict()
    type_flight_time = {pt: (float(t_grid[int(np.where(~np.isnan(avg_map[pt][0]))[0][-1])]) if (~np.isnan(avg_map[pt][0])).any() else np.nan) for pt in avg_map}

    rows = []; sep_thresh_ft = SEP_THRESH_IN/12.0
    for a in types_avail:
        for b in types_avail:
            if a == b or a not in avg_map or b not in avg_map: continue
            met = tunneling_between_types(avg_map[a], avg_map[b], t_grid, sep_thresh_ft=sep_thresh_ft)
            if met is None: continue
            rows.append(dict(Pitcher=sel_pitcher, TypeA=a, TypeB=b, MeanVeloA=float(type_mean_speed.get(a, np.nan)), MeanVeloB=float(type_mean_speed.get(b, np.nan)), TflightA=float(type_flight_time.get(a, np.nan)), TflightB=float(type_flight_time.get(b, np.nan)), **met))
    tbl = pd.DataFrame(rows)

    rel_tbl = df_pool.groupby("TaggedPitchType").agg(N=("PitchNo","count"), x0_std=("x0","std"), z0_std=("z0","std")).reset_index()
    rel_tbl["xz_std"] = np.sqrt(rel_tbl["x0_std"].fillna(0)**2 + rel_tbl["z0_std"].fillna(0)**2)
    rel_tbl["ConsistencyScore"] = 1.0/(1.0 + rel_tbl["xz_std"])

    def add_tunneling_score(tbl, rel_tbl):
        base_cols = [] if tbl is None else list(tbl.columns)
        expected_cols = base_cols + ["time_norm","dist_norm","gap_norm","rel_norm","TunnelingScore"]
        if tbl is None or len(tbl) == 0:
            return pd.DataFrame(columns=expected_cols)

        out = tbl.copy()
        w_time, w_dist, w_gap, w_rel = float(W_TIME), float(W_DIST), float(W_GAP), float(W_REL)
        s = w_time + w_dist + w_gap + w_rel
        w_time, w_dist, w_gap, w_rel = w_time/s, w_dist/s, w_gap/s, w_rel/s

        tmax = float(np.nanmax(out["TunnelingTime_ms"])) if len(out) else np.nan
        if tmax > 0:
            abs_norm = out["TunnelingTime_ms"] / tmax
        else:
            abs_norm = pd.Series(0.0, index=out.index)
        shared_t = np.minimum(out.get("TflightA", np.nan), out.get("TflightB", np.nan))
        pct_of_flight = (out["TunnelingTime_ms"]/(shared_t*1000.0)).clip(0,1)
        pct_of_flight = pct_of_flight.fillna(0.0)

        alpha = 0.60
        t_blend = alpha*abs_norm.fillna(0.0) + (1-alpha)*pct_of_flight

        dv = (out.get("MeanVeloA", np.nan) - out.get("MeanVeloB", np.nan)).abs()
        damp = 1.0/(1.0 + 0.35*(dv/10.0))
        damp = damp.fillna(1.0)

        out["time_norm"] = (t_blend * damp).astype(float)

        dmax = float(np.nanmax(out["TunnelingDist_ft"])) if len(out) else np.nan
        out["dist_norm"] = out["TunnelingDist_ft"]/dmax if dmax>0 else 0.0

        cap_ft = 12.0/12.0
        out["gap_norm"] = 1.0 - np.clip(out["EntryGap_ft"], 0, cap_ft)/cap_ft

        rel_map = rel_tbl.set_index("TaggedPitchType")["ConsistencyScore"].to_dict()
        rel_vals = [float(np.nanmean([rel_map.get(a, np.nan), rel_map.get(b, np.nan)]))
                    for a,b in zip(out["TypeA"], out["TypeB"])]
        rel_series = pd.Series(rel_vals, index=out.index)
        if not np.isnan(rel_series).all():
            rel_series = rel_series.fillna(float(np.nanmean(rel_series)))
        else:
            rel_series = pd.Series(0.5, index=out.index)
        out["rel_norm"] = rel_series

        out["TunnelingScore"] = 100.0*(w_time*out["time_norm"] + w_dist*out["dist_norm"] + w_gap*out["gap_norm"] + w_rel*out["rel_norm"])

        for c in expected_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out[expected_cols]

    tbl_scored = add_tunneling_score(tbl, rel_tbl)

    if tbl_scored is None or len(tbl_scored) == 0 or "TunnelingScore" not in tbl_scored.columns:
        tbl_sorted = pd.DataFrame(columns=[
            "TypeA","TypeB","TunnelingScore","TunnelingTime_ms","TunnelingDist_ft",
            "EntryGap_ft","MeanVeloA","MeanVeloB","nA","nB","i_sep","i_last"
        ])
    else:
        sort_cols = [c for c in ["TunnelingScore","TunnelingTime_ms"] if c in tbl_scored.columns]
        tbl_sorted = tbl_scored.sort_values(sort_cols, ascending=[False]*len(sort_cols)).reset_index(drop=True)

    subtab1, subtab2, subtab3, subtab4 = st.tabs(["3D Tunnels + Explanations", "Release Consistency", "Outcome Lift (A→B)", "Outcome Lift (Same Pitch Back-to-Back)"])

    with subtab1:
        st.write("All tunneling pairs for this pitcher (higher score = better tunnel). Click column headers to sort.")
        if tbl_sorted.empty:
            st.info("No tunneling pairs found with the current filters. Try lowering the min-pitches threshold or widening count filters.")
        else:
            show_cols = ["TypeA","TypeB","TunnelingScore","TunnelingTime_ms","TunnelingDist_ft","EntryGap_ft","MeanVeloA","MeanVeloB","nA","nB"]
            tbl_show = tbl_sorted[show_cols].copy()
            for c in ["TunnelingScore","TunnelingTime_ms","TunnelingDist_ft","EntryGap_ft","MeanVeloA","MeanVeloB"]:
                tbl_show[c] = tbl_show[c].astype(float).round(2)
            st.dataframe(tbl_show, width="stretch")

            st.write("---")
            pairs = [f"{r.TypeA} → {r.TypeB}" for _, r in tbl_sorted.iterrows()]
            sel = st.selectbox("Tunneling pair to visualize", pairs, index=0, key="pair_vis")
            sel_row = tbl_sorted.iloc[pairs.index(sel)]; A, B = sel_row["TypeA"], sel_row["TypeB"]
            avgA, avgB = avg_map.get(A), avg_map.get(B)
            if avgA is None or avgB is None:
                st.info("Average trajectories unavailable for the selected pair.")
            else:
                XA, YA, ZA, nA = avgA; XB, YB, ZB, nB = avgB; i_sep = int(sel_row["i_sep"]); i_last = int(sel_row["i_last"])
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter3d(x=XA[:i_sep+1], y=YA[:i_sep+1], z=ZA[:i_sep+1], mode="lines", line=dict(width=6, color=GREEN), name=f"{A} — tunnel"))
                fig3.add_trace(go.Scatter3d(x=XA[i_sep:i_last+1], y=YA[i_sep:i_last+1], z=ZA[i_sep:i_last+1], mode="lines", line=dict(width=6, color=RED), name=f"{A} — diverge"))
                fig3.add_trace(go.Scatter3d(x=XB[:i_sep+1], y=YB[:i_sep+1], z=ZB[:i_sep+1], mode="lines", line=dict(width=6, color=GREEN), name=f"{B} — tunnel"))
                fig3.add_trace(go.Scatter3d(x=XB[i_sep:i_last+1], y=YB[i_sep:i_last+1], z=ZB[i_sep:i_last+1], mode="lines", line=dict(width=6, color=RED), name=f"{B} — diverge"))
                fig3.add_trace(go.Scatter3d(x=[XA[i_sep]], y=[YA[i_sep]], z=[ZA[i_sep]], mode="markers", marker=dict(size=8, color=BLACK, symbol="diamond"), name="Separation point"))
                fig3.update_layout(height=650, scene=dict(xaxis_title="X (ft)", yaxis_title="Y (ft)", zaxis_title="Z (ft)", aspectmode="data"), legend=dict(bgcolor="rgba(255,255,255,0.85)"))
                st.plotly_chart(fig3, width="stretch")

                w_sum = W_TIME + W_DIST + W_GAP + W_REL
                tmax = float(np.nanmax(tbl_scored["TunnelingTime_ms"])) if len(tbl_scored) else np.nan
                abs_norm = sel_row["TunnelingTime_ms"]/tmax if tmax>0 else 0.0
                shared_t = np.nanmin([sel_row.get("TflightA", np.nan), sel_row.get("TflightB", np.nan)])
                pct_of_flight = (sel_row["TunnelingTime_ms"]/(shared_t*1000.0)) if (shared_t and shared_t>0) else 0.0
                t_blend = 0.60*abs_norm + 0.40*pct_of_flight
                dv = abs(sel_row.get("MeanVeloA", np.nan) - sel_row.get("MeanVeloB", np.nan))
                damp = 1.0/(1.0 + 0.35*(dv/10.0)); time_norm = t_blend * damp
                dmax = float(np.nanmax(tbl_scored["TunnelingDist_ft"])) if len(tbl_scored) else np.nan
                dist_norm = sel_row["TunnelingDist_ft"]/dmax if dmax>0 else 0.0
                cap_ft = 12.0/12.0; gap_norm = 1.0 - min(max(sel_row["EntryGap_ft"], 0.0), cap_ft)/cap_ft
                rel_map = rel_tbl.set_index("TaggedPitchType")["ConsistencyScore"].to_dict()
                rel_pair = np.nanmean([rel_map.get(A, np.nan), rel_map.get(B, np.nan)])
                if np.isnan(rel_pair): rel_pair = float(np.nanmean(list(rel_map.values()))) if len(rel_map) else 0.5
                rel_norm = rel_pair
                contrib_time = (W_TIME/w_sum)*time_norm; contrib_dist = (W_DIST/w_sum)*dist_norm; contrib_gap = (W_GAP/w_sum)*gap_norm; contrib_rel = (W_REL/w_sum)*rel_norm
                total_score  = 100.0*(contrib_time + contrib_dist + contrib_gap + contrib_rel)
                st.markdown(f"**Explanation for {A} → {B}:**  \n- Tunneling time: **{sel_row['TunnelingTime_ms']:.0f} ms** (normalized={time_norm:.2f})  \n- Tunnel distance: **{sel_row['TunnelingDist_ft']:.2f} ft** (normalized={dist_norm:.2f})  \n- Entry gap at plate: **{sel_row['EntryGap_ft']:.2f} ft** (normalized={gap_norm:.2f})  \n- Release consistency (avg): **{rel_norm:.2f}**  \n- **Weights** — Time: {W_TIME:.2f}, Distance: {W_DIST:.2f}, EntryGap: {W_GAP:.2f}, Release: {W_REL:.2f}  \n- **TunnelingScore ≈ {total_score:.1f}**")

    with subtab2:
        st.write("Release point scatter by pitch type (x = release X; y = release Y).")
        sub = df_pool.copy()
        if sub.empty:
            st.info("No pitches after filters.")
        else:
            scatter_df = sub[["x0","z0","TaggedPitchType"]].rename(columns={"x0":"Release X (ft)","z0":"Release Y (ft)"}).dropna()
            if scatter_df.empty:
                st.info("No valid release coordinates available.")
            else:
                fig_rel = px.scatter(scatter_df, x="Release X (ft)", y="Release Y (ft)", color="TaggedPitchType", opacity=0.85, height=600)
                fig_rel.update_traces(marker=dict(size=9, line=dict(width=0.5, color="black")))
                fig_rel.update_layout(legend_title_text="Pitch Type", xaxis=dict(scaleanchor="y", scaleratio=1))
                st.plotly_chart(fig_rel, width="stretch")

    with subtab3:
        st.write("Next Pitch Outcome — A→B (count‑aware)")
        lift_scope = st.radio("Apply count filter to:", ["Count at A (prev pitch)", "Count at B (current pitch)", "Both A and B"], index=1, horizontal=True)
        if tbl_sorted.empty: st.info("No tunneling pairs available.")
        else:
            lift_next = outcome_lift_next_pitch(df, sel_pitcher, tbl_sorted, count_mode, sel_counts, sel_states, lift_scope)
            if lift_next.empty: st.info("No sequences found with current count filters.")
            else:
                plot = lift_next.copy().fillna(0.0); plot["label"] = plot["TypeA"].astype(str) + "→" + plot["TypeB"].astype(str)
                fig = go.Figure(); fig.add_trace(go.Bar(x=plot["label"], y=plot["Lift_vsType_Chase"] * 100.0, name="Lift vs Type (Chase)", marker_color=RED))
                fig.add_trace(go.Bar(x=plot["label"], y=plot["Lift_vsPitcher_Chase"] * 100.0, name="Lift vs Pitcher (Chase)", marker_color=GRAY))
                fig.update_layout(barmode="group", xaxis_title="Sequence A→B", yaxis_title="Chase Lift (pp)", height=500); st.plotly_chart(fig, width="stretch")

    with subtab4:
        st.write("Same Pitch Back‑to‑Back (A==B) — count‑aware")
        lift_scope_same = st.radio("Apply count filter to:", ["Count at A (prev pitch)", "Count at B (current pitch)", "Both A and B"], index=1, horizontal=True, key="same_scope")
        lift_same = outcome_lift_same_pitch(df, sel_pitcher, count_mode, sel_counts, sel_states, lift_scope_same)
        if lift_same.empty: st.info("No back‑to‑back sequences found (A==B).")
        else:
            plot = lift_same.copy().fillna(0.0); plot["label"] = plot["Type"].astype(str)
            plot["L_vP_Whiff_pp"] = plot["Lift_vsPitcher_Whiff"] * 100.0; plot["L_vP_Chase_pp"] = plot["Lift_vsPitcher_Chase"] * 100.0
            hover_texts = [f"<b>{r['Type']} → {r['Type']}</b><br>Whiff% after back-to-back: {r['Whiff']*100:.1f}%<br>Chase% after back-to-back: {r['Chase']*100:.1f}%<br><br><b>Lift vs Pitcher (Whiff)</b>: {r['L_vP_Whiff_pp']:+.1f} pp<br><b>Lift vs Pitcher (Chase)</b>: {r['L_vP_Chase_pp']:+.1f} pp" for _, r in plot.iterrows()]
            fig_same = go.Figure()
            fig_same.add_trace(go.Bar(x=plot["label"], y=plot["L_vP_Chase_pp"], name="Lift vs Pitcher (Chase)", marker_color=RED, hovertext=hover_texts, hovertemplate="%{hovertext}<extra></extra>"))
            fig_same.add_trace(go.Bar(x=plot["label"], y=plot["L_vP_Whiff_pp"], name="Lift vs Pitcher (Whiff)", marker_color=GRAY, hovertext=hover_texts, hovertemplate="%{hovertext}<extra></extra>"))
            fig_same.update_layout(barmode="group", xaxis_title="Back‑to‑Back Pitch Type (A==B)", yaxis_title="Lift vs Pitcher (percentage points)", height=500, legend=dict(bgcolor="rgba(255,255,255,0.85)"))
            st.plotly_chart(fig_same, width="stretch")
