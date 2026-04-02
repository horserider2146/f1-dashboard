"""
Reusable Plotly chart builders used across all dashboard pages.
Each function accepts plain Python lists/dicts and returns a Plotly Figure.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional

# Team colour palette fallback
COMPOUND_COLOURS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD": "#EBEBEB",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
    "UNKNOWN": "#888888",
}


# ── Lap time chart ─────────────────────────────────────────────────────────────

def lap_time_chart(laps: list, drivers: list | None = None) -> go.Figure:
    """Line chart of lap times across the race for each driver."""
    df = pd.DataFrame(laps)
    if df.empty or "lap_time_s" not in df.columns:
        return go.Figure()

    df = df.dropna(subset=["lap_time_s"])
    if drivers:
        df = df[df["driver_id"].isin(drivers)]

    fig = px.line(
        df,
        x="lap_number",
        y="lap_time_s",
        color="driver_id",
        title="Lap Time Comparison",
        labels={"lap_number": "Lap", "lap_time_s": "Lap Time (s)", "driver_id": "Driver"},
        template="plotly_dark",
    )
    fig.update_traces(mode="lines+markers", marker=dict(size=4))
    fig.update_layout(legend_title="Driver", hovermode="x unified")
    return fig


# ── Position change chart ──────────────────────────────────────────────────────

def position_change_chart(position_history: dict) -> go.Figure:
    """
    Line chart tracking race position over laps for all drivers.
    position_history: { driver_id: { lap: position } }
    """
    fig = go.Figure()
    for driver, lap_pos in position_history.items():
        laps = sorted(lap_pos.keys(), key=int)
        positions = [lap_pos[l] for l in laps]
        fig.add_trace(go.Scatter(
            x=[int(l) for l in laps],
            y=positions,
            mode="lines",
            name=driver,
            line=dict(width=2),
        ))

    fig.update_layout(
        title="Race Position History",
        xaxis_title="Lap",
        yaxis_title="Position",
        yaxis=dict(autorange="reversed", dtick=1),
        template="plotly_dark",
        hovermode="x unified",
        legend_title="Driver",
    )
    return fig


# ── Tyre strategy bar chart ────────────────────────────────────────────────────

def tyre_strategy_chart(stints: list) -> go.Figure:
    """
    Horizontal stacked bar chart showing each driver's tyre stints.
    """
    if not stints:
        return go.Figure()

    df = pd.DataFrame(stints)
    fig = go.Figure()

    drivers = df["driver_id"].unique().tolist()

    for compound, grp in df.groupby("compound"):
        colour = COMPOUND_COLOURS.get(compound.upper(), "#888888")
        fig.add_trace(go.Bar(
            name=compound,
            x=grp["laps_on_tyre"],
            y=grp["driver_id"],
            base=grp["start_lap"] - 1,
            orientation="h",
            marker_color=colour,
            text=compound,
            textposition="inside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"Compound: {compound}<br>"
                "Start lap: %{base}<br>"
                "Laps on tyre: %{x}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="stack",
        title="Tyre Strategy",
        xaxis_title="Lap",
        yaxis_title="Driver",
        template="plotly_dark",
        legend_title="Compound",
    )
    return fig


# ── Pit stop timeline ──────────────────────────────────────────────────────────

def pit_stop_timeline(pit_stops: list) -> go.Figure:
    """Scatter plot showing pit stop laps per driver."""
    if not pit_stops:
        return go.Figure()

    df = pd.DataFrame(pit_stops)
    fig = px.scatter(
        df,
        x="lap",
        y="driver_id",
        color="new_compound",
        symbol="stop_number",
        title="Pit Stop Timeline",
        labels={"lap": "Lap", "driver_id": "Driver", "new_compound": "New Tyre"},
        template="plotly_dark",
        color_discrete_map=COMPOUND_COLOURS,
        size_max=12,
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(legend_title="New Compound", hovermode="closest")
    return fig


# ── Track map ─────────────────────────────────────────────────────────────────

def track_map(track_points: list, title: str = "Track Map") -> go.Figure:
    """
    Draw the circuit outline from a single driver's X/Y telemetry.
    """
    if not track_points:
        return go.Figure()

    df = pd.DataFrame(track_points)
    fig = px.line(
        df, x="x", y="y",
        title=title,
        template="plotly_dark",
    )
    fig.update_traces(line=dict(color="white", width=1))
    fig.update_layout(
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def track_animation(driver_data: dict, lap: int,
                    circuit_points: list | None = None,
                    downsample: int = 8,
                    speed_factor: float = 1.0) -> go.Figure:
    """
    Animated scatter plot of multiple drivers moving around the track.

    driver_data:     { driver_id: [{ts, x, y}, ...] }
    circuit_points:  [{x, y}] — static circuit outline (one driver's full lap)
    downsample:      keep every Nth telemetry point to limit frame count

    Builds the figure manually (without px.scatter) so the circuit outline
    trace at index 0 is never referenced by any animation frame and therefore
    stays visible throughout the entire animation.
    """
    if not driver_data:
        return go.Figure()

    # ── Reference timestamps (downsampled from first driver) ───────────────────
    ref_driver = next(iter(driver_data))
    ref_ts = sorted({p["ts"] for p in driver_data[ref_driver]})[::downsample]

    drivers = list(driver_data.keys())

    # Pre-sort each driver's telemetry by ts for fast nearest-point lookup
    driver_lookup: dict = {}
    for drv, pts in driver_data.items():
        ts_sorted = sorted(pts, key=lambda p: p["ts"])
        driver_lookup[drv] = {
            "ts_vals": [p["ts"] for p in ts_sorted],
            "pts": ts_sorted,
        }

    def _nearest(drv: str, ts: float):
        info = driver_lookup[drv]
        ts_vals = info["ts_vals"]
        idx = min(range(len(ts_vals)), key=lambda i: abs(ts_vals[i] - ts))
        return info["pts"][idx]

    # Driver colour palette
    COLOURS = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]
    drv_colour = {drv: COLOURS[i % len(COLOURS)] for i, drv in enumerate(drivers)}

    # ── Build figure manually ─────────────────────────────────────────────────
    fig = go.Figure()

    # Trace 0 — circuit outline (NEVER updated by any frame)
    if circuit_points:
        cp = pd.DataFrame(circuit_points)
        fig.add_trace(go.Scatter(
            x=cp["x"], y=cp["y"],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.5)", width=10),
            name="Circuit",
            hoverinfo="skip",
            showlegend=False,
        ))
        circuit_trace_count = 1
    else:
        circuit_trace_count = 0

    # Traces 1..N — one marker per driver (initial position at first timestamp)
    first_ts = ref_ts[0]
    for drv in drivers:
        p0 = _nearest(drv, first_ts)
        fig.add_trace(go.Scatter(
            x=[p0["x"]], y=[p0["y"]],
            mode="markers",
            name=drv,
            marker=dict(size=14, color=drv_colour[drv],
                        line=dict(width=1.5, color="white")),
        ))

    driver_trace_indices = list(range(circuit_trace_count,
                                     circuit_trace_count + len(drivers)))

    # ── Compute frame duration to match real race time, scaled by speed_factor.
    # FastF1 telemetry can be at up to 240 Hz, so raw gaps can be ~4 ms.
    # We enforce a minimum of 100 ms (10 fps cap) so browsers render smoothly.
    # speed_factor > 1  → faster than real time (e.g. 5.0 = 5× speed)
    # speed_factor < 1  → slower than real time (e.g. 0.5 = half speed)
    if len(ref_ts) > 1:
        gaps = [ref_ts[i + 1] - ref_ts[i] for i in range(len(ref_ts) - 1)]
        avg_gap_s = sum(gaps) / len(gaps)
        frame_duration_ms = max(100, int(avg_gap_s * 1000 / speed_factor))
    else:
        frame_duration_ms = 800

    # ── Animation frames — only update driver traces ───────────────────────────
    frames = []
    for ts in ref_ts:
        frame_data = []
        for drv in drivers:
            p = _nearest(drv, ts)
            frame_data.append(go.Scatter(x=[p["x"]], y=[p["y"]]))
        frames.append(go.Frame(
            data=frame_data,
            traces=driver_trace_indices,
            name=str(round(ts, 2)),
        ))
    fig.frames = frames

    # ── Slider steps ──────────────────────────────────────────────────────────
    slider_steps = [
        dict(
            method="animate",
            args=[[str(round(ts, 2))],
                  {"frame": {"duration": frame_duration_ms, "redraw": False},
                   "mode": "immediate",
                   "transition": {"duration": 0}}],
            label=str(round(ts, 2)),
        )
        for ts in ref_ts
    ]

    # Compute axis range from circuit outline (or all driver positions as fallback)
    # so the view never auto-scales to just the driver markers during animation.
    if circuit_points:
        cp_x = [p["x"] for p in circuit_points]
        cp_y = [p["y"] for p in circuit_points]
    else:
        cp_x = [p["x"] for pts in driver_data.values() for p in pts]
        cp_y = [p["y"] for pts in driver_data.values() for p in pts]

    x_pad = (max(cp_x) - min(cp_x)) * 0.05
    y_pad = (max(cp_y) - min(cp_y)) * 0.05
    x_range = [min(cp_x) - x_pad, max(cp_x) + x_pad]
    y_range = [min(cp_y) - y_pad, max(cp_y) + y_pad]

    fig.update_layout(
        title=f"Lap {lap} — Driver Positions",
        xaxis=dict(visible=False, range=x_range, fixedrange=True),
        yaxis=dict(visible=False, range=y_range, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=40, b=80),
        legend_title="Driver",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0, x=0.1, xanchor="right", yanchor="top",
            buttons=[
                dict(label="▶",
                     method="animate",
                     args=[None, {"frame": {"duration": frame_duration_ms, "redraw": False},
                                  "fromcurrent": True,
                                  "transition": {"duration": 0}}]),
                dict(label="■",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.1, y=0,
            len=0.9,
            xanchor="left", yanchor="top",
            currentvalue=dict(prefix="frame_ts=", visible=True, xanchor="right"),
        )],
    )
    return fig


# ── Speed trace ───────────────────────────────────────────────────────────────

def speed_trace_chart(trace_data: list, driver: str) -> go.Figure:
    """Speed, throttle, brake and gear vs distance for one lap."""
    if not trace_data:
        return go.Figure()

    df = pd.DataFrame(trace_data)
    fig = go.Figure()

    if "speed" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["distance"], y=df["speed"],
            name="Speed (km/h)", line=dict(color="#E8002D", width=2)))

    if "throttle" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["distance"], y=df["throttle"],
            name="Throttle (%)", line=dict(color="#00D2BE", width=1.5),
            yaxis="y2"))

    # Shade DRS active zones based on FastF1 open codes (10, 12, 14)
    if "drs" in df.columns and "distance" in df.columns:
        drs_series = pd.to_numeric(df["drs"], errors="coerce")
        drs_active = drs_series.isin([10, 12, 14])
        in_zone, zone_start = False, None
        for dist, active in zip(df["distance"], drs_active):
            if active and not in_zone:
                zone_start = dist
                in_zone = True
            elif not active and in_zone:
                fig.add_vrect(x0=zone_start, x1=dist,
                              fillcolor="rgba(0, 210, 190, 0.2)", line_width=0,
                              annotation_text="DRS", annotation_position="top left",
                              annotation_font_size=9)
                in_zone = False
        if in_zone and zone_start is not None:
            fig.add_vrect(x0=zone_start, x1=float(df["distance"].iloc[-1]),
                          fillcolor="rgba(0, 210, 190, 0.2)", line_width=0)

    fig.update_layout(
        title=f"Speed Trace — {driver}",
        xaxis_title="Distance (m)",
        yaxis=dict(title="Speed (km/h)", side="left"),
        yaxis2=dict(title="Throttle (%)", overlaying="y", side="right",
                    range=[0, 110]),
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Tyre degradation prediction ───────────────────────────────────────────────

def tyre_deg_chart(predictions: list, driver: str, compound: str) -> go.Figure:
    """Line chart of predicted lap times across a stint."""
    if not predictions:
        return go.Figure()

    df = pd.DataFrame(predictions)
    colour = COMPOUND_COLOURS.get(compound.upper(), "#888")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["lap_number"],
        y=df["predicted_lap_time"],
        mode="lines+markers",
        name=f"{driver} — {compound}",
        line=dict(color=colour, width=2),
        marker=dict(size=5),
    ))
    fig.update_layout(
        title=f"Tyre Degradation Prediction — {driver} ({compound})",
        xaxis_title="Lap Number",
        yaxis_title="Predicted Lap Time (s)",
        template="plotly_dark",
    )
    return fig


# ── Lap delta ─────────────────────────────────────────────────────────────────

def lap_delta_chart(delta_data: list, driver_a: str, driver_b: str) -> go.Figure:
    """Bar chart showing lap-by-lap time delta between two drivers."""
    if not delta_data:
        return go.Figure()

    df = pd.DataFrame(delta_data)
    colours = ["#E8002D" if v > 0 else "#00D2BE" for v in df["delta_s"]]

    fig = go.Figure(go.Bar(
        x=df["lap_number"],
        y=df["delta_s"],
        marker_color=colours,
        name=f"{driver_a} vs {driver_b}",
        hovertemplate="Lap %{x}<br>Delta: %{y:.3f}s<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
    fig.update_layout(
        title=f"Lap Delta: {driver_a} vs {driver_b} (positive = {driver_a} slower)",
        xaxis_title="Lap",
        yaxis_title="Delta (s)",
        template="plotly_dark",
    )
    return fig
