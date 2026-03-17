"""
Page 4 — Telemetry & Track Map
Shows: speed trace, animated track map with multiple drivers.
"""
import streamlit as st
import pandas as pd
from dashboard import api_client as api
from dashboard.components.charts import speed_trace_chart, track_map, track_animation


def render(year: int, gp: str):
    st.header("📡 Car Telemetry & Track Map")

    # ── Load drivers ───────────────────────────────────────────────────────────
    try:
        drivers_data = api.get_drivers(year, gp)
        drivers = [d["driver_id"] for d in drivers_data] if drivers_data else []
        laps_data = api.get_laps(year, gp)
        available_laps = sorted({r["lap_number"] for r in laps_data
                                  if r.get("lap_number")})
    except Exception as e:
        st.error(f"Could not load session data: {e}")
        return

    if not drivers:
        st.warning("No drivers found for this race.")
        return

    # ── Speed trace ────────────────────────────────────────────────────────────
    st.subheader("🏎️ Speed Trace")
    col1, col2 = st.columns(2)
    with col1:
        trace_driver = st.selectbox("Driver", drivers, key="trace_driver")
    with col2:
        trace_lap = st.selectbox(
            "Lap",
            options=available_laps,
            index=min(9, len(available_laps) - 1),
            key="trace_lap",
        )

    if st.button("Load Speed Trace", key="speed_trace_btn"):
        with st.spinner(f"Fetching telemetry for {trace_driver} lap {trace_lap}..."):
            try:
                trace = api.get_speed_trace(year, gp, trace_driver, trace_lap)
                if trace:
                    fig = speed_trace_chart(trace, trace_driver)
                    st.plotly_chart(fig, use_container_width=True)

                    # Show raw data
                    with st.expander("Raw telemetry data"):
                        st.dataframe(pd.DataFrame(trace), use_container_width=True)
                else:
                    st.warning("No speed trace data available.")
            except Exception as e:
                st.error(f"Speed trace failed: {e}")

    st.divider()

    # ── Track map (circuit outline) ────────────────────────────────────────────
    st.subheader("🗺️ Circuit Track Map")
    col1, col2 = st.columns(2)
    with col1:
        map_driver = st.selectbox("Driver", drivers,
                                  index=0, key="map_driver")
    with col2:
        map_lap = st.selectbox(
            "Lap",
            options=available_laps,
            index=min(4, len(available_laps) - 1),
            key="map_lap",
        )

    if st.button("Draw Track Map", key="track_map_btn"):
        with st.spinner("Loading track coordinates..."):
            try:
                pts = api.get_track_map(year, gp, map_driver, map_lap)
                if pts:
                    fig = track_map(pts, title=f"{gp} {year} — Circuit Layout")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No track coordinate data available.")
            except Exception as e:
                st.error(f"Track map failed: {e}")

    st.divider()

    # ── Multi-driver track animation ───────────────────────────────────────────
    st.subheader("🎬 Live Track Animation (Multi-Driver)")
    st.caption(
        "Watch multiple drivers move around the circuit on the same lap. "
        "Select up to 6 drivers for best performance."
    )

    selected_drivers = st.multiselect(
        "Select drivers",
        options=drivers,
        default=drivers[:4],
        max_selections=6,
        key="anim_drivers",
    )
    anim_lap = st.selectbox(
        "Lap",
        options=available_laps,
        index=min(19, len(available_laps) - 1),
        key="anim_lap",
    )

    if st.button("Generate Animation", key="anim_btn"):
        if not selected_drivers:
            st.warning("Select at least one driver.")
        else:
            with st.spinner("Building track animation — this may take a moment..."):
                try:
                    driver_str = ",".join(selected_drivers)
                    data = api.get_track_animation(year, gp, anim_lap, driver_str)
                    if data:
                        fig = track_animation(data, lap=anim_lap)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(
                            "Tip: Press ▶ on the animation slider to watch the cars move."
                        )
                    else:
                        st.warning("No animation data returned. "
                                   "Try a different lap or drivers.")
                except Exception as e:
                    st.error(f"Animation failed: {e}")

    st.divider()

    # ── DRS info ───────────────────────────────────────────────────────────────
    st.subheader("📶 DRS Activation Info")
    st.info(
        "DRS (Drag Reduction System) data is embedded in telemetry. "
        "Load a speed trace above — when the DRS column shows values 10–14, "
        "the wing is open. Typically visible as speed spikes on straights."
    )
