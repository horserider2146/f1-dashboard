"""
Page 2 — Lap Analysis
Shows: lap time comparison chart, lap delta between two drivers, safety car laps.
"""
import streamlit as st
import pandas as pd
from dashboard import api_client as api
from dashboard.components.charts import lap_time_chart, lap_delta_chart


def render(year: int, gp: str):
    st.header("⏱️ Lap Analysis")

    # ── Load lap data ──────────────────────────────────────────────────────────
    with st.spinner("Loading lap data..."):
        try:
            laps = api.get_laps(year, gp)
            drivers = sorted({r["driver_id"] for r in laps})
        except Exception as e:
            st.error(f"Failed to load laps: {e}")
            return

    # ── Lap time comparison ────────────────────────────────────────────────────
    st.subheader("📊 Lap Time Comparison")
    selected_drivers = st.multiselect(
        "Select drivers to compare",
        options=drivers,
        default=drivers[:5],
        key="lap_cmp_drivers",
    )
    if selected_drivers:
        fig = lap_time_chart(laps, drivers=selected_drivers)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Select at least one driver above.")

    st.divider()

    # ── Lap delta ──────────────────────────────────────────────────────────────
    st.subheader("🔀 Lap-by-Lap Time Delta")
    col1, col2 = st.columns(2)
    with col1:
        driver_a = st.selectbox("Driver A", options=drivers, index=0, key="delta_a")
    with col2:
        driver_b = st.selectbox("Driver B", options=drivers,
                                index=min(1, len(drivers) - 1), key="delta_b")

    if driver_a and driver_b and driver_a != driver_b:
        with st.spinner("Calculating delta..."):
            try:
                delta = api.get_lap_delta(year, gp, driver_a, driver_b)
                if delta:
                    fig = lap_delta_chart(delta, driver_a, driver_b)
                    st.plotly_chart(fig, width="stretch")

                    # Summary stats
                    df = pd.DataFrame(delta)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Delta (s)", f"{df['delta_s'].mean():.3f}")
                    col2.metric("Max Advantage", f"{df['delta_s'].max():.3f}")
                    col3.metric("Min Advantage", f"{df['delta_s'].min():.3f}")
                else:
                    st.info("No delta data available.")
            except Exception as e:
                st.error(f"Could not compute delta: {e}")
    elif driver_a == driver_b:
        st.warning("Please select two different drivers.")

    st.divider()

    # ── Safety car / VSC laps ──────────────────────────────────────────────────
    st.subheader("🚗 Safety Car & VSC Detection")
    with st.spinner("Detecting safety car laps..."):
        try:
            sc_laps = api.get_safety_car_laps(year, gp)
            if sc_laps:
                sc_df = pd.DataFrame(sc_laps)
                sc_candidates = sc_df[sc_df["sc_candidate"] == True]

                if not sc_candidates.empty:
                    st.warning(
                        f"⚠️ {len(sc_candidates)} potential Safety Car / VSC laps detected."
                    )
                    st.dataframe(
                        sc_candidates[["lap_number", "avg_lap_time", "median_lap_time"]]
                        .rename(columns={
                            "lap_number": "Lap",
                            "avg_lap_time": "Avg Lap Time (s)",
                            "median_lap_time": "Race Median (s)",
                        }),
                        width="stretch",
                    )
                else:
                    st.success("No Safety Car / VSC laps detected.")
        except Exception as e:
            st.error(f"Safety car detection failed: {e}")

    st.divider()

    # ── Raw lap data table ─────────────────────────────────────────────────────
    with st.expander("🔍 Raw Lap Data"):
        lap_df = pd.DataFrame(laps)
        driver_filter = st.selectbox("Filter by driver", ["All"] + drivers,
                                     key="raw_lap_filter")
        if driver_filter != "All":
            lap_df = lap_df[lap_df["driver_id"] == driver_filter]
        st.dataframe(lap_df, width="stretch")
