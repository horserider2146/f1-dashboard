"""
Page 4 — Telemetry
Shows: speed trace and telemetry interpretation.
"""
import streamlit as st
import pandas as pd
from dashboard import api_client as api
from dashboard.components.charts import speed_trace_chart


def render(year: int, gp: str):
    st.header("📡 Car Telemetry")

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
                    st.session_state["telem_trace"] = trace
                    fig = speed_trace_chart(trace, trace_driver)
                    st.plotly_chart(fig, width="stretch")

                    # Show raw data
                    with st.expander("Raw telemetry data"):
                        st.dataframe(pd.DataFrame(trace), width="stretch")
                else:
                    st.warning("No speed trace data available.")
            except Exception as e:
                st.error(f"Speed trace failed: {e}")

    st.divider()

    # ── DRS info ───────────────────────────────────────────────────────────────
    st.subheader("📶 DRS Activation Info")
    _trace = st.session_state.get("telem_trace")
    if _trace:
        df_t = pd.DataFrame(_trace)
        if "drs" in df_t.columns and "distance" in df_t.columns:
            drs_series = pd.to_numeric(df_t["drs"], errors="coerce")
            # FastF1 docs: actual open states are 10, 12 and 14
            drs_on = df_t[drs_series.isin([10, 12, 14])]
            total_dist = df_t["distance"].max() - df_t["distance"].min()
            frac = len(drs_on) / max(len(df_t), 1)
            drs_km = total_dist * frac / 1000

            col1, col2, col3 = st.columns(3)
            col1.metric("Active Rows", f"{len(drs_on)}")
            col2.metric("~DRS Distance", f"{drs_km:.2f} km")
            col3.metric("% of Lap", f"{frac * 100:.1f}%")
            st.caption("DRS zones highlighted in teal on the speed trace above.")
        else:
            st.info("No DRS data available in this telemetry package.")
    else:
        st.info("Load a speed trace above to view DRS activation data.")
