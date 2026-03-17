"""
Page 3 — Tyre Strategy
Shows: tyre strategy bar chart, pit stop timeline, undercut detector,
       tyre degradation predictions.
"""
import streamlit as st
import pandas as pd
from dashboard import api_client as api
from dashboard.components.charts import (
    tyre_strategy_chart, pit_stop_timeline, tyre_deg_chart,
)

COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def render(year: int, gp: str):
    st.header("🔧 Tyre Strategy Analysis")

    # ── Strategy overview ──────────────────────────────────────────────────────
    st.subheader("📋 Strategy Comparison")
    with st.spinner("Loading strategies..."):
        try:
            strategies = api.get_strategy_comparison(year, gp)
            if strategies:
                st.dataframe(
                    pd.DataFrame(strategies).rename(
                        columns={"driver_id": "Driver",
                                 "strategy": "Strategy",
                                 "num_stops": "Stops"}
                    ),
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Could not load strategies: {e}")

    st.divider()

    # ── Tyre strategy chart ────────────────────────────────────────────────────
    st.subheader("🟡 Tyre Stint Visualisation")
    with st.spinner("Loading stints..."):
        try:
            stints = api.get_stints(year, gp)
            if stints:
                fig = tyre_strategy_chart(stints)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No stint data available.")
        except Exception as e:
            st.error(f"Stint chart failed: {e}")

    st.divider()

    # ── Pit stop timeline ──────────────────────────────────────────────────────
    st.subheader("🛑 Pit Stop Timeline")
    with st.spinner("Loading pit stops..."):
        try:
            pit_stops = api.get_pit_stops(year, gp)
            if pit_stops:
                fig = pit_stop_timeline(pit_stops)
                st.plotly_chart(fig, use_container_width=True)

                # Table
                with st.expander("Pit stop details"):
                    st.dataframe(
                        pd.DataFrame(pit_stops).rename(
                            columns={
                                "driver_id": "Driver",
                                "lap": "Lap",
                                "old_compound": "Out",
                                "new_compound": "In",
                                "stop_number": "Stop #",
                            }
                        ),
                        use_container_width=True,
                    )
            else:
                st.info("No pit stop data available.")
        except Exception as e:
            st.error(f"Pit stop timeline failed: {e}")

    st.divider()

    # ── Undercut detector ──────────────────────────────────────────────────────
    st.subheader("⚔️ Undercut / Overcut Detection")
    with st.spinner("Detecting undercuts..."):
        try:
            undercuts = api.get_undercuts(year, gp)
            if undercuts:
                uc_df = pd.DataFrame(undercuts)
                success_df = uc_df[uc_df["success"] == True]
                fail_df = uc_df[uc_df["success"] == False]

                col1, col2 = st.columns(2)
                col1.metric("✅ Successful Undercuts", len(success_df))
                col2.metric("❌ Failed Attempts", len(fail_df))

                st.dataframe(
                    uc_df.rename(columns={
                        "undercut_driver": "Undercut Driver",
                        "target_driver": "Target",
                        "pit_lap_undercut": "Pit Lap",
                        "pit_lap_target": "Rival Pit Lap",
                        "time_gain_s": "Time Gain (s)",
                        "success": "Success",
                    }),
                    use_container_width=True,
                )
            else:
                st.info("No undercut attempts detected.")
        except Exception as e:
            st.error(f"Undercut detection failed: {e}")

    st.divider()

    # ── Tyre degradation predictor ─────────────────────────────────────────────
    st.subheader("📉 Tyre Degradation Prediction")

    try:
        drivers_data = api.get_drivers(year, gp)
        drivers = [d["driver_id"] for d in drivers_data] if drivers_data else []
    except Exception:
        drivers = []

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pred_driver = st.selectbox("Driver", drivers or ["HAM"], key="deg_driver")
    with col2:
        pred_compound = st.selectbox("Compound", COMPOUNDS, key="deg_compound")
    with col3:
        pred_start = st.number_input("Stint Start Lap", min_value=1, value=1, key="deg_start")
    with col4:
        pred_length = st.number_input("Stint Length (laps)", min_value=5,
                                      max_value=50, value=20, key="deg_length")

    if st.button("Predict Degradation", key="deg_predict_btn"):
        with st.spinner("Running degradation model..."):
            try:
                preds = api.predict_tyre_deg(
                    year, gp, pred_driver, pred_compound,
                    int(pred_start), int(pred_length)
                )
                if preds:
                    fig = tyre_deg_chart(preds, pred_driver, pred_compound)
                    st.plotly_chart(fig, use_container_width=True)

                    # Optimal pit window
                    pit_window = api._get(
                        f"/analytics/{year}/{gp}/tyre-deg/optimal-pit-window",
                        params={
                            "driver": pred_driver,
                            "compound": pred_compound,
                            "current_tyre_age": 1,
                            "lap_number": int(pred_start),
                        }
                    )
                    laps_left = pit_window.get("laps_until_threshold")
                    recommendation = pit_window.get("recommendation", "")
                    st.info(
                        f"🔍 Optimal pit window: **{laps_left} laps** remaining "
                        f"before performance drops 1.5s. Recommendation: **{recommendation}**"
                    )
                else:
                    st.warning("No prediction data returned.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.divider()

    # ── Compound summary ───────────────────────────────────────────────────────
    st.subheader("🏎️ Compound Performance Summary")
    with st.spinner("Loading compound stats..."):
        try:
            compound_stats = api.get_compound_summary(year, gp)
            if compound_stats:
                st.dataframe(
                    pd.DataFrame(compound_stats).rename(columns={
                        "compound": "Compound",
                        "mean_lap_time_s": "Mean Lap Time (s)",
                        "deg_rate_s_per_lap": "Deg Rate (s/lap)",
                        "stint_count": "Drivers Used",
                    }),
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Compound summary failed: {e}")
