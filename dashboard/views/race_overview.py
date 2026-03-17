"""
Page 1 — Race Overview
Shows: live leaderboard, position history chart, fastest laps table.
"""
import streamlit as st
import pandas as pd
from dashboard import api_client as api
from dashboard.components.charts import position_change_chart, track_map, track_animation


def render(year: int, gp: str):
    st.header("🏁 Race Overview")

    # ── Race metadata ──────────────────────────────────────────────────────────
    with st.spinner("Loading race info..."):
        try:
            meta = api.get_race_metadata(year, gp)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Race", meta.get("race_name", gp))
            col2.metric("Track", meta.get("track", "—"))
            col3.metric("Season", meta.get("season", year))
            col4.metric("Total Laps", meta.get("total_laps", "—"))
        except Exception as e:
            st.warning(f"Could not load race metadata: {e}")

    st.divider()

    # ── Track map ──────────────────────────────────────────────────────────────
    st.subheader("🗺️ Circuit Track Map & Driver Animation")

    # Load drivers + lap list (needed for selectors)
    try:
        drivers_data = api.get_drivers(year, gp)
        all_driver_ids = [d["driver_id"] for d in drivers_data] if drivers_data else []
        laps_data = api.get_laps(year, gp)
        available_laps = sorted({r["lap_number"] for r in laps_data if r.get("lap_number")})
    except Exception as e:
        st.warning(f"Could not load driver/lap list: {e}")
        all_driver_ids, available_laps = [], []

    if all_driver_ids and available_laps:
        # ── View mode ─────────────────────────────────────────────────────────
        view_mode = st.radio(
            "View mode",
            options=["Single Lap", "Lap Range", "Full Race"],
            horizontal=True,
            key="overview_view_mode",
        )

        col_lap, col_drivers, col_speed = st.columns([2, 3, 2])

        with col_lap:
            mid = len(available_laps) // 2
            if view_mode == "Single Lap":
                map_lap = st.selectbox(
                    "Lap", options=available_laps,
                    index=mid, key="overview_map_lap",
                )
                map_lap_end = None
            elif view_mode == "Lap Range":
                map_lap = st.selectbox(
                    "Start lap", options=available_laps,
                    index=0, key="overview_map_lap",
                )
                map_lap_end = st.selectbox(
                    "End lap", options=available_laps,
                    index=min(9, len(available_laps) - 1),
                    key="overview_map_lap_end",
                )
            else:  # Full Race
                map_lap = available_laps[0]
                map_lap_end = -1
                st.info(f"Full race: laps {available_laps[0]}–{available_laps[-1]}")

        with col_drivers:
            sel_drivers = st.multiselect(
                "Drivers to animate (up to 6)",
                options=all_driver_ids,
                default=all_driver_ids[:6],
                max_selections=6,
                key="overview_map_drivers",
            )

        with col_speed:
            speed_label = st.select_slider(
                "Playback speed",
                options=["0.25×", "0.5×", "1× (real)", "2×", "5×", "10×"],
                value="1× (real)",
                key="overview_speed",
            )
            speed_map = {
                "0.25×": 0.25, "0.5×": 0.5, "1× (real)": 1.0,
                "2×": 2.0, "5×": 5.0, "10×": 10.0,
            }
            speed_factor = speed_map[speed_label]

        if view_mode == "Full Race":
            load_label = "🗺️ Load Full Race Animation (slow — loads all laps)"
        elif view_mode == "Lap Range":
            load_label = f"🗺️ Load Laps {map_lap}–{map_lap_end} Animation"
        else:
            load_label = "🗺️ Load Track Map & Animation"

        if st.button(load_label, key="overview_map_btn", use_container_width=True):
            if not sel_drivers:
                st.warning("Select at least one driver.")
            else:
                # Circuit outline — always use first driver's single reference lap
                ref_lap = available_laps[mid] if view_mode in ("Lap Range", "Full Race") else map_lap
                with st.spinner("Loading circuit outline..."):
                    circuit_pts = None
                    try:
                        circuit_pts = api.get_track_map(year, gp, sel_drivers[0], ref_lap)
                    except Exception as e:
                        st.warning(f"Circuit outline API failed ({e}), will use animation data as fallback.")

                if circuit_pts:
                    fig_outline = track_map(circuit_pts, title=f"{gp} {year} — Circuit Layout")
                    st.plotly_chart(fig_outline, use_container_width=True)

                # Animation data
                spinner_msg = {
                    "Single Lap": f"Loading telemetry for {len(sel_drivers)} drivers on lap {map_lap}…",
                    "Lap Range": f"Loading laps {map_lap}–{map_lap_end} for {len(sel_drivers)} drivers…",
                    "Full Race": f"Loading full race for {len(sel_drivers)} drivers — this may take 60+ s…",
                }[view_mode]

                with st.spinner(spinner_msg + " (first load may be slow)"):
                    try:
                        driver_str = ",".join(sel_drivers)
                        anim_data = api.get_track_animation(
                            year, gp, map_lap, driver_str, lap_end=map_lap_end
                        )
                        if anim_data:
                            if not circuit_pts:
                                first_drv = next(iter(anim_data))
                                circuit_pts = anim_data[first_drv]
                                fig_outline = track_map(
                                    circuit_pts, title=f"{gp} {year} — Circuit Layout"
                                )
                                st.plotly_chart(fig_outline, use_container_width=True)

                            anim_label = (
                                map_lap if view_mode == "Single Lap"
                                else f"{map_lap}–{map_lap_end if map_lap_end != -1 else available_laps[-1]}"
                            )
                            fig_anim = track_animation(
                                anim_data,
                                lap=anim_label,
                                circuit_points=circuit_pts,
                                speed_factor=speed_factor,
                            )
                            st.plotly_chart(fig_anim, use_container_width=True)
                            st.caption(
                                f"▶ Playback at {speed_label}. "
                                "Press play to watch drivers move around the circuit."
                            )
                        else:
                            st.info("No telemetry data returned.")
                    except Exception as e:
                        st.error(f"Animation failed: {e}")
    else:
        st.info("Select a race above to load the track map.")

    st.divider()

    # ── Fastest laps table ─────────────────────────────────────────────────────
    st.subheader("⚡ Fastest Laps")
    with st.spinner("Fetching fastest laps..."):
        try:
            fl = api.get_fastest_laps(year, gp)
            if fl:
                fl_df = pd.DataFrame(fl)
                fl_df["lap_time_s"] = fl_df["lap_time_s"].round(3)
                fl_df.index = range(1, len(fl_df) + 1)
                st.dataframe(
                    fl_df[["driver_id", "lap_number", "lap_time_s", "compound"]].rename(
                        columns={
                            "driver_id": "Driver",
                            "lap_number": "Lap",
                            "lap_time_s": "Time (s)",
                            "compound": "Compound",
                        }
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No fastest lap data available.")
        except Exception as e:
            st.error(f"Could not load fastest laps: {e}")

    st.divider()

    # ── Position history chart ─────────────────────────────────────────────────
    st.subheader("📈 Race Position History")
    with st.spinner("Loading position history..."):
        try:
            pos_hist = api.get_position_history(year, gp)
            if pos_hist:
                # Let user filter drivers
                all_drivers = sorted(pos_hist.keys())
                selected = st.multiselect(
                    "Filter drivers",
                    options=all_drivers,
                    default=all_drivers[:6],
                    key="pos_hist_filter",
                )
                filtered = {d: pos_hist[d] for d in selected if d in pos_hist}
                if filtered:
                    fig = position_change_chart(filtered)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Position data not available for this race.")
        except Exception as e:
            st.error(f"Could not load position history: {e}")

    st.divider()

    # ── Position changes summary ───────────────────────────────────────────────
    st.subheader("🔄 Net Position Changes")
    with st.spinner("Loading position changes..."):
        try:
            changes = api.get_overtakes(year, gp)
            if changes:
                # Aggregate total overtakes per driver
                ov_df = pd.DataFrame(changes)
                summary = (ov_df.groupby("driver_id")
                           .agg(overtakes=("positions_gained", "count"),
                                total_positions_gained=("positions_gained", "sum"))
                           .sort_values("overtakes", ascending=False)
                           .reset_index()
                           .rename(columns={"driver_id": "Driver",
                                            "overtakes": "Overtakes",
                                            "total_positions_gained": "Total Positions Gained"}))
                st.dataframe(summary, use_container_width=True)
            else:
                st.info("No overtake data available.")
        except Exception as e:
            st.error(f"Could not load overtake data: {e}")
