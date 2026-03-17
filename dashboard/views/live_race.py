"""
Page 5 — Live Race
Shows: real-time leaderboard, pit stops, track status.
Auto-refreshes every N seconds using st.rerun().
"""
import time
import streamlit as st
import pandas as pd
from dashboard import api_client as api


def render():
    st.header("🔴 Live Race Monitor")
    st.caption("Data sourced from OpenF1 API — updates every 5 seconds during a live session.")

    # ── Session key input ──────────────────────────────────────────────────────
    session_key = st.text_input(
        "OpenF1 Session Key (leave blank for latest)",
        value="",
        placeholder="e.g. 9158",
        key="live_session_key",
    )
    sk = session_key.strip() or None

    auto_refresh = st.checkbox("Auto-refresh every 5 seconds", value=False,
                               key="live_auto_refresh")

    if not sk:
        try:
            latest = api.get_latest_session()
            if latest:
                st.info(
                    f"Using latest session: **{latest.get('session_name', '?')}** — "
                    f"{latest.get('circuit_short_name', '?')} "
                    f"({latest.get('year', '?')})  |  "
                    f"Session key: `{latest.get('session_key', '?')}`"
                )
        except Exception:
            pass

    st.divider()

    # ── Track status banner ────────────────────────────────────────────────────
    try:
        track_statuses = api.get_track_status(session_key=sk)
        if track_statuses:
            latest_status = track_statuses[-1]
            status_msg = latest_status.get("message", "") or latest_status.get("flag", "")
            category = str(latest_status.get("category", "")).lower()
            flag = str(latest_status.get("flag", "")).lower()
            if "safety car" in category or "safety car" in status_msg.lower():
                emoji = "🔴"
            elif "vsc" in category or "virtual" in status_msg.lower():
                emoji = "🟠"
            elif "red" in flag:
                emoji = "🔴"
            elif "yellow" in flag:
                emoji = "🟡"
            elif "clear" in flag or "green" in flag:
                emoji = "🟢"
            else:
                emoji = "⚪"
            st.markdown(f"### Track Status: {emoji} {status_msg or 'All Clear'}")
        else:
            st.markdown("### Track Status: 🟢 All Clear")
    except Exception:
        st.markdown("### Track Status: ⚪ Unknown")

    st.divider()

    # ── Live leaderboard ───────────────────────────────────────────────────────
    st.subheader("🏆 Live Leaderboard")
    with st.spinner("Fetching live positions..."):
        try:
            leaderboard = api.get_live_leaderboard(session_key=sk)
            if leaderboard:
                lb_df = pd.DataFrame(leaderboard)

                def fmt_time(t):
                    if t is None:
                        return "—"
                    try:
                        t = float(t)
                        mins = int(t // 60)
                        secs = t % 60
                        return f"{mins}:{secs:06.3f}"
                    except Exception:
                        return str(t)

                if "lap_duration" in lb_df.columns:
                    lb_df["lap_duration"] = lb_df["lap_duration"].apply(fmt_time)

                rename = {
                    "position": "P", "driver_number": "Car #",
                    "lap_number": "Lap", "lap_duration": "Last Lap",
                    "compound": "Tyre", "gap_to_leader": "Gap",
                }
                lb_df = lb_df.rename(columns={k: v for k, v in rename.items() if k in lb_df.columns})
                st.dataframe(lb_df, use_container_width=True, hide_index=True)
            else:
                st.info("No live position data available. Active during live sessions only.")
        except Exception as e:
            st.error(f"Live leaderboard error: {e}")

    st.divider()

    # ── Live pit stops ─────────────────────────────────────────────────────────
    st.subheader("🛑 Recent Pit Stops")
    with st.spinner("Fetching pit stops..."):
        try:
            pit_stops = api.get_live_pit_stops(session_key=sk)
            if pit_stops:
                ps_df = pd.DataFrame(pit_stops)
                ps_df = ps_df.rename(columns={
                    "driver_number": "Car #", "lap_number": "Lap",
                    "pit_duration": "Duration (s)", "date": "Time",
                })
                st.dataframe(ps_df, use_container_width=True, hide_index=True)
            else:
                st.info("No pit stop data available yet.")
        except Exception as e:
            st.error(f"Pit stop error: {e}")

    st.divider()

    # ── Auto-refresh ───────────────────────────────────────────────────────────
    if auto_refresh:
        st.caption("⏳ Next refresh in 5 seconds...")
        time.sleep(5)
        st.rerun()
    else:
        if st.button("🔄 Refresh Now", key="live_refresh_btn"):
            st.rerun()
