"""
F1 Analytics Dashboard — Main Streamlit entry point.

Run with:
    streamlit run dashboard/app.py
"""
import streamlit as st

from dashboard.views import (
    race_overview,
    lap_analysis,
    strategy,
    telemetry,
    live_race,
    predictor,
    stats_analysis,
)
from dashboard import api_client as api

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Analytics Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark sidebar */
    [data-testid="stSidebar"] {
        background-color: #111111;
    }
    /* Red accent headings */
    h1, h2, h3 { color: #E8002D; }
    /* Metric labels */
    [data-testid="stMetricLabel"] { font-size: 0.75rem; color: #999; }
    /* Divider colour */
    hr { border-color: #333; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar — Race selector ────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/"
        "250px-F1.svg.png",
        width=120,
    )
    st.title("F1 Analytics")
    st.markdown("---")

    # Season
    CURRENT_YEAR = 2025
    year = st.selectbox(
        "Season",
        options=list(range(CURRENT_YEAR, 2017, -1)),
        index=0,
        key="sidebar_year",
    )

    # Grand Prix schedule
    st.markdown("**Grand Prix**")
    with st.spinner("Loading calendar..."):
        try:
            schedule = api.get_schedule(year)
            gp_options = {
                f"R{e['round']} — {e['event_name']}": e["event_name"]
                for e in schedule
            }
        except Exception:
            # Fallback list if API unreachable
            gp_options = {
                "R1 — Bahrain Grand Prix": "Bahrain Grand Prix",
                "R5 — Monaco Grand Prix": "Monaco Grand Prix",
                "R10 — British Grand Prix": "British Grand Prix",
            }

    selected_label = st.selectbox("Race", options=list(gp_options.keys()),
                                  key="sidebar_gp")
    gp = gp_options[selected_label]

    st.markdown("---")

    # Page navigation
    PAGE_ICONS = {
        "🏁 Race Overview":      "race_overview",
        "⏱️ Lap Analysis":       "lap_analysis",
        "🔧 Tyre Strategy":      "strategy",
        "📡 Telemetry & Track":  "telemetry",
        "🔴 Live Race":          "live_race",
        "🤖 Race Predictor":     "predictor",
        "📊 Statistical Analysis": "stats_analysis",
    }
    page_label = st.radio("Navigation", list(PAGE_ICONS.keys()),
                          key="sidebar_page")
    page_key = PAGE_ICONS[page_label]

    st.markdown("---")
    st.caption(f"Selected: **{gp}** ({year})")
    st.caption("Data: FastF1 · OpenF1 · Jolpica")

# ── Route to selected page ─────────────────────────────────────────────────────
if page_key == "race_overview":
    race_overview.render(year, gp)

elif page_key == "lap_analysis":
    lap_analysis.render(year, gp)

elif page_key == "strategy":
    strategy.render(year, gp)

elif page_key == "telemetry":
    telemetry.render(year, gp)

elif page_key == "live_race":
    live_race.render()

elif page_key == "predictor":
    predictor.render(year, gp)

elif page_key == "stats_analysis":
    stats_analysis.render(year, gp)
