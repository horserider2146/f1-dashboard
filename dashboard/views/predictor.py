"""
Page 6 — Race Outcome Predictor
Trains the ML model on selected race data and shows predicted final positions.
"""
import streamlit as st
import pandas as pd
from dashboard import api_client as api


def render(year: int, gp: str):
    st.header("🤖 Race Outcome Predictor")
    st.markdown(
        """
        This page uses a **Random Forest** model trained on lap pace, tyre strategy,
        pit stop count, and grid position to predict each driver's final race position.

        **Step 1:** Train the model on the selected race.
        **Step 2:** View the predicted finishing order.
        """
    )

    st.divider()

    # ── Train ──────────────────────────────────────────────────────────────────
    st.subheader("⚙️ Train Model")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(
            f"Training on: **{gp} {year}**. "
            "This fits the model on all lap data from this race session."
        )
    with col2:
        train_btn = st.button("🏋️ Train Predictor", key="train_predictor_btn",
                              use_container_width=True)

    if train_btn:
        with st.spinner("Training model..."):
            try:
                result = api.train_predictor(year, gp)
                st.success(
                    f"✅ Model trained successfully on {result.get('race', gp)} "
                    f"{result.get('year', year)}!"
                )
                st.session_state["predictor_trained"] = f"{year}|{gp}"
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.divider()

    # ── Predict ────────────────────────────────────────────────────────────────
    st.subheader("🔮 Predicted Finishing Order")

    trained_key = st.session_state.get("predictor_trained")
    current_key = f"{year}|{gp}"

    if trained_key != current_key:
        st.warning(
            "Model not yet trained for this race. "
            "Press **Train Predictor** above first."
        )
    else:
        with st.spinner("Predicting outcome..."):
            try:
                predictions = api.predict_outcome(year, gp)
                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    pred_df = pred_df.sort_values("predicted_position").reset_index(drop=True)
                    pred_df.index = pred_df.index + 1

                    # Highlight podium
                    def highlight_podium(row):
                        if row["predicted_position"] == 1:
                            return ["background-color: #FFD700; color: black"] * len(row)
                        elif row["predicted_position"] == 2:
                            return ["background-color: #C0C0C0; color: black"] * len(row)
                        elif row["predicted_position"] == 3:
                            return ["background-color: #CD7F32; color: black"] * len(row)
                        return [""] * len(row)

                    styled = (
                        pred_df
                        .rename(columns={
                            "driver_id": "Driver",
                            "predicted_position": "Predicted Position",
                        })
                        .style.apply(highlight_podium, axis=1)
                    )
                    st.dataframe(styled, use_container_width=True)

                    # Podium card
                    top3 = pred_df.head(3)["driver_id"].tolist()
                    if len(top3) >= 3:
                        st.markdown("### 🏆 Predicted Podium")
                        c1, c2, c3 = st.columns(3)
                        c2.metric("🥇 Winner", top3[0])
                        c1.metric("🥈 P2", top3[1])
                        c3.metric("🥉 P3", top3[2] if len(top3) > 2 else "—")

                else:
                    st.info("No predictions returned.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.divider()

    st.subheader("ℹ️ How the Model Works")
    with st.expander("Model details"):
        st.markdown("""
        **Algorithm:** Random Forest Classifier

        **Features used:**
        - Average lap pace (median lap time per driver)
        - Number of pit stops
        - Tyre compound used most (encoded)
        - Best sector times

        **Target:** Final race position

        **Limitations:**
        - Trained on a single race — more data = better accuracy
        - Does not account for mechanical failures or incidents
        - Safety car periods may skew pace estimates
        """)
