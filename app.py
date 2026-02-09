import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import regime  # your existing file

st.set_page_config(
    page_title="Market Regime Radar",
    layout="centered"
)

st.title("📈 Market Regime Radar")
st.caption("Detects hidden market regimes using a Hidden Markov Model")

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Settings")

symbol = st.sidebar.text_input("Stock symbol", value="AAPL")
period = st.sidebar.selectbox(
    "Historical period",
    ["6mo", "1y", "2y", "5y"],
    index=2
)

n_regimes = st.sidebar.slider(
    "Number of regimes",
    min_value=2,
    max_value=5,
    value=3
)

run = st.sidebar.button("Run Analysis")

# ----------------------------
# Run pipeline
# ----------------------------
if run:
    with st.spinner("Fetching data & fitting model..."):
        prices = regime.load_price_data(symbol, period)
        returns = regime.compute_log_returns(prices)
        model = regime.fit_hmm(returns, n_regimes=n_regimes)

        q1 = regime.current_regime_info(model, returns)
        q2 = regime.regime_duration(model, returns)
        q3 = regime.regime_transition_info(model)

        regime_stats = regime.summarize_regimes(model)
        regime_labels = regime.label_regimes(regime_stats)

        current_label = regime_labels[q1["current_regime"]]


    st.success("Analysis complete")

    # ----------------------------
    # Q1 — Current regime
    # ----------------------------
    st.subheader("1️⃣ Current Market Regime")

    st.markdown(
        f"""
        **Current Market Regime: {current_label}**

        *(Internal ID: Regime {q1['current_regime']})*

        - Confidence: **{q1['confidence']:.2%}**
        - Average return: **{q1['mean_return']:.4f}**
        - Volatility (risk): **{q1['volatility']:.4f}**
        """
    )

    with st.expander("ℹ️ What do these regimes mean?"):
        st.markdown("""
            **Bullish / Growth**  
            Prices tend to rise steadily with controlled risk.

            **Calm / Sideways**  
            Market moves in a narrow range with low momentum.

            **High Volatility / Uncertain**  
            Large price swings with no clear direction.

            **Bearish / Turbulent**  
            Downward pressure combined with elevated risk.
        """)



    # ----------------------------
    # Q2 — Duration
    # ----------------------------
    st.subheader("2️⃣ Regime History")

    st.markdown(
        f"""
        - Previous regime: **{q2['previous_regime']}**
        - Current regime duration: **{q2['duration_steps']} time steps**

        This tells us **how long the market has behaved this way**.
        """
    )

    # ----------------------------
    # Q3 — Transitions
    # ----------------------------
    st.subheader("3️⃣ Regime Transitions")

    st.markdown("**Expected duration of each regime:**")
    for i, d in enumerate(q3["expected_durations"]):
        st.write(f"- Regime {i}: ~{d:.2f} steps")

    st.markdown("**Most likely next transitions:**")
    for t in q3["most_likely_transitions"]:
        st.write(
            f"{regime_labels[t['from']]} → {regime_labels[t['to']]} "
            f"(probability {t['probability']:.2%})"
        )

    # ----------------------------
    # Visualization
    # ----------------------------
    st.subheader("📊 Visualizations")

    states = model.predict(returns)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prices.index[-len(states):], prices.values[-len(states):])
    ax.scatter(
        prices.index[-len(states):],
        prices.values[-len(states):],
        c=states,
        s=10
    )

    ax.set_title(f"{symbol} Price with Regimes")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")

    st.pyplot(fig)

else:
    st.info("Select parameters and click **Run Analysis**")
