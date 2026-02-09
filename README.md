# 📈 Market Regime Radar

Market Regime Radar is a **probabilistic market analysis tool** that detects
hidden market regimes using a **Hidden Markov Model (HMM)**.

Instead of predicting prices, it focuses on answering a more fundamental question:

> *“What kind of market are we in right now?”*

---

## 🚀 What This App Does

Given historical price data of a stock, the app:

1. **Identifies the current market regime**
   - e.g. Bullish, Calm, High Volatility, Bearish
2. **Explains regime characteristics**
   - average return
   - volatility (risk)
   - confidence level
3. **Shows regime history**
   - previous regime
   - how long the current regime has lasted
4. **Estimates regime transitions**
   - expected duration of regimes
   - most likely next regime
5. **Visualizes regimes on the price chart**

All results are presented in **plain English**, with no financial expertise required.

---

## 🧠 Core Idea

Markets do not behave the same way all the time.
They switch between different **behavioral phases (regimes)**.

This project models those phases as:
- **hidden states** (regimes)
- evolving over time via a **Markov process**
- generating observable returns with different statistical properties

The approach is inspired by techniques used in **quantitative finance**, but designed
to be transparent and interpretable.

---

## ⚠️ What This App Is NOT

- ❌ Not a trading bot
- ❌ Not a price prediction system
- ❌ Not financial advice

It is a **market state interpretation tool**, not a forecasting oracle.

---

## 🛠️ Tech Stack

- **Python**
- **Hidden Markov Models** (`hmmlearn`)
- **NumPy / Pandas**
- **matplotlib**
- **Streamlit**
- **Yahoo Finance** (for historical price data)

---

## 📊 How It Works (High Level)

1. Fetch historical closing prices
2. Convert prices to log-returns
3. Fit a Gaussian HMM with multiple regimes
4. Infer:
   - regime probabilities
   - regime statistics
   - transition dynamics
5. Translate regimes into human-readable labels

---

## ▶️ How to Run Locally

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🌐 Deployment

This app is deployed using Streamlit Cloud.

Simply connect the GitHub repository and select:

- app.py as the entry point
- requirements.txt for dependencies

## 📌 Future Improvements

- Regime stability smoothing

- Multi-asset comparison

- Online (streaming) inference

- Regime confidence over time

- API backend support

## 👤 Developer

Sourav pati - Student at Institue of Technical Education and Research (ITER), Bhubaneswar
github - [github.com/sourav-625](https://github.com/sourav-625)