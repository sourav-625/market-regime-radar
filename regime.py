import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import yfinance as yf

def load_price_data(symbol="AAPL", period="2y"):
    data = yf.download(symbol, period=period)
    prices = data["Close"].dropna()
    return prices

def compute_log_returns(prices):
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns.values.reshape(-1, 1)

def fit_hmm(returns, n_regimes=3):
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="diag",
        n_iter=1000,
        random_state=42
    )
    model.fit(returns)
    return model

def current_regime_info(model, returns):
    regime_probs = model.predict_proba(returns)
    current_probs = regime_probs[-1]

    current_regime = np.argmax(current_probs)

    mu = model.means_[current_regime][0]
    sigma = np.sqrt(model.covars_[current_regime][0])

    return {
        "current_regime": int(current_regime),
        "confidence": float(current_probs[current_regime]),
        "mean_return": float(mu),
        "volatility": float(sigma)
    }

def regime_duration(model, returns):
    states = model.predict(returns)

    current_regime = states[-1]

    duration = 1
    for s in reversed(states[:-1]):
        if s == current_regime:
            duration += 1
        else:
            break

    previous_regime = states[-duration - 1] if duration < len(states) else None

    return {
        "current_regime": int(current_regime),
        "previous_regime": int(previous_regime) if previous_regime is not None else None,
        "duration_steps": duration
    }

def regime_transition_info(model):
    A = model.transmat_

    expected_durations = 1 / (1 - np.diag(A))

    transitions = []
    for i in range(A.shape[0]):
        next_regime = np.argmax(A[i] * (1 - np.eye(A.shape[0])[i]))
        transitions.append({
            "from": i,
            "to": int(next_regime),
            "probability": float(A[i][next_regime])
        })

    return {
        "transition_matrix": A,
        "expected_durations": expected_durations,
        "most_likely_transitions": transitions
    }

if __name__ == "__main__":
    prices = load_price_data("AAPL", "2y")
    returns = compute_log_returns(prices)

    model = fit_hmm(returns, n_regimes=3)

    q1 = current_regime_info(model, returns)
    q2 = regime_duration(model, returns)
    q3 = regime_transition_info(model)

    print("Q1 — Current Regime Info")
    print(q1)

    print("\nQ2 — Regime Duration")
    print(q2)

    print("\nQ3 — Regime Transition Info")
    print(q3)
