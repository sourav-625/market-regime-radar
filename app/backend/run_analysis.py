import json
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import regime

def run(symbol="AAPL", period="2y", n_regimes=3):

    prices = regime.load_price_data(symbol, period)
    returns = regime.compute_log_returns(prices)

    model = regime.fit_hmm(returns, n_regimes=n_regimes)

    q1 = regime.current_regime_info(model, returns)
    q2 = regime.regime_duration(model, returns)
    q3 = regime.regime_transition_info(model)

    regime_stats = regime.summarize_regimes(model)
    regime_labels = regime.label_regimes(regime_stats)

    current_label = regime_labels[q1["current_regime"]]

    states = model.predict(returns)

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(prices.index[-len(states):], prices.values[-len(states):])
    ax.scatter(
        prices.index[-len(states):],
        prices.values[-len(states):],
        c=states,
        s=10
    )

    ax.set_title(f"{symbol} Price with Regimes")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "current_label": current_label,
        "q1": q1,
        "q2": q2,
        "q3": {
            "expected_durations": q3["expected_durations"].tolist(),
            "transitions": q3["most_likely_transitions"]
        },
        "chart": img_base64
    }

if __name__ == "__main__":
    result = run()
    print(json.dumps(result))