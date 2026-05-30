"use client";

import { useState } from "react";

export default function Home() {
  const [open, setOpen] = useState(true);
  const [range, setRange] = useState(50);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col dark:text-black">
      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 h-full w-64 bg-white shadow-lg p-6 transform transition-transform duration-300 z-30
        ${open ? "translate-x-0" : "-translate-x-full"}`}
      >
        <h3 className="font-bold text-lg mb-4">Sidebar</h3>

        <p className="text-gray-600 text-sm mb-4">
          This sidebar can contain navigation links, announcements,
          or other contextual information.
        </p>

        {/* Checkboxes */}
        <div className="mb-6">
          <p className="font-medium mb-2">Options</p>

          <label className="flex items-center gap-2 mb-1">
            <input type="checkbox" />
            Option 1
          </label>

          <label className="flex items-center gap-2 mb-1">
            <input type="checkbox" />
            Option 2
          </label>

          <label className="flex items-center gap-2">
            <input type="checkbox" />
            Option 3
          </label>
        </div>

        {/* Range Slider */}
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">
            Range: {range}
          </label>

          <input
            type="range"
            min="0"
            max="100"
            value={range}
            onChange={(e) => setRange(Number(e.target.value))}
            className="w-full"
          />
        </div>

        <button
          onClick={() => setOpen(false)}
          className="mt-6 px-3 py-2 bg-gray-200 rounded hover:bg-gray-300"
        >
          Close
        </button>
      </aside>
      <div
        className={`flex-1 transition-all duration-300 ml-64
        ${open ? "ml-64" : "ml-0"}`}
      >
        <div className="max-w-4xl mx-auto bg-white p-10 shadow-lg mt-10">

          {/* Title */}
          <h1 className="text-4xl font-bold text-center mb-2">
            Welcome to My Website
          </h1>

          {/* Description */}
          <p className="text-gray-500 text-center mb-6">
            This is a placeholder description explaining what this website
            is about. Replace it with something meaningful later.
          </p>

          <hr className="my-6" />

          {/* Section Heading */}
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-semibold">
              Featured Section
            </h2>

            <button
              onClick={() => setOpen(true)}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
            >
              Open Sidebar
            </button>
          </div>

          {/* Main Content */}
          <p className="text-gray-700">
            This is the main content area. For now it contains placeholder
            text, but later you could place articles, cards, or interactive
            elements here.
          </p>

          <button
            onClick={async () => {
              setLoading(true);

              try {
                const res = await fetch("/api/regime");
                const data = await res.json();

                setResult(data);
              } catch (err) {
                console.error(err);
              }

              setLoading(false);
            }
            }
            disabled={loading}
            className="mt-4 px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            {loading ? "Running Analysis..." : "Run Analysis"}
          </button>
          {result && result.q1 && result.q2 && result.q3 && (
            <div className="mt-10 space-y-8">

              {/* Current Regime */}
              <div>
                <h2 className="text-xl font-bold mb-2">Current Market Regime</h2>

                <p>
                  <b>{result.current_label}</b>
                </p>

                <p>Confidence: {result?.q1?.confidence}</p>
                <p>Average return: {result?.q1?.mean_return}</p>
                <p>Volatility: {result?.q1?.volatility}</p>
              </div>

              {/* Regime History */}
              <div>
                <h2 className="text-xl font-bold mb-2">Regime History</h2>

                <p>Previous regime: {result?.q2?.previous_regime}</p>
                <p>Duration: {result?.q2?.duration_steps}</p>
              </div>

              {/* Transitions */}
              <div>
                <h2 className="text-xl font-bold mb-2">Regime Transitions</h2>

                {result?.q3?.expected_durations.map((d: number, i: number) => (
                  <p key={i}>
                    Regime {i}: {d}
                  </p>
                ))}
              </div>

              {/* Chart */}
              <div>
                <h2 className="text-xl font-bold mb-2">Visualization</h2>

                <img
                  src={`data:image/png;base64,${result?.chart}`}
                  alt="chart"
                />
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}