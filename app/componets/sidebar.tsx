"use client";

import { useState } from "react";

export default function Sidebar() {
    const [open, setOpen] = useState(true);
    const [range, setRange] = useState(50);

    return (
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
    );
}