/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: { bg: "#0a0f1e", panel: "#101830", line: "rgba(148,163,184,.16)", ink: "#e6edf7", mute: "#8b9ab5",
                cyan: { DEFAULT: "#22d3ee" }, violet: { DEFAULT: "#a78bfa" }, pink: { DEFAULT: "#f472b6" }, good: "#34d399", warn: "#fbbf24", bad: "#fb7185" },
      fontFamily: { sans: ["Space Grotesk", "Inter", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "ui-monospace", "monospace"] },
      boxShadow: { glow: "0 0 40px rgba(34,211,238,.10)" },
    },
  },
  plugins: [],
};
