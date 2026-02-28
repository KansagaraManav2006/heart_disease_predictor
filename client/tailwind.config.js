/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: '#0f172a',
        darkCard: '#1e293b',
        primary: '#22c55e',
        primaryHover: '#16a34a',
        danger: '#ef4444',
        textMain: '#f8fafc',
        textMuted: '#cbd5e1'
      }
    },
  },
  plugins: [],
}
