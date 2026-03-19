/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkBg: '#0f172a', /* Kept just in case, but background will be light */
        background: '#f5f7fa',
        card: '#ffffff',
        primary: '#1E88E5',
        primaryHover: '#1565C0',
        secondary: '#26A69A',
        danger: '#EF5350',
        textMain: '#1e293b',
        textMuted: '#64748b',
        borderLight: '#e2e8f0'
      }
    },
  },
  plugins: [],
}
