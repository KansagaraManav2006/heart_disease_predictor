# Project Behavioral Rules & Theme Constraints

## Theme Directive (MANDATORY & PERMANENT)
Always use the user-provided Warm Stone & Sturdy Crimson theme definition in `client/src/index.css` and `client/tailwind.config.js`. NEVER revert or change the theme to cyan, electric teal, or blue themes.

### Permanent CSS Theme Configuration:
- Light Base (`:root`): `--background: #faf7f5`, `--card: #faf7f5`, `--primary: #9b2c2c`, `--secondary: #fdf2d6`, `--accent: #fef3c7`, `--border: #f5e8d2`, `--font-sans: Poppins, sans-serif`.
- Dark Base (`.dark`): `--background: #1c1917`, `--card: #292524`, `--primary: #b91c1c`, `--secondary: #92400e`, `--accent: #b45309`, `--border: #44403c`, `--foreground: #f5f5f4`.
- All components MUST bind to semantic tokens: `bg-background`, `bg-card`, `bg-primary`, `text-primary-foreground`, `text-foreground`, `border-border`, `bg-accent`, `bg-secondary`, `ring-ring`.
