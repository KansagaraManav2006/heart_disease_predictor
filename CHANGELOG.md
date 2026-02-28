# Changelog

All notable changes to the Disease Prediction System project.

## [3.0.0] - 2026-02-28

### Changed - Architecture & Tech Stack

- **Modernized Fullstack Application**:
  - Replaced the legacy Python Streamlit frontend with a modern React 18 frontend powered by Vite.
  - Implemented a Glassmorphism UI using Tailwind CSS v4.
  - Created a dedicated Node.js/Express backend to serve RESTful API endpoints for predictions.
  - Removed `app.py` and Streamlit dependencies entirely.
  - Set up `concurrently` in the root `package.json` for seamless fullstack development.

## [2.0.0] - 2026-02-17

### Added - PDF Report Generation

- **PDF Report Functionality**: Implemented professional PDF report generation.
- **Patient Name Input**: Added patient name field to personalize reports.

### Added - Testing Infrastructure

- **PDF Report Tests**: Tests for PDF generation.

## [1.0.0] - Previous Version

### Features

- Diabetes & Heart disease risk prediction models using Scikit-learn.
- Modular Python architecture with `utils.py`.
