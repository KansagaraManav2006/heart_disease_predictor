import React, { lazy, Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import PublicShell from './components/layout/PublicShell';
import WorkspaceShell from './components/layout/WorkspaceShell';
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider } from './context/AuthContext';
import Skeleton from './components/Skeleton';

// Lazy-loaded page components for route-based bundle splitting
const Home = lazy(() => import('./pages/Home'));
const SignIn = lazy(() => import('./pages/auth/SignIn'));
const Register = lazy(() => import('./pages/auth/Register'));
const About = lazy(() => import('./pages/About'));

const DiabetesPrediction = lazy(() => import('./pages/DiabetesPrediction'));
const HeartDiseasePrediction = lazy(() => import('./pages/HeartDiseasePrediction'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const ClinicianWorklist = lazy(() => import('./pages/clinician/ClinicianWorklist'));
const ModelAnalytics = lazy(() => import('./pages/admin/ModelAnalytics'));
const AuditLog = lazy(() => import('./pages/admin/AuditLog'));
const SystemHealth = lazy(() => import('./pages/admin/SystemHealth'));
const MedicalKnowledge = lazy(() => import('./pages/knowledge/MedicalKnowledge'));

const PageFallback = () => (
  <div className="w-full space-y-6 py-8" aria-label="Loading page contents">
    <Skeleton variant="title" className="w-1/3" />
    <Skeleton variant="text" className="w-2/3" />
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
      <Skeleton variant="card" className="h-60" />
      <Skeleton variant="card" className="h-60" />
    </div>
  </div>
);

function App() {
  return (
    <Router>
      <AuthProvider>
        <Suspense fallback={<PageFallback />}>
          <Routes>
            {/* Public Routes */}
            <Route
              path="/"
              element={
                <PublicShell>
                  <Home />
                </PublicShell>
              }
            />
            <Route
              path="/login"
              element={
                <PublicShell>
                  <SignIn />
                </PublicShell>
              }
            />
            <Route
              path="/register"
              element={
                <PublicShell>
                  <Register />
                </PublicShell>
              }
            />

            {/* Workspace Routes */}
            <Route
              path="/diabetes"
              element={
                <WorkspaceShell>
                  <DiabetesPrediction />
                </WorkspaceShell>
              }
            />
            <Route
              path="/heart"
              element={
                <WorkspaceShell>
                  <HeartDiseasePrediction />
                </WorkspaceShell>
              }
            />
            <Route
              path="/dashboard"
              element={
                <WorkspaceShell>
                  <Dashboard />
                </WorkspaceShell>
              }
            />
            <Route
              path="/worklist"
              element={
                <ProtectedRoute allowedRoles={['CLINICIAN', 'ADMIN']}>
                  <WorkspaceShell wide>
                    <ClinicianWorklist />
                  </WorkspaceShell>
                </ProtectedRoute>
              }
            />
            <Route
              path="/models"
              element={
                <WorkspaceShell wide>
                  <ModelAnalytics />
                </WorkspaceShell>
              }
            />
            <Route
              path="/audit"
              element={
                <ProtectedRoute allowedRoles={['CLINICIAN', 'ADMIN']}>
                  <WorkspaceShell wide>
                    <AuditLog />
                  </WorkspaceShell>
                </ProtectedRoute>
              }
            />
            <Route
              path="/system-health"
              element={
                <ProtectedRoute allowedRoles={['CLINICIAN', 'ADMIN']}>
                  <WorkspaceShell wide>
                    <SystemHealth />
                  </WorkspaceShell>
                </ProtectedRoute>
              }
            />
            <Route
              path="/knowledge"
              element={
                <WorkspaceShell>
                  <MedicalKnowledge />
                </WorkspaceShell>
              }
            />
            <Route
              path="/about"
              element={
                <WorkspaceShell>
                  <About />
                </WorkspaceShell>
              }
            />
          </Routes>
        </Suspense>
      </AuthProvider>
    </Router>
  );
}

export default App;
