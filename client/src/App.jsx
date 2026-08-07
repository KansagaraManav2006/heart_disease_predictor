import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './pages/Home';
import DiabetesPrediction from './pages/DiabetesPrediction';
import HeartDiseasePrediction from './pages/HeartDiseasePrediction';
import About from './pages/About';
import Dashboard from './pages/Dashboard';
import SignIn from './pages/auth/SignIn';
import Register from './pages/auth/Register';
import ClinicianWorklist from './pages/clinician/ClinicianWorklist';
import ModelAnalytics from './pages/admin/ModelAnalytics';
import MedicalKnowledge from './pages/knowledge/MedicalKnowledge';
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider } from './context/AuthContext';

function App() {
  return (
    <Router>
      <AuthProvider>
        <div className="min-h-screen bg-background text-textMain flex font-sans">
          
          {/* Sidebar Left */}
          <Sidebar className="hidden md:block" />

          {/* Main Wrapper */}
          <div className="flex-1 flex flex-col md:ml-64 min-h-screen transition-all duration-300">
            
            {/* Top Header */}
            <Header />

            {/* Page Content */}
            <main className="flex-1 p-4 md:p-8 relative">
              <div className="max-w-5xl mx-auto flex flex-col min-h-full">
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/login" element={<SignIn />} />
                  <Route path="/register" element={<Register />} />
                  <Route path="/diabetes" element={<DiabetesPrediction />} />
                  <Route path="/heart" element={<HeartDiseasePrediction />} />
                  <Route path="/dashboard" element={<Dashboard />} />
                  <Route
                    path="/worklist"
                    element={
                      <ProtectedRoute allowedRoles={['CLINICIAN', 'ADMIN']}>
                        <ClinicianWorklist />
                      </ProtectedRoute>
                    }
                  />
                  <Route path="/models" element={<ModelAnalytics />} />
                  <Route path="/knowledge" element={<MedicalKnowledge />} />
                  <Route path="/about" element={<About />} />
                </Routes>
              </div>
            </main>

            {/* Footer at bottom */}
            <Footer />

          </div>
        </div>
      </AuthProvider>
    </Router>
  );
}

export default App;
