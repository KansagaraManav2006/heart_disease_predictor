import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './pages/Home';
import DiabetesPrediction from './pages/DiabetesPrediction';
import HeartDiseasePrediction from './pages/HeartDiseasePrediction';
import About from './pages/About';

function App() {
  return (
    <Router>
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
                <Route path="/diabetes" element={<DiabetesPrediction />} />
                <Route path="/heart" element={<HeartDiseasePrediction />} />
                <Route path="/about" element={<About />} />
              </Routes>
            </div>
          </main>

          {/* Footer at bottom */}
          <Footer />

        </div>
      </div>
    </Router>
  );
}

export default App;
