import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import DiabetesPrediction from './pages/DiabetesPrediction';
import HeartDiseasePrediction from './pages/HeartDiseasePrediction';
import About from './pages/About';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-darkBg text-textMain selection:bg-primary/30 relative overflow-hidden">
        {/* Vibrant Ambient Orbs for Glass Refraction */}
        <div className="fixed top-[-10%] left-[-10%] w-[40vw] h-[40vw] rounded-full bg-primary/20 blur-[120px] pointer-events-none opacity-60"></div>
        <div className="fixed bottom-[-10%] right-[-10%] w-[35vw] h-[35vw] rounded-full bg-blue-500/20 blur-[120px] pointer-events-none opacity-60"></div>
        <div className="fixed top-[30%] left-[60%] w-[25vw] h-[25vw] rounded-full bg-purple-500/20 blur-[100px] pointer-events-none opacity-50"></div>

        <div className="fixed inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.03] pointer-events-none mix-blend-overlay"></div>

        <Navbar />

        <main className="container mx-auto px-4 relative z-10 pb-20">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/diabetes" element={<DiabetesPrediction />} />
            <Route path="/heart" element={<HeartDiseasePrediction />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
