import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { getCurrentUser } from './services/api';

// Import components
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import Upload from './pages/Upload';
import History from './pages/History';
import Starred from './pages/Starred';
import About from './pages/About';

// Protected route component
const ProtectedRoute = ({ children }) => {
  const currentUser = getCurrentUser();
  if (!currentUser) {
    return <Navigate to="/login" />;
  }
  return children;
};

function App() {
  useEffect(() => {
    // Check for saved theme preference or use system preference
    const savedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (savedTheme === 'dark' || (!savedTheme && systemPrefersDark)) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-gradient-light dark:bg-gradient-dark transition-colors duration-200">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/about" element={<About />} />
            
            {/* Protected routes */}
            <Route 
              path="/upload" 
              element={
                <ProtectedRoute>
                  <Upload />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/history" 
              element={
                <ProtectedRoute>
                  <History />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/starred" 
              element={
                <ProtectedRoute>
                  <Starred />
                </ProtectedRoute>
              } 
            />
            
            {/* Fallback route */}
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
        {/* <footer className="bg-gray-900 text-white py-6">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center">
            <span className="text-2xl">🧠</span>
            <span className="ml-2 text-lg font-semibold">BrainScan AI</span>
          </div>
          <div className="mt-4 md:mt-0 text-sm text-gray-400">
            <p className="mb-1">© 2025 BrainScan AI. All rights reserved.</p>
            <p className="text-gray-500">
              Disclaimer: This is an AI-assisted tool and should not replace professional medical diagnosis.
            </p>
          </div>
        </div>
      </div>
    </footer> */}
      </div>
    </Router>
  );
}

export default App;
