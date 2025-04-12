import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { getCurrentUser, logout } from '../services/api';
import ThemeToggle from './ThemeToggle';
import { Link } from 'react-router-dom';

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const [currentUser, setCurrentUser] = useState(getCurrentUser());

  useEffect(() => {
    // Update currentUser when location changes
    setCurrentUser(getCurrentUser());
  }, [location.pathname]);

  const handleLogout = () => {
    logout();
    navigate('/login');
    setIsMenuOpen(false);
  };

  const handleNavigation = (path) => {
    try {
      setIsMenuOpen(false);
      // Add a small delay to ensure state updates are processed
      setTimeout(() => {
        navigate(path);
      }, 0);
    } catch (error) {
      console.error('Navigation error:', error);
    }
  };

  const isActive = (path) => {
    return location.pathname === path;
  };

  const navItems = [
    {
      path: '/',
      label: 'Home',
      icon: '🏠',
      requiresAuth: false
    },
    {
      path: '/upload',
      label: 'Upload',
      icon: '⬆️',
      requiresAuth: true
    },
    {
      path: '/history',
      label: 'History',
      icon: '🗃️',
      requiresAuth: true
    },
    {
      path: '/starred',
      label: 'Starred',
      icon: '⭐',
      requiresAuth: true
    },
    {
      path: '/about',
      label: 'About',
      icon: 'ℹ️',
      requiresAuth: false
    }
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 bg-white/80 dark:bg-dark-light/80 backdrop-blur-xs shadow-card dark:shadow-dark-card z-50 animate-fade-in">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center">
              <span className="text-2xl brain-float">🧠</span>
              <span className="ml-2 text-xl font-bold text-gray-900 dark:text-white">BrainScan AI</span>
            </Link>
          </div>

          {/* Center Navigation */}
          <div className="hidden sm:flex sm:items-center sm:space-x-1">
            {navItems.map((item) => (
              (!item.requiresAuth || currentUser) && (
                <button
                  key={item.path}
                  onClick={() => handleNavigation(item.path)}
                  className={`nav-link ${isActive(item.path) ? 'active' : 'text-cool-DEFAULT hover:text-primary-600 dark:text-gray-300 dark:hover:text-blue-400'}`}
                >
                  <span className="mr-2 text-lg group-hover:animate-float">{item.icon}</span>
                  {item.label}
                </button>
              )
            ))}
          </div>

          {/* Right side - Login/Logout and Theme Toggle */}
          <div className="flex items-center space-x-4">
            {/* Theme Toggle */}
            <ThemeToggle />
            
            {currentUser ? (
              <button
                onClick={handleLogout}
                className="btn-primary"
              >
                Logout
              </button>
            ) : (
              <button
                onClick={() => handleNavigation('/login')}
                className="btn-primary"
              >
                Login
              </button>
            )}

            {/* Mobile menu button */}
            <div className="sm:hidden">
              <button
                onClick={() => setIsMenuOpen(!isMenuOpen)}
                className="inline-flex items-center justify-center p-2 rounded-lg text-cool-DEFAULT hover:text-primary-600 hover:bg-primary-50 dark:text-gray-300 dark:hover:text-blue-400 dark:hover:bg-dark-lighter focus:outline-none focus:ring-2 focus:ring-inset focus:ring-purple-light dark:focus:ring-blue-400 transition-colors duration-200"
                aria-expanded={isMenuOpen}
              >
                <span className="sr-only">Open main menu</span>
                {isMenuOpen ? (
                  <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                ) : (
                  <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <div 
        className={`sm:hidden transition-all duration-200 ease-in-out ${
          isMenuOpen ? 'animate-slide-down opacity-100' : 'opacity-0 -translate-y-4'
        }`}
      >
        {isMenuOpen && (
          <div className="p-2 bg-white/80 dark:bg-dark-light/80 backdrop-blur-xs shadow-card dark:shadow-dark-card">
            {navItems.map((item) => (
              (!item.requiresAuth || currentUser) && (
                <button
                  key={item.path}
                  onClick={() => handleNavigation(item.path)}
                  className={`w-full flex items-center px-4 py-3 text-base font-medium rounded-lg transition-all duration-200 ${
                    isActive(item.path)
                      ? 'bg-gradient-primary dark:bg-gradient-dark text-white shadow-glow dark:shadow-dark-glow'
                      : 'text-cool-DEFAULT hover:bg-primary-50 hover:text-primary-600 dark:text-gray-300 dark:hover:bg-dark-lighter dark:hover:text-blue-400'
                  }`}
                >
                  <span className="mr-3 text-lg">{item.icon}</span>
                  {item.label}
                </button>
              )
            ))}
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar; 