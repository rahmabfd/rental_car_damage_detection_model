// frontend/src/App.js
import React, { useState } from 'react';
import Rent from './rent';
import Return from './return';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('rent');

  return (
    <div className="app">
      <nav className="app-nav">
        <div className="nav-container">
          <div className="nav-brand">
            <h2>🚗 CarRental Pro</h2>
          </div>
          <div className="nav-menu">
            <button 
              className={`nav-btn ${currentView === 'rent' ? 'active' : ''}`}
              onClick={() => setCurrentView('rent')}
            >
              📝 Create Rental
            </button>
            <button 
              className={`nav-btn ${currentView === 'return' ? 'active' : ''}`}
              onClick={() => setCurrentView('return')}
            >
              🔄 Return Vehicle
            </button>
          </div>
        </div>
      </nav>

      <main className="app-main">
        {currentView === 'rent' && <Rent />}
        {currentView === 'return' && <Return />}
      </main>
    </div>
  );
}

export default App;