import React, { useState } from 'react';
import Header from './components/Header';
import SalaryPredictionForm from './components/SalaryPredictionForm';
import SalaryResultDisplay from './components/SalaryResultDisplay';
import MarketAnalysis from './components/MarketAnalysis';
import Footer from './components/Footer';

function App() {
  const [salaryData, setSalaryData] = useState(null);

  // Function to update salaryData state after fetching from Flask API

  return (
    <div>
      <Header />
      <SalaryPredictionForm />
      <SalaryResultDisplay salaryData={salaryData} />
      <MarketAnalysis />
      <Footer />
    </div>
  );
}

export default App;
