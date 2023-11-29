import React from 'react';

function SalaryResultDisplay({ salaryData }) {
  return (
    <div>
      {salaryData && (
        <div>
          <h2>Predicted Salary: {salaryData.predictedSalary}</h2>
          {/* Display more data if available */}
        </div>
      )}
    </div>
  );
}

export default SalaryResultDisplay;
