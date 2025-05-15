import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [formData, setFormData] = useState({
    Pregnancies: '',
    Glucose: '',
    BloodPressure: '',
    SkinThickness: '',
    Insulin: '',
    BMI: '',
    DiabetesPedigreeFunction: '',
    Age: ''
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    // Parse all values to numbers before sending
    const parsedFormData = Object.keys(formData).reduce((acc, key) => {
        acc[key] = parseFloat(formData[key]);
        return acc;
    }, {});

    try {
        const res = await axios.post('http://localhost:5000/predict', parsedFormData);
        setPrediction(res.data.prediction);
    } catch (error) {
        console.error(error);
        setError('Error: Unable to fetch prediction. Please try again.');
    } finally {
        setLoading(false);
    }
};


  const handleClear = () => {
    setFormData({
      Pregnancies: '',
      Glucose: '',
      BloodPressure: '',
      SkinThickness: '',
      Insulin: '',
      BMI: '',
      DiabetesPedigreeFunction: '',
      Age: ''
    });
    setPrediction(null);
  };

  return (
    <div className="App bg-gray-900 text-white min-h-screen flex flex-col justify-center items-center">
      <h1 className="text-4xl font-bold mb-6">Diabetes Prediction</h1>
      <form onSubmit={handleSubmit} className="bg-gray-800 p-6 rounded-lg shadow-lg w-80">
        {Object.keys(formData).map(key => (
          <div key={key} className="mb-4">
            <label className="block text-sm font-semibold">{key}:</label>
            <input
              type="number"
              name={key}
              value={formData[key]}
              onChange={handleChange}
              className="mt-2 p-2 w-full rounded-md bg-gray-700 text-white"
              required
            />
          </div>
        ))}
        <button
          type="submit"
          className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md"
          disabled={loading}  // Disable button while loading
        >
          {loading ? 'Predicting...' : 'Predict'}
        </button>
        <button
          type="button"
          onClick={handleClear}
          className="w-full mt-4 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-md"
        >
          Clear
        </button>
      </form>

      {loading && (
        <div className="mt-6 text-lg text-gray-300">Loading...</div>
      )}

      {error && (
        <div className="mt-6 text-lg text-red-600">{error}</div>
      )}

      {prediction !== null && !loading && (
        <div className="mt-6 text-lg">
          <h2 className="text-xl font-semibold">
            Result: {prediction === 1 ? "Diabetic" : "Not Diabetic"}
          </h2>
        </div>
      )}
    </div>
  );
}

export default App;
