from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend React app to access backend

# Load all three models at startup
models = {
    "logistic": joblib.load('logistic_model.pkl'),
    "tree": joblib.load('decision_tree_model.pkl'),
    "forest": joblib.load('random_forest_model.pkl')
}

# Optional: Load the scaler if you used one during training (Uncomment if needed)
# scaler = joblib.load('scaler.pkl')  # Replace with actual scaler file if used

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = [
            data['Pregnancies'],
            data['Glucose'],
            data['BloodPressure'],
            data['SkinThickness'],
            data['Insulin'],
            data['BMI'],
            data['DiabetesPedigreeFunction'],
            data['Age']
        ]
        
        print("Input features:", features)  # Debugging line

        model_choice = data.get('model', 'forest')

        if model_choice not in models:
            return jsonify({'error': 'Invalid model choice. Choose from: logistic, tree, forest.'}), 400

        model = models[model_choice]
        prediction = model.predict([features])
        
        print("Model Prediction:", prediction)  # Debugging line
        result = int(prediction[0])
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
