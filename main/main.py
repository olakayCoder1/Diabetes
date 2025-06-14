from flask import Flask, request, jsonify

from main.setup import AdvancedDiabetesPredictor

app = Flask(__name__)
predictor = AdvancedDiabetesPredictor()
predictor.load_model('diabetes_model.pkl')

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    data = request.json
    user_input = data['features']  # List of 8 features
    
    try:
        result = predictor.predict_with_confidence(user_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)