# Diabetes Prediction Algorithm - Usage Guide
"""
This guide shows how to use the diabetes prediction system with real data
and implement it for production use.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

class SimpleDiabetesPredictor:
    """
    Simplified version for easy implementation and testing
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age'
        ]
    
    def load_real_diabetes_data(self):
        """
        Load the famous Pima Indians Diabetes Dataset
        """
        try:
            # Try to load from sklearn datasets
            diabetes = fetch_openml('diabetes', version=1, as_frame=True)
            X = diabetes.data
            y = diabetes.target.astype(int)
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=self.feature_names)
            df['target'] = y
            
            return df
        except:
            print("Could not load online dataset. Using local CSV if available...")
            # If you have a local CSV file, load it here
            # df = pd.read_csv('diabetes.csv')
            # return df
            return None
    
    def create_sample_data(self):
        """
        Create realistic sample data for testing
        """
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'pregnancies': np.random.poisson(3, n_samples),
            'glucose': np.random.normal(120, 30, n_samples),
            'blood_pressure': np.random.normal(70, 15, n_samples),
            'skin_thickness': np.random.normal(25, 10, n_samples),
            'insulin': np.random.normal(100, 50, n_samples),
            'bmi': np.random.normal(30, 8, n_samples),
            'diabetes_pedigree': np.random.exponential(0.5, n_samples),
            'age': np.random.normal(40, 15, n_samples)
        }
        
        # Create target based on realistic rules
        target = np.zeros(n_samples)
        for i in range(n_samples):
            score = 0
            if data['glucose'][i] > 140: score += 2
            if data['bmi'][i] > 30: score += 1
            if data['age'][i] > 45: score += 1
            if data['diabetes_pedigree'][i] > 0.5: score += 1
            if data['pregnancies'][i] > 5: score += 1
            
            # Add some randomness
            if np.random.random() < (score / 8) * 0.8 + 0.1:
                target[i] = 1
        
        df = pd.DataFrame(data)
        df['target'] = target.astype(int)
        
        return df

def quick_diabetes_prediction_example():
    """
    Quick example showing how to use the system
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    
    print("=== Quick Diabetes Prediction Example ===")
    
    # Create sample data
    predictor = SimpleDiabetesPredictor()
    df = predictor.create_sample_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Diabetes cases: {df['target'].sum()} out of {len(df)} ({df['target'].mean()*100:.1f}%)")
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Example prediction
    print("\n=== Example Prediction ===")
    sample_patient = [
        2,      # pregnancies
        148,    # glucose
        72,     # blood_pressure
        35,     # skin_thickness
        0,      # insulin
        33.6,   # bmi
        0.627,  # diabetes_pedigree
        50      # age
    ]
    
    sample_scaled = scaler.transform([sample_patient])
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0]
    
    print(f"Patient data: {sample_patient}")
    print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    print(f"Probability of diabetes: {probability[1]:.3f}")
    
    return model, scaler

def production_ready_predictor():
    """
    Production-ready diabetes predictor with error handling
    """
    class ProductionDiabetesPredictor:
        def __init__(self):
            self.model = None
            self.scaler = None
            self.is_trained = False
            self.feature_ranges = {
                'pregnancies': (0, 20),
                'glucose': (50, 300),
                'blood_pressure': (40, 180),
                'skin_thickness': (10, 80),
                'insulin': (0, 500),
                'bmi': (15, 50),
                'diabetes_pedigree': (0, 3),
                'age': (18, 100)
            }
        
        def validate_input(self, input_data):
            """Validate user input"""
            if len(input_data) != 8:
                raise ValueError("Input must contain exactly 8 features")
            
            feature_names = list(self.feature_ranges.keys())
            for i, (value, feature) in enumerate(zip(input_data, feature_names)):
                min_val, max_val = self.feature_ranges[feature]
                if not min_val <= value <= max_val:
                    raise ValueError(f"{feature} value {value} is outside valid range [{min_val}, {max_val}]")
        
        def predict(self, input_data):
            """Make prediction with error handling"""
            try:
                # Validate input
                self.validate_input(input_data)
                
                # Scale input
                input_scaled = self.scaler.transform([input_data])
                
                # Make prediction
                prediction = self.model.predict(input_scaled)[0]
                probabilities = self.model.predict_proba(input_scaled)[0]
                
                return {
                    'prediction': int(prediction),
                    'prediction_label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
                    'probability_diabetic': float(probabilities[1]),
                    'risk_level': self._get_risk_level(probabilities[1]),
                    'confidence': float(max(probabilities))
                }
            
            except Exception as e:
                return {'error': str(e)}
        
        def _get_risk_level(self, prob):
            """Determine risk level"""
            if prob < 0.2:
                return 'Very Low'
            elif prob < 0.4:
                return 'Low'
            elif prob < 0.6:
                return 'Moderate'
            elif prob < 0.8:
                return 'High'
            else:
                return 'Very High'
    
    return ProductionDiabetesPredictor()

# Usage examples and testing
if __name__ == "__main__":
    print("Running Diabetes Prediction Examples...")
    
    # Run quick example
    model, scaler = quick_diabetes_prediction_example()
    
    print("\n" + "="*50)
    print("TESTING DIFFERENT PATIENT SCENARIOS")
    print("="*50)
    
    # Test cases
    test_cases = [
        {
            'name': 'Healthy Young Adult',
            'data': [0, 85, 70, 20, 0, 22.5, 0.1, 25],
            'expected': 'Low risk'
        },
        {
            'name': 'High Risk Patient',
            'data': [5, 180, 90, 40, 200, 35.0, 0.8, 55],
            'expected': 'High risk'
        },
        {
            'name': 'Moderate Risk Patient',
            'data': [2, 120, 75, 25, 80, 28.0, 0.3, 35],
            'expected': 'Moderate risk'
        }
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}: {case['data']}")
        prediction = model.predict(scaler.transform([case['data']]))[0]
        probability = model.predict_proba(scaler.transform([case['data']]))[0]
        print(f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
        print(f"Diabetes probability: {probability[1]:.3f}")
        print(f"Expected: {case['expected']}")
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Show which features are most important
    feature_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                    'insulin', 'bmi', 'diabetes_pedigree', 'age']
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Most important features for diabetes prediction:")
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE TIPS")
    print("="*50)
    
    print("""
    To improve your diabetes prediction model:
    
    1. DATA QUALITY:
       - Handle missing values properly (don't just use 0)
       - Remove outliers carefully
       - Ensure balanced dataset
    
    2. FEATURE ENGINEERING:
       - Create BMI categories (underweight, normal, overweight, obese)
       - Age groups (young, middle-aged, elderly)
       - Glucose level categories
       - Interaction features (BMI * age, glucose * BMI)
    
    3. MODEL SELECTION:
       - Try ensemble methods (Random Forest, Gradient Boosting)
       - Consider neural networks for larger datasets
       - Use cross-validation for robust evaluation
    
    4. EVALUATION METRICS:
       - Don't rely only on accuracy
       - Use precision, recall, F1-score
       - Consider AUC-ROC for imbalanced data
       - Use confusion matrix analysis
    
    5. HYPERPARAMETER TUNING:
       - Use GridSearchCV or RandomizedSearchCV
       - Optimize for the right metric (F1 for balanced importance)
       - Use stratified cross-validation
    """)