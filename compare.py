import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from tabulate import tabulate
import warnings
import json
import os
warnings.filterwarnings('ignore')

class AdvancedDiabetesPredictor:
    def __init__(self):
        self.best_model = None
        self.feature_selector = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = None
        self.performance_history = []
    
    def advanced_feature_engineering(self, df):
        df_enhanced = df.copy()
        df_enhanced['bmi_category'] = pd.cut(
            df_enhanced['bmi'], 
            bins=[0, 18.5, 25, 30, float('inf')], 
            labels=[0, 1, 2, 3]
        )
        df_enhanced['age_group'] = pd.cut(
            df_enhanced['age'], 
            bins=[0, 30, 45, 60, float('inf')], 
            labels=[0, 1, 2, 3]
        )
        df_enhanced['glucose_level'] = pd.cut(
            df_enhanced['glucose'], 
            bins=[0, 100, 125, 200, float('inf')], 
            labels=[0, 1, 2, 3]
        )
        df_enhanced['bmi_age_interaction'] = df_enhanced['bmi'] * df_enhanced['age']
        df_enhanced['glucose_bmi_interaction'] = df_enhanced['glucose'] * df_enhanced['bmi']
        df_enhanced['insulin_glucose_ratio'] = df_enhanced['insulin'] / (df_enhanced['glucose'] + 1)
        df_enhanced['pregnancy_risk'] = (df_enhanced['pregnancies'] > 0).astype(int)
        df_enhanced['high_pregnancy'] = (df_enhanced['pregnancies'] > 5).astype(int)
        df_enhanced['bp_category'] = pd.cut(
            df_enhanced['blood_pressure'], 
            bins=[0, 80, 90, 140, float('inf')], 
            labels=[0, 1, 2, 3]
        )
        df_enhanced['skin_thickness_cat'] = pd.cut(
            df_enhanced['skin_thickness'], 
            bins=[0, 20, 30, 40, float('inf')], 
            labels=[0, 1, 2, 3]
        )
        df_enhanced['pedigree_risk'] = pd.cut(
            df_enhanced['diabetes_pedigree'], 
            bins=[0, 0.2, 0.5, 1.0, float('inf')], 
            labels=[0, 1, 2, 3]
        )
        return df_enhanced
    
    def create_ensemble_model(self):
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        svm = SVC(probability=True, random_state=42)
        
        voting_clf = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('svm', svm)],
            voting='soft'
        )
        
        stacking_clf = StackingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            final_estimator=LogisticRegression(),
            cv=5
        )
        
        return voting_clf, stacking_clf
    
    def advanced_model_training(self, X, y, use_feature_selection=True):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.feature_selection import SelectKBest, f_classif
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if use_feature_selection:
            selector = SelectKBest(score_func=f_classif, k=min(10, X_train.shape[1]))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            self.feature_selector = selector
        else:
            X_train_selected = X_train
            X_test_selected = X_test
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        self.scaler = scaler
        
        voting_clf, stacking_clf = self.create_ensemble_model()
        models = {
            'Voting_Classifier': voting_clf,
            'Stacking_Classifier': stacking_clf,
            'Random_Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'Gradient_Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.best_model = results[best_model_name]['model']
        self.is_trained = True
        self.performance_history.append(results)
        return results
    
    def predict_with_confidence(self, user_input):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        user_input = np.array(user_input).reshape(1, -1)
        if self.feature_selector is not None:
            user_input = self.feature_selector.transform(user_input)
        
        user_input_scaled = self.scaler.transform(user_input)
        prediction = self.best_model.predict(user_input_scaled)[0]
        probabilities = self.best_model.predict_proba(user_input_scaled)[0]
        confidence = abs(probabilities[1] - 0.5) * 2
        
        return {
            'prediction': int(prediction),
            'prediction_label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability_diabetic': float(probabilities[1]),
            'probability_non_diabetic': float(probabilities[0]),
            'confidence': float(confidence),
            'risk_assessment': self._detailed_risk_assessment(probabilities[1])
        }
    
    def _detailed_risk_assessment(self, prob):
        if prob < 0.1:
            return {'level': 'Very Low', 'description': 'Minimal diabetes risk.', 'recommendation': 'Maintain healthy lifestyle.'}
        elif prob < 0.3:
            return {'level': 'Low', 'description': 'Below average risk.', 'recommendation': 'Continue regular monitoring.'}
        elif prob < 0.5:
            return {'level': 'Moderate', 'description': 'Moderate risk detected.', 'recommendation': 'Consider lifestyle improvements.'}
        elif prob < 0.7:
            return {'level': 'High', 'description': 'Above average risk.', 'recommendation': 'Consult healthcare provider.'}
        else:
            return {'level': 'Very High', 'description': 'High risk indicated.', 'recommendation': 'Seek immediate medical consultation.'}

def compare_model_predictions():
    predictor = AdvancedDiabetesPredictor()
    
    # Generate synthetic training data
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000, n_features=8, n_informative=6, 
        n_redundant=2, n_clusters_per_class=1, random_state=42
    )
    
    feature_names = [
        'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
        'insulin', 'bmi', 'diabetes_pedigree', 'age'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['pregnancies'] = (np.abs(df['pregnancies']) * 2 + 1).astype(int)
    df['glucose'] = np.abs(df['glucose']) * 30 + 100
    df['blood_pressure'] = np.abs(df['blood_pressure']) * 20 + 60
    df['skin_thickness'] = np.abs(df['skin_thickness']) * 15 + 15
    df['insulin'] = np.abs(df['insulin']) * 100 + 50
    df['bmi'] = np.abs(df['bmi']) * 10 + 20
    df['diabetes_pedigree'] = np.abs(df['diabetes_pedigree']) * 0.5 + 0.1
    df['age'] = (np.abs(df['age']) * 15 + 25).astype(int)
    df['target'] = y
    
    # Feature engineering and training
    df_enhanced = predictor.advanced_feature_engineering(df)
    X_enhanced = df_enhanced.drop('target', axis=1)
    y_enhanced = df_enhanced['target']
    results = predictor.advanced_model_training(X_enhanced, y_enhanced)
    
    # Define test input samples (representing different patient profiles)
    test_inputs = [
        {'pregnancies': 2, 'glucose': 120, 'blood_pressure': 80, 'skin_thickness': 25, 'insulin': 100, 'bmi': 28.5, 'diabetes_pedigree': 0.3, 'age': 35},  # Normal
        {'pregnancies': 5, 'glucose': 160, 'blood_pressure': 90, 'skin_thickness': 30, 'insulin': 150, 'bmi': 35.0, 'diabetes_pedigree': 0.8, 'age': 45},  # High risk
        {'pregnancies': 0, 'glucose': 90, 'blood_pressure': 70, 'skin_thickness': 20, 'insulin': 80, 'bmi': 22.0, 'diabetes_pedigree': 0.2, 'age': 25},   # Low risk
        {'pregnancies': 8, 'glucose': 180, 'blood_pressure': 100, 'skin_thickness': 35, 'insulin': 200, 'bmi': 40.0, 'diabetes_pedigree': 1.0, 'age': 55}, # Very high risk
        {'pregnancies': 3, 'glucose': 130, 'blood_pressure': 85, 'skin_thickness': 28, 'insulin': 120, 'bmi': 30.0, 'diabetes_pedigree': 0.5, 'age': 40}   # Moderate risk
    ]
    
    # Prepare models for comparison (including the best model as Advanced_Model)
    models = {
        'Voting_Classifier': results['Votin' \
        'g_Classifier']['model'],
        'Stacking_Classifier': results['Stacking_Classifier']['model'],
        'Random_Forest': results['Random_Forest']['model'],
        'Gradient_Boosting': results['Gradient_Boosting']['model'],
        'Logistic_Regression': results['Logistic_Regression']['model'],
        'Advanced_Model': predictor.best_model
    }
    
    # Store results for all inputs and models
    comparison_results = []
    chart_configs = []
    
    print("\nComparing Model Predictions Across Different Inputs")
    print("=" * 60)
    

    import os
    import matplotlib.pyplot as plt
    

    # Assuming you already have 'models', 'predictor', and 'test_inputs' defined
    comparison_results = []

    # Directory to save images
    image_dir = 'static/images'
    os.makedirs(image_dir, exist_ok=True)

    for i, input_data in enumerate(test_inputs, 1):
        patient_df = pd.DataFrame([input_data])
        patient_enhanced = predictor.advanced_feature_engineering(patient_df)
        patient_array = patient_enhanced.values[0]

        input_result = {'Input': f'Patient {i}'}
        input_description = (
            f"Patient {i}: Preg={input_data['pregnancies']}, Gluc={input_data['glucose']}, "
            f"BP={input_data['blood_pressure']}, Skin={input_data['skin_thickness']}, "
            f"Ins={input_data['insulin']}, BMI={input_data['bmi']}, "
            f"Ped={input_data['diabetes_pedigree']:.1f}, Age={input_data['age']}"
        )
        
        print(f"\n{input_description}")
        print("-" * 60)
        
        probabilities = []
        labels = []
        colors = []

        for model_name, model in models.items():
            predictor.best_model = model
            result = predictor.predict_with_confidence(patient_array)
            prob_diabetic = result['probability_diabetic']
            risk_level = result['risk_assessment']['level']

            input_result[f"{model_name}_Prob"] = prob_diabetic
            input_result[f"{model_name}_Risk"] = risk_level

            labels.append(model_name.replace("_", " "))
            probabilities.append(prob_diabetic)
            
            # Advanced_Model in red, others in teal
            if model_name == "Advanced_Model":
                colors.append("#ff6b6b")
            else:
                colors.append("#4ecdc4")

            print(f"{model_name.replace('_', ' '):20} | Prob Diabetic: {prob_diabetic:.3f} | Risk: {risk_level}")

        comparison_results.append(input_result)

        # Generate Bar Chart with matplotlib
        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, probabilities, color=colors, edgecolor='#333333')
        plt.ylim(0, 1)
        plt.ylabel("Probability of Diabetes")
        plt.xlabel("Model")
        plt.title(f"Diabetes Probability for {input_description}")
        
        # Show probability value on top of each bar
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{prob:.2f}", ha='center')

        # Save image
        image_path = os.path.join(image_dir, f"patient_{i}_prediction_chart.png")
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()

        print(f"Image saved to {image_path}")

    # Restore the best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    predictor.best_model = results[best_model_name]['model']

    print("\nAll charts generated and saved.")

    # for i, input_data in enumerate(test_inputs, 1):
    #     patient_df = pd.DataFrame([input_data])
    #     patient_enhanced = predictor.advanced_feature_engineering(patient_df)
    #     patient_array = patient_enhanced.values[0]
        
    #     input_result = {'Input': f'Patient {i}'}
    #     input_description = f"Patient {i}: Preg={input_data['pregnancies']}, Gluc={input_data['glucose']}, BP={input_data['blood_pressure']}, Skin={input_data['skin_thickness']}, Ins={input_data['insulin']}, BMI={input_data['bmi']}, Ped={input_data['diabetes_pedigree']:.1f}, Age={input_data['age']}"
        
    #     print(f"\n{input_description}")
    #     print("-" * 60)
        
    #     # Collect probabilities for chart
    #     probabilities = []
    #     for model_name, model in models.items():
    #         # Temporarily set model for prediction
    #         predictor.best_model = model
    #         result = predictor.predict_with_confidence(patient_array)
    #         prob_diabetic = result['probability_diabetic']
    #         risk_level = result['risk_assessment']['level']
            
    #         input_result[f"{model_name}_Prob"] = prob_diabetic
    #         input_result[f"{model_name}_Risk"] = risk_level
    #         probabilities.append(prob_diabetic)
    #         print(f"{model_name.replace('_', ' '):20} | Prob Diabetic: {prob_diabetic:.3f} | Risk: {risk_level}")
        
    #     comparison_results.append(input_result)
        
    #     # Create Chart.js configuration for this patient
    #     chart_config = {
    #         "type": "bar",
    #         "data": {
    #             "labels": [name.replace('_', ' ') for name in models.keys()],
    #             "datasets": [{
    #                 "label": "Probability of Diabetes",
    #                 "data": probabilities,
    #                 "backgroundColor": [
    #                     "#4ecdc4" if name != "Advanced_Model" else "#ff6b6b"
    #                     for name in models.keys()
    #                 ],
    #                 "borderColor": [
    #                     "#3ca8a1" if name != "Advanced_Model" else "#cc5555"
    #                     for name in models.keys()
    #                 ],
    #                 "borderWidth": 1
    #             }]
    #         },
    #         "options": {
    #             "scales": {
    #                 "y": {
    #                     "beginAtZero": True,
    #                     "max": 1,
    #                     "title": {
    #                         "display": True,
    #                         "text": "Probability of Diabetes"
    #                     }
    #                 },
    #                 "x": {
    #                     "title": {
    #                         "display": True,
    #                         "text": "Model"
    #                     }
    #                 }
    #             },
    #             "plugins": {
    #                 "title": {
    #                     "display": True,
    #                     "text": f"Diabetes Probability for {input_description}",
    #                     "font": {
    #                         "size": 16
    #                     }
    #                 },
    #                 "legend": {
    #                     "display": False
    #                 }
    #             }
    #         }
    #     }
    #     chart_configs.append((f"patient_{i}_prediction_chart", chart_config))
    
    # # Create table headers to match dictionary keys
    # headers = ['Input']
    # for model_name in models.keys():
    #     headers.extend([f"{model_name}_Prob", f"{model_name}_Risk"])
    
    # # Print results in table format
    # print("\nPrediction Comparison Table")
    # print("=" * 100)

    # print("=*="*20)
    # print(headers)
    # print(comparison_results) 
    # # print(tabulate(comparison_results, headers=headers, tablefmt='grid', floatfmt='.3f'))
    
    # # Save chart configurations
    # static_dir = 'static'
    # if not os.path.exists(static_dir):
    #     os.makedirs(static_dir)
    
    # for chart_name, chart_config in chart_configs:
    #     with open(os.path.join(static_dir, f"{chart_name}.json"), 'w') as f:
    #         json.dump(chart_config, f, indent=2)
    #     print(f"Chart configuration saved to {os.path.join(static_dir, f'{chart_name}.json')}")
    
    # # Restore the best model
    # best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    # predictor.best_model = results[best_model_name]['model']
    
    # return comparison_results, chart_configs

if __name__ == '__main__':
    comparison_results, chart_configs = compare_model_predictions()
    
    # Print Chart.js configurations for visualization
    for chart_name, chart_config in chart_configs:
        print(f"\nChart for {chart_name}:")
        print("```chartjs")
        print(json.dumps(chart_config, indent=2))
        print("```")