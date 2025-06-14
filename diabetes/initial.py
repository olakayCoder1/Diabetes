import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictionSystem:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                             'insulin', 'bmi', 'diabetes_pedigree', 'age']
    
    def load_sample_data(self):
        """
        Load or create sample diabetes dataset
        In practice, you'd load the Pima Indians Diabetes Dataset
        """
        # Create synthetic data similar to diabetes dataset
        np.random.seed(42)
        X, y = make_classification(n_samples=768, n_features=8, n_informative=6, 
                                 n_redundant=2, n_clusters_per_class=1, random_state=42)
        
        # Scale features to match typical diabetes dataset ranges
        X[:, 0] = np.abs(X[:, 0] * 2 + 3)  # pregnancies
        X[:, 1] = np.abs(X[:, 1] * 30 + 120)  # glucose
        X[:, 2] = np.abs(X[:, 2] * 20 + 70)  # blood pressure
        X[:, 3] = np.abs(X[:, 3] * 15 + 20)  # skin thickness
        X[:, 4] = np.abs(X[:, 4] * 100 + 80)  # insulin
        X[:, 5] = np.abs(X[:, 5] * 10 + 25)  # BMI
        X[:, 6] = np.abs(X[:, 6] * 0.5 + 0.3)  # diabetes pedigree
        X[:, 7] = np.abs(X[:, 7] * 15 + 30)  # age
        
        df = pd.DataFrame(X, columns=self.feature_names)
        df['target'] = y
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the diabetes dataset
        """
        # Handle missing values (represented as 0 in some features)
        zero_features = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
        
        for feature in zero_features:
            if feature in df.columns:
                # Replace 0 with median for these features
                median_val = df[df[feature] != 0][feature].median()
                df[feature] = df[feature].replace(0, median_val)
        
        # Separate features and target
        X = df[self.feature_names]
        y = df['target']
        
        return X, y
    
    def initialize_models(self):
        """
        Initialize different machine learning models for comparison
        """
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
    
    def train_and_evaluate(self, X, y, test_size=0.2):
        """
        Train all models and evaluate their performance
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                          random_state=42, stratify=y)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        print("Training and evaluating models...")
        print("=" * 60)
        
        for name, model in self.models.items():
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print("-" * 40)
        
        self.is_fitted = True
        return results, X_test, y_test, X_test_scaled
    
    def plot_model_comparison(self, results):
        """
        Plot comparison of different models
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            
            axes[i].bar(model_names, values, color='skyblue', alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Cross-validation comparison
        cv_means = [results[model]['cv_mean'] for model in model_names]
        cv_stds = [results[model]['cv_std'] for model in model_names]
        
        axes[5].bar(model_names, cv_means, yerr=cv_stds, capsize=5, 
                   color='lightcoral', alpha=0.7)
        axes[5].set_title('Cross-Validation Score Comparison')
        axes[5].set_ylabel('CV Score')
        axes[5].set_ylim(0, 1)
        axes[5].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, results, X_test_scaled, y_test):
        """
        Plot confusion matrices for all models
        """
        n_models = len(results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.ravel()
        
        for i, (name, result) in enumerate(results.items()):
            model = result['model']
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name} - Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(len(results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def get_best_model(self, results):
        """
        Identify the best performing model based on F1-score
        """
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_model = results[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"F1-Score: {best_model['f1_score']:.4f}")
        print(f"Accuracy: {best_model['accuracy']:.4f}")
        print(f"AUC: {best_model['auc']:.4f}")
        
        return best_model_name, best_model['model']
    
    def predict_diabetes(self, user_input, model_name='Random Forest'):
        """
        Predict diabetes for user input
        user_input should be a list or array with 8 features:
        [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
        """
        if not self.is_fitted:
            raise ValueError("Models have not been trained yet. Call train_and_evaluate first.")
        
        # Convert to numpy array and reshape
        user_input = np.array(user_input).reshape(1, -1)
        
        # Scale the input
        user_input_scaled = self.scaler.transform(user_input)
        
        # Get the specified model
        model = self.models[model_name]
        
        # Make prediction
        prediction = model.predict(user_input_scaled)[0]
        probability = model.predict_proba(user_input_scaled)[0]
        
        result = {
            'prediction': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability_non_diabetic': probability[0],
            'probability_diabetic': probability[1],
            'risk_level': self._get_risk_level(probability[1])
        }
        
        return result
    
    def _get_risk_level(self, prob):
        """
        Determine risk level based on probability
        """
        if prob < 0.3:
            return 'Low Risk'
        elif prob < 0.6:
            return 'Medium Risk'
        else:
            return 'High Risk'
    
    def hyperparameter_tuning(self, X, y, model_name='Random Forest'):
        """
        Perform hyperparameter tuning for a specific model
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          random_state=42, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
        
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            model = GradientBoostingClassifier(random_state=42)
        
        else:
            print(f"Hyperparameter tuning not implemented for {model_name}")
            return None
        
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"Best parameters for {model_name}:")
        print(grid_search.best_params_)
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_

# Example usage and demonstration
def main():
    # Initialize the system
    diabetes_system = DiabetesPredictionSystem()
    
    # Load sample data
    df = diabetes_system.load_sample_data()
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Preprocess data
    X, y = diabetes_system.preprocess_data(df)
    
    # Initialize models
    diabetes_system.initialize_models()
    
    # Train and evaluate models
    results, X_test, y_test, X_test_scaled = diabetes_system.train_and_evaluate(X, y)
    
    # Plot comparisons
    diabetes_system.plot_model_comparison(results)
    diabetes_system.plot_confusion_matrices(results, X_test_scaled, y_test)
    
    # Get best model
    best_model_name, best_model = diabetes_system.get_best_model(results)
    
    # Example prediction for a user
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    
    # Sample user input: [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    sample_input = [2, 138, 62, 35, 0, 33.6, 0.127, 47]
    
    print(f"User Input: {sample_input}")
    print("Features: pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age")
    
    prediction_result = diabetes_system.predict_diabetes(sample_input, best_model_name)
    
    print(f"\nPrediction: {prediction_result['prediction']}")
    print(f"Risk Level: {prediction_result['risk_level']}")
    print(f"Probability of Diabetes: {prediction_result['probability_diabetic']:.3f}")
    print(f"Probability of Non-Diabetes: {prediction_result['probability_non_diabetic']:.3f}")
    
    # Hyperparameter tuning example
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    best_tuned_model = diabetes_system.hyperparameter_tuning(X, y, 'Random Forest')
    
    return diabetes_system, results

if __name__ == "__main__":
    system, results = main()