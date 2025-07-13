import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import traceback
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')




class AdvancedDiabetesPredictor:
    
    def __init__(self):
        self.best_model = None              # Will store the best performing model
        self.feature_selector = None        # Feature selection transformer
        self.scaler = None                  # Data standardization transformer
        self.is_trained = False            # Flag indicating training status
        self.feature_names = None          # Names of features used in training
        self.performance_history = []      # History of training sessions
        
    def advanced_feature_engineering(self, df):
        # Create a copy to avoid modifying the original dataframe
        df_enhanced = df.copy()
        # BMI Risk Categories
        # Based on WHO classification for diabetes risk
        df_enhanced['bmi_category'] = pd.cut(
            df_enhanced['bmi'], 
            bins=[0, 18.5, 25, 30, float('inf')], 
            labels=[0, 1, 2, 3]  # underweight, normal, overweight, obese
        )
        # Age Risk Groups
        # Age is a major diabetes risk factor with different risk levels
        df_enhanced['age_group'] = pd.cut(
            df_enhanced['age'], 
            bins=[0, 30, 45, 60, float('inf')], 
            labels=[0, 1, 2, 3]  # young, middle-aged, senior, elderly
        )
        # Glucose Risk Levels
        # Based on ADA (American Diabetes Association) guidelines
        df_enhanced['glucose_level'] = pd.cut(
            df_enhanced['glucose'], 
            bins=[0, 100, 125, 200, float('inf')], 
            labels=[0, 1, 2, 3]  # normal, prediabetic, diabetic, severe
        )
        
        # Interaction Features
        # These capture how risk factors combine multiplicatively
        df_enhanced['bmi_age_interaction'] = (
            df_enhanced['bmi'] * df_enhanced['age']
        )
        df_enhanced['glucose_bmi_interaction'] = (
            df_enhanced['glucose'] * df_enhanced['bmi']
        )
        # Medical Ratio Features
        # Insulin-to-glucose ratio is clinically meaningful
        df_enhanced['insulin_glucose_ratio'] = (
            df_enhanced['insulin'] / (df_enhanced['glucose'] + 1)  # +1 to avoid division by zero
        )
        # Pregnancy Risk Factors
        # Gestational diabetes history is a strong predictor
        df_enhanced['pregnancy_risk'] = (
            (df_enhanced['pregnancies'] > 0).astype(int)
        )
        df_enhanced['high_pregnancy'] = (
            (df_enhanced['pregnancies'] > 5).astype(int)
        )
        # Blood Pressure Risk Categories
        # Based on AHA (American Heart Association) guidelines
        df_enhanced['bp_category'] = pd.cut(
            df_enhanced['blood_pressure'], 
            bins=[0, 80, 90, 140, float('inf')], 
            labels=[0, 1, 2, 3]  # normal, elevated, high, crisis
        )
        
        # Skin Thickness Categories
        # Triceps skinfold thickness as obesity indicator
        df_enhanced['skin_thickness_cat'] = pd.cut(
            df_enhanced['skin_thickness'], 
            bins=[0, 20, 30, 40, float('inf')], 
            labels=[0, 1, 2, 3]
        )
        
        # Diabetes Pedigree Function Risk Levels
        # Family history strength indicator
        df_enhanced['pedigree_risk'] = pd.cut(
            df_enhanced['diabetes_pedigree'], 
            bins=[0, 0.2, 0.5, 1.0, float('inf')], 
            labels=[0, 1, 2, 3]  # low, moderate, high, very high family risk
        )
        
        return df_enhanced
    
    def create_ensemble_model(self):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        # Random Forest: Handles feature interactions well, resistant to overfitting
        rf = RandomForestClassifier(
            n_estimators=100,    # Number of trees
            random_state=42      # For reproducible results
        )
        # Gradient Boosting: Sequential learning, good for complex patterns
        gb = GradientBoostingClassifier(
            n_estimators=100,   
            random_state=42
        )
        # Logistic Regression: Linear baseline, fast and interpretable
        lr = LogisticRegression(
            random_state=42,
            max_iter=1000       # Increase iterations for convergence
        )
        # Support Vector Machine: Non-linear decision boundaries
        svm = SVC(
            probability=True,    # Enable probability predictions
            random_state=42
        )
        
        # Voting Classifier: Simple democratic voting
        voting_clf = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr), ('svm', svm)],
            voting='soft'        # Use probability predictions for voting
        )
        
        # Stacking Classifier: Meta-learning approach
        # Uses cross-validation to train meta-estimator
        stacking_clf = StackingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            final_estimator=LogisticRegression(),  # Meta-learner
            cv=5                 # 5-fold cross-validation for meta-features
        )
        
        return voting_clf, stacking_clf
    
    def feature_selection_analysis(self, X, y, k=10):
        # Method 1: SelectKBest with F-statistic
        # Uses ANOVA F-test to measure linear dependency between features and target
        selector_f = SelectKBest(score_func=f_classif, k=k)
        X_selected_f = selector_f.fit_transform(X, y)
        
        # Create feature importance ranking
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': selector_f.scores_
        }).sort_values('score', ascending=False)
        
        print("Top features by F-score:")
        print(feature_scores.head(k))
        
        # Method 2: Recursive Feature Elimination (RFE)
        # Uses model-based feature importance to iteratively eliminate features
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe = RFE(estimator=rf, n_features_to_select=k)
        X_selected_rfe = rfe.fit_transform(X, y)
        
        # Get selected feature names
        selected_features_rfe = X.columns[rfe.support_].tolist()
        print(f"\nTop {k} features by RFE:")
        print(selected_features_rfe)
        
        return selector_f, feature_scores
    
    def advanced_model_training(self, X, y, use_feature_selection=True):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        # Step 1: Split data with stratification
        # Stratification ensures equal class distribution in train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,      # 20% for testing
            random_state=42,    # Reproducible splits
            stratify=y          # Maintain class balance
        )
        
        # Step 2: Feature Selection (if requested)
        if use_feature_selection:
            print("Performing feature selection...")
            selector, _ = self.feature_selection_analysis(
                X_train, y_train, 
                k=min(10, X_train.shape[1])  # Don't select more features than available
            )
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)
            self.feature_selector = selector
            print(f"Selected {X_train_selected.shape[1]} features from {X_train.shape[1]}")
        else:
            X_train_selected = X_train
            X_test_selected = X_test
            print("Using all features (no feature selection)")
        
        # Step 3: Feature Standardization
        # Many ML algorithms perform better with standardized features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        self.scaler = scaler
        print("Features standardized (mean=0, std=1)")
        
        # Step 4: Create Model Collection
        # Combine ensemble and individual models for comparison
        voting_clf, stacking_clf = self.create_ensemble_model()
        
        models = {
            'Voting Classifier': voting_clf,
            'Stacking Classifier': stacking_clf
        }
        
        # Individual models for comparison and analysis
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        
        individual_models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Combine all models for comprehensive evaluation
        models.update(individual_models)
        
        # Step 5: Model Training and Evaluation
        results = {}
        
        print("Training advanced models...")
        print("=" * 60)
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions on test set
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability of diabetes
            
            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Cross-validation for robust performance estimation
            cv_scores = self.stratified_cross_validation(model, X_train_scaled, y_train)
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Display results
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print("-" * 40)
        
        # Step 6: Select Best Model
        # Use F1-score as primary metric (balances precision and recall)
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\nBest Model Selected: {best_model_name}")
        print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
        print(f"AUC: {results[best_model_name]['auc']:.4f}")
        
        # Update training status
        self.is_trained = True
        self.performance_history.append(results)
        
        return results
    
    def stratified_cross_validation(self, model, X, y, cv=5):
        from sklearn.model_selection import cross_val_score
        
        # Stratified K-Fold ensures each fold has similar class distribution
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # F1-score is robust for imbalanced datasets
        f1_scorer = make_scorer(f1_score)
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=skf, scoring=f1_scorer)
        return scores
    
    def hyperparameter_optimization(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import RandomizedSearchCV
        
        # Define parameter search space
        # These ranges are based on common best practices
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],           # Number of trees
            'max_depth': [None, 10, 20, 30, 40],            # Tree depth
            'min_samples_split': [2, 5, 10, 15],            # Min samples to split
            'min_samples_leaf': [1, 2, 4, 8],               # Min samples per leaf
            'max_features': ['sqrt', 'log2', None],          # Features per split
            'bootstrap': [True, False]                       # Bootstrap sampling
        }
        
        # Base model
        rf = RandomForestClassifier(random_state=42)
        
        # Randomized search for efficiency
        # Tests 50 random combinations instead of all possibilities
        random_search = RandomizedSearchCV(
            rf, param_distributions, 
            n_iter=50,              # Number of parameter combinations to try
            cv=5,                   # 5-fold cross-validation
            scoring='f1',           # Optimize for F1-score
            n_jobs=-1,              # Use all available CPU cores
            random_state=42         # Reproducible results
        )
        
        # Perform optimization
        print("Optimizing hyperparameters...")
        random_search.fit(X, y)
        
        # Display results
        print("Best parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best F1 score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def model_interpretation(self, X):
        if not self.is_trained:
            raise ValueError("Model must be trained before interpretation. "
                           "Call advanced_model_training() first.")
        
        # Check if model supports feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature names (handle both DataFrame and array inputs)
            if hasattr(X, 'columns'):
                feature_names = X.columns
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot top 15 features
            top_features = importance_df.head(15)
            bars = plt.barh(range(len(top_features)), top_features['importance'])
            
            # Customize the plot
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance Score')
            plt.title('Top 15 Most Important Health Factors for Diabetes Prediction')
            plt.gca().invert_yaxis()  # Most important at top
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.grid(axis='x', alpha=0.3)
            plt.show()
            
            # Print top 10 features
            print("\nTop 10 Most Important Health Factors:")
            print("-" * 40)
            for i, row in importance_df.head(10).iterrows():
                print(f"{row['feature']:25} {row['importance']:.4f}")
            
            return importance_df
        else:
            print("Model does not support feature importance analysis.")
            print("Consider using tree-based models (Random Forest, Gradient Boosting) "
                  "for feature importance insights.")
            return None
    
    def predict_with_confidence(self, user_input):
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. "
                           "Call advanced_model_training() first.")
        
        # Step 1: Prepare input data
        user_input = np.array(user_input).reshape(1, -1)
        
        # Step 2: Apply feature selection if it was used during training
        if self.feature_selector is not None:
            user_input = self.feature_selector.transform(user_input)
        
        # Step 3: Standardize input using training scaler
        user_input_scaled = self.scaler.transform(user_input)
        
        # Step 4: Make prediction
        prediction = self.best_model.predict(user_input_scaled)[0]
        probabilities = self.best_model.predict_proba(user_input_scaled)[0]
        
        confidence = abs(probabilities[1] - 0.5) * 2
        
        # Step 6: Create comprehensive result
        result = {
            'prediction': int(prediction),
            'prediction_label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability_diabetic': float(probabilities[1]),
            'probability_non_diabetic': float(probabilities[0]),
            'confidence': float(confidence),
            'risk_assessment': self._detailed_risk_assessment(probabilities[1])
        }
        
        return result
    
    def _detailed_risk_assessment(self, prob):

        if prob < 0.1:
            return {
                'level': 'Very Low',
                'description': 'Minimal diabetes risk based on current health factors. '
                              'Your health profile suggests very low likelihood of diabetes.',
                'recommendation': 'Maintain current healthy lifestyle habits. '
                                'Continue regular physical activity and balanced nutrition. '
                                'Annual health check-ups are sufficient.'
            }
        elif prob < 0.3:
            return {
                'level': 'Low',
                'description': 'Below average diabetes risk. Your health indicators '
                              'suggest lower than typical diabetes risk for your demographic.',
                'recommendation': 'Continue current health practices with regular monitoring. '
                                'Maintain healthy weight, stay physically active, and '
                                'schedule routine health screenings every 1-2 years.'
            }
        elif prob < 0.5:
            return {
                'level': 'Moderate',
                'description': 'Moderate diabetes risk detected. Some health factors '
                              'suggest increased attention to diabetes prevention is warranted.',
                'recommendation': 'Consider lifestyle improvements such as increased physical activity, '
                                'dietary modifications, and weight management. '
                                'Consult with healthcare provider for personalized prevention plan.'
            }
        elif prob < 0.7:
            return {
                'level': 'High',
                'description': 'Above average diabetes risk identified. Multiple health factors '
                              'suggest significant risk for developing diabetes.',
                'recommendation': 'Strongly recommend consulting with healthcare provider for '
                                'comprehensive evaluation. Consider diabetes prevention program, '
                                'lifestyle interventions, and more frequent monitoring.'
            }
        else:
            return {
                'level': 'Very High',
                'description': 'High diabetes risk indicated by multiple concerning health factors. '
                              'Immediate attention to diabetes risk management is strongly advised.',
                'recommendation': 'Seek immediate medical consultation for comprehensive diabetes '
                                'screening and evaluation. Urgent lifestyle interventions and '
                                'possible medical management may be necessary.'
            }
    
    def save_model(self, filepath):

        model_data = {
            'best_model': self.best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'is_trained': self.is_trained,
            'performance_history': self.performance_history,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model successfully saved to {filepath}")
        print(f"File size: {round(os.path.getsize(filepath) / 1024 / 1024, 2)} MB")
    
    def load_model(self, filepath):

        try:
            model_data = joblib.load(filepath)
            
            # Restore all components
            self.best_model = model_data['best_model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.is_trained = model_data['is_trained']
            self.performance_history = model_data.get('performance_history', [])
            self.feature_names = model_data.get('feature_names', None)
            
            print(f"Model successfully loaded from {filepath}")
            print(f"Model type: {type(self.best_model).__name__}")
            print(f"Training sessions: {len(self.performance_history)}")
            
        except FileNotFoundError:
            print(f"Error: Model file '{filepath}' not found.")
        except Exception as e:
            print(f"Error loading model: {str(e)}")


    def compare_with_baseline_models(self, X, y):
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
        from sklearn.dummy import DummyClassifier
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        
        if not self.is_trained:
            raise ValueError("Advanced model must be trained first. Call advanced_model_training().")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # Apply same preprocessing as advanced model
        X_train_processed = X_train
        X_test_processed = X_test
        
        if self.feature_selector is not None:
            X_train_processed = self.feature_selector.transform(X_train_processed)
            X_test_processed = self.feature_selector.transform(X_test_processed)
        
        X_train_scaled = self.scaler.transform(X_train_processed)
        X_test_scaled = self.scaler.transform(X_test_processed)
        
        # Define baseline models
        baseline_models = {
            'Dummy (Most Frequent)': DummyClassifier(strategy='most_frequent'),
            'Dummy (Stratified)': DummyClassifier(strategy='stratified'),
            'Simple Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Basic Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
        }
        baseline_models['🏆 Advanced Model (Ours)'] = self.best_model
        
        # Store results
        comparison_results = {}
        for name, model in baseline_models.items():
            if name == '🏆 Advanced Model (Ours)':
                # Our model is already trained
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                # Train baseline model
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Handle models that don't support predict_proba
                try:
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                except AttributeError:
                    y_pred_proba = y_pred.astype(float)  # Fallback for dummy classifiers
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Calculate AUC (handle edge cases)
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                auc = 0.5  # Random performance for constant predictions
            
            # Cross-validation score
            try:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='f1')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = f1
                cv_std = 0
            
            comparison_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
            
            # Print results
            print(f"{name:25} | Acc: {accuracy:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
        
        # Create comparison visualization
        self._create_comparison_plot(comparison_results)
        
        # Print detailed comparison table
        self._print_comparison_table(comparison_results)
        
        # Calculate improvements
        advanced_results = comparison_results['🏆 Advanced Model (Ours)']
        improvements = self._calculate_improvements(comparison_results, advanced_results)
        
        
        return {
            'comparison_results': comparison_results,
            'improvements': improvements,
            'summary': {
                'best_model': '🏆 Advanced Model (Ours)',
                'best_f1_score': advanced_results['f1_score'],
                'best_auc': advanced_results['auc'],
                'total_models_compared': len(comparison_results)
            }
        }


    def compare_with_baseline__models(self, X, y):
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier

        if not self.is_trained:
            raise ValueError("Advanced model must be trained first. Call advanced_model_training().")

        print("🔍 Comparing Advanced Model with Baseline Models")
        print("=" * 60)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_processed = X_train
        X_test_processed = X_test

        if self.feature_selector is not None:
            X_train_processed = self.feature_selector.transform(X_train_processed)
            X_test_processed = self.feature_selector.transform(X_test_processed)

        X_train_scaled = self.scaler.transform(X_train_processed)
        X_test_scaled = self.scaler.transform(X_test_processed)

        baseline_models = {
            'Dummy (Most Frequent)': DummyClassifier(strategy='most_frequent'),
            'Dummy (Stratified)': DummyClassifier(strategy='stratified'),
            'Simple Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Naive Bayes': GaussianNB(),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Basic Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
        }
        baseline_models['🏆 Advanced Model (Ours)'] = self.best_model

        comparison_results = {}
        trained_models = {}
        for name, model in baseline_models.items():
            if name == '🏆 Advanced Model (Ours)':
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                try:
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                except AttributeError:
                    y_pred_proba = y_pred.astype(float)

            trained_models[name] = model

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                auc = 0.5

            try:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='f1')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = f1
                cv_std = 0

            comparison_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }

            print(f"{name:25} | Acc: {accuracy:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

        self._create_comparison_plot(comparison_results)
        self._print_comparison_table(comparison_results)
        advanced_results = comparison_results['🏆 Advanced Model (Ours)']
        improvements = self._calculate_improvements(comparison_results, advanced_results)

        print("\n📊 PERFORMANCE IMPROVEMENTS SUMMARY")
        print("=" * 50)
        print(f"🎯 Our Advanced Model vs Best Baseline:")
        print(f"   • Accuracy improvement: +{improvements['best_accuracy_improvement']:.1%}")
        print(f"   • F1-Score improvement: +{improvements['best_f1_improvement']:.1%}")
        print(f"   • AUC improvement: +{improvements['best_auc_improvement']:.1%}")
        print(f"\n🔥 Our Advanced Model vs Simple Baseline:")
        print(f"   • Accuracy improvement: +{improvements['simple_accuracy_improvement']:.1%}")
        print(f"   • F1-Score improvement: +{improvements['simple_f1_improvement']:.1%}")
        print(f"   • AUC improvement: +{improvements['simple_auc_improvement']:.1%}")

        return {
            'comparison_results': comparison_results,
            'improvements': improvements,
            'summary': {
                'best_model': '🏆 Advanced Model (Ours)',
                'best_f1_score': advanced_results['f1_score'],
                'best_auc': advanced_results['auc'],
                'total_models_compared': len(comparison_results)
            },
            'trained_models': trained_models
        }




if __name__ == '__main__':
    load_or_train_model()


    print(json.dumps(compare_models_infor().get('detailed_results',{})))

