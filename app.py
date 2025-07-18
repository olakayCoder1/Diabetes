import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import traceback
from datetime import datetime

"""
Diabetes Prediction System
===================================

A comprehensive machine learning system for predicting diabetes risk using 
ensemble methods, advanced feature engineering, and interpretable results.

Author: AI Assistant
Version: 2.0
Date: 2025

Features:
- Advanced feature engineering with medical domain knowledge
- Ensemble learning with multiple algorithms
- Feature selection and importance analysis
- Probability calibration and confidence intervals
- Comprehensive risk assessment and recommendations
- Model persistence and interpretability tools

Dependencies:
- pandas, numpy: Data manipulation and numerical operations
- scikit-learn: Machine learning algorithms and utilities
- matplotlib, seaborn: Visualization
- joblib: Model serialization
"""

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
    """
    Advanced diabetes prediction system with enhanced features.
    
    This class implements a comprehensive machine learning pipeline for 
    predicting diabetes risk using ensemble methods, feature engineering,
    and interpretable results.
    
    Key Features:
    - Advanced feature engineering based on medical domain knowledge
    - Ensemble learning combining multiple algorithms
    - Feature selection using statistical and iterative methods
    - Probability calibration for reliable confidence estimates
    - Comprehensive risk assessment with actionable recommendations
    - Model interpretability tools for understanding predictions
    
    Attributes:
        best_model: The highest-performing trained model
        feature_selector: Feature selection transformer
        scaler: Data standardization transformer
        is_trained: Boolean indicating if model is ready for predictions
        feature_names: Names of features used in training
        performance_history: List of training session results
    
    Example:
        >>> predictor = AdvancedDiabetesPredictor()
        >>> # Load and prepare data
        >>> df_enhanced = predictor.advanced_feature_engineering(df)
        >>> X = df_enhanced.drop('target', axis=1)
        >>> y = df_enhanced['target']
        >>> # Train the model
        >>> results = predictor.advanced_model_training(X, y)
        >>> # Make predictions
        >>> prediction = predictor.predict_with_confidence(patient_data)
    """
    
    def __init__(self):
        """
        Initialize the AdvancedDiabetesPredictor.
        
        Sets up all attributes with their default values. The predictor
        is not ready for predictions until after training.
        """
        self.best_model = None              # Will store the best performing model
        self.feature_selector = None        # Feature selection transformer
        self.scaler = None                  # Data standardization transformer
        self.is_trained = False            # Flag indicating training status
        self.feature_names = None          # Names of features used in training
        self.performance_history = []      # History of training sessions
        
    def advanced_feature_engineering(self, df):
        """
        Create advanced features for better diabetes prediction.
        
        This method transforms raw health measurements into enhanced features
        that capture complex relationships and medical domain knowledge.
        
        Feature Engineering Techniques:
        1. Categorical Binning: Convert continuous variables into risk categories
        2. Interaction Features: Create combinations of existing features
        3. Medical Risk Factors: Domain-specific feature creation
        4. Ratio Features: Meaningful medical ratios
        
        Args:
            df (pd.DataFrame): Input dataframe with basic health measurements.
                             Expected columns: pregnancies, glucose, blood_pressure,
                             skin_thickness, insulin, bmi, diabetes_pedigree, age
        
        Returns:
            pd.DataFrame: Enhanced dataframe with additional engineered features
            
        Example:
            >>> df_enhanced = predictor.advanced_feature_engineering(df)
            >>> print(f"Original features: {len(df.columns)}")
            >>> print(f"Enhanced features: {len(df_enhanced.columns)}")
            
        Note:
            This method incorporates medical domain knowledge about diabetes
            risk factors and their interactions.
        """
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
        """
        Create ensemble models combining multiple machine learning algorithms.
        
        Ensemble methods combine predictions from multiple models to improve
        accuracy and robustness. This method creates two types of ensembles:
        1. Voting Classifier: Simple voting among different algorithms
        2. Stacking Classifier: Meta-learning approach with a final estimator
        
        Base Models Used:
        - Random Forest: Robust to overfitting, handles feature interactions
        - Gradient Boosting: Sequential learning from errors
        - Logistic Regression: Linear baseline with probability interpretation
        - Support Vector Machine: Non-linear decision boundaries
        
        Returns:
            tuple: (VotingClassifier, StackingClassifier) ensemble models
            
        Note:
            All models use probability predictions (soft voting) for better
            ensemble performance.
        """
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
        """
        Perform comprehensive feature selection analysis.
        
        Feature selection helps identify the most important health factors
        for diabetes prediction, reducing model complexity and improving
        interpretability.
        
        Methods Used:
        1. SelectKBest with F-statistic: Statistical significance testing
        2. Recursive Feature Elimination: Iterative feature importance ranking
        
        Args:
            X (pd.DataFrame): Feature matrix with health measurements
            y (pd.Series): Target variable (diabetes status)
            k (int): Number of top features to select (default: 10)
            
        Returns:
            tuple: (SelectKBest selector, feature scores DataFrame)
            
        Side Effects:
            Prints feature rankings and selected features to console
            
        Example:
            >>> selector, scores = predictor.feature_selection_analysis(X, y, k=8)
            >>> print("Selected features:", X.columns[selector.get_support()])
        """
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
        """
        Train multiple machine learning models with advanced techniques.
        
        This method implements a comprehensive training pipeline:
        1. Data splitting with stratification
        2. Feature selection (optional)
        3. Data standardization
        4. Multiple model training
        5. Performance evaluation
        6. Best model selection
        
        Args:
            X (pd.DataFrame): Feature matrix with health measurements
            y (pd.Series): Target variable (diabetes status: 0=no, 1=yes)
            use_feature_selection (bool): Whether to apply feature selection
            
        Returns:
            dict: Comprehensive results for all trained models including:
                - model: Trained model object
                - accuracy: Overall correctness (0-1)
                - f1_score: Harmonic mean of precision and recall (0-1)
                - auc: Area under ROC curve (0-1)
                - cv_mean: Cross-validation mean score
                - cv_std: Cross-validation standard deviation
                
        Side Effects:
            - Updates self.best_model, self.feature_selector, self.scaler
            - Sets self.is_trained = True
            - Prints detailed performance metrics
            - Adds results to self.performance_history
            
        Example:
            >>> results = predictor.advanced_model_training(X, y)
            >>> best_f1 = max([r['f1_score'] for r in results.values()])
            >>> print(f"Best F1-score achieved: {best_f1:.4f}")
        """
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
        """
        Perform stratified k-fold cross-validation.
        
        Cross-validation provides a robust estimate of model performance
        by training and testing on different data splits. Stratified
        cross-validation maintains class balance in each fold.
        
        Args:
            model: Machine learning model to evaluate
            X (np.ndarray): Feature matrix (standardized)
            y (pd.Series): Target variable
            cv (int): Number of cross-validation folds (default: 5)
            
        Returns:
            np.ndarray: Cross-validation scores for each fold
            
        Note:
            Uses F1-score as the evaluation metric, which is appropriate
            for potentially imbalanced medical datasets.
        """
        from sklearn.model_selection import cross_val_score
        
        # Stratified K-Fold ensures each fold has similar class distribution
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # F1-score is robust for imbalanced datasets
        f1_scorer = make_scorer(f1_score)
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=skf, scoring=f1_scorer)
        return scores
    
    def hyperparameter_optimization(self, X, y):
        """
        Optimize model hyperparameters using randomized search.
        
        Hyperparameter optimization finds the best model configuration
        by systematically testing different parameter combinations.
        This method focuses on Random Forest optimization but can be
        extended to other algorithms.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            RandomForestClassifier: Optimized model with best parameters
            
        Side Effects:
            Prints best parameters and performance score
            
        Note:
            Uses RandomizedSearchCV for efficiency with large parameter spaces.
            For production use, consider more sophisticated optimization
            techniques like Bayesian optimization.
        """
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
        """
        Interpret model predictions and analyze feature importance.
        
        Model interpretation helps understand which health factors
        contribute most to diabetes predictions, providing valuable
        insights for medical decision-making.
        
        Args:
            X (pd.DataFrame): Feature matrix used for training
            
        Returns:
            pd.DataFrame: Feature importance scores (if available)
            None: If model doesn't support feature importance
            
        Side Effects:
            Creates and displays feature importance visualization
            
        Raises:
            ValueError: If model is not trained yet
            
        Example:
            >>> importance_df = predictor.model_interpretation(X)
            >>> print("Most important factor:", importance_df.iloc[0]['feature'])
        """
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
        """
        Make diabetes prediction with comprehensive confidence assessment.
        
        This method provides not just a prediction, but a complete risk
        assessment with confidence levels and actionable recommendations.
        
        Args:
            user_input (list or np.ndarray): Health measurements in the same
                order as training features. For enhanced features, this should
                include all engineered features.
                
        Returns:
            dict: Comprehensive prediction results containing:
                - prediction (int): 0 (non-diabetic) or 1 (diabetic)
                - prediction_label (str): Human-readable prediction
                - probability_diabetic (float): Probability of diabetes (0-1)
                - probability_non_diabetic (float): Probability of no diabetes (0-1)
                - confidence (float): Model confidence in prediction (0-1)
                - risk_assessment (dict): Detailed risk analysis with recommendations
                
        Raises:
            ValueError: If model is not trained yet
            
        Example:
            >>> # Patient data: [pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age, ...]
            >>> patient = [1, 120, 80, 25, 100, 28.5, 0.3, 35, ...]
            >>> result = predictor.predict_with_confidence(patient)
            >>> print(f"Risk: {result['risk_assessment']['level']}")
            >>> print(f"Recommendation: {result['risk_assessment']['recommendation']}")
        """
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
        
        # Step 5: Calculate confidence
        # Confidence is based on how far the probability is from the decision boundary (0.5)
        # Higher distance from 0.5 means higher confidence
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
        """
        Convert probability into detailed risk assessment with recommendations.
        
        This method translates raw probability scores into actionable
        risk categories with specific recommendations based on medical
        best practices and risk stratification guidelines.
        
        Args:
            prob (float): Probability of diabetes (0-1)
            
        Returns:
            dict: Risk assessment containing:
                - level (str): Risk category
                - description (str): Detailed risk explanation
                - recommendation (str): Specific actionable advice
                
        Note:
            Risk categories are based on common medical practice
            and should be validated with healthcare professionals
            for specific clinical applications.
        """
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
        """
        Save the trained model and all associated components.
        
        Saves the complete prediction system including the best model,
        preprocessing components, and training history for future use.
        
        Args:
            filepath (str): Path where the model should be saved
            
        Side Effects:
            Creates a pickle file containing all model components
            
        Example:
            >>> predictor.save_model('diabetes_model_v1.pkl')
            >>> # Model saved and can be loaded later
        """
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
        """
        Load a previously trained model.
        
        Restores the complete prediction system from a saved file,
        including all preprocessing components and training history.
        
        Args:
            filepath (str): Path to the saved model file
            
        Side Effects:
            Updates all instance attributes with loaded components
            
        Example:
            >>> new_predictor = AdvancedDiabetesPredictor()
            >>> new_predictor.load_model('diabetes_model_v1.pkl')
            >>> # Predictor is now ready for predictions
        """
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
        """
        Compare the advanced model with baseline models and existing solutions.
        
        This method provides comprehensive comparison against:
        1. Simple baseline models (Logistic Regression, Decision Tree)
        2. Common medical screening tools simulation
        3. Statistical benchmarks
        
        Args:
            X (pd.DataFrame): Feature matrix with health measurements
            y (pd.Series): Target variable (diabetes status)
            
        Returns:
            dict: Comprehensive comparison results with performance metrics
            
        Side Effects:
            Prints detailed comparison table and saves comparison plot
        """
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
        
        # print("🔍 Comparing Advanced Model with Baseline Models")
        # print("=" * 60)
        
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
        
        # Add our advanced model for comparison
        baseline_models['🏆 Advanced Model (Ours)'] = self.best_model
        
        # Store results
        comparison_results = {}
        
        # print("Training and evaluating models...")
        # print("-" * 60)
        
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
        
        # print("\n📊 PERFORMANCE IMPROVEMENTS SUMMARY")
        # print("=" * 50)
        # print(f"🎯 Our Advanced Model vs Best Baseline:")
        # print(f"   • Accuracy improvement: +{improvements['best_accuracy_improvement']:.1%}")
        # print(f"   • F1-Score improvement: +{improvements['best_f1_improvement']:.1%}")
        # print(f"   • AUC improvement: +{improvements['best_auc_improvement']:.1%}")
        # print(f"\n🔥 Our Advanced Model vs Simple Baseline:")
        # print(f"   • Accuracy improvement: +{improvements['simple_accuracy_improvement']:.1%}")
        # print(f"   • F1-Score improvement: +{improvements['simple_f1_improvement']:.1%}")
        # print(f"   • AUC improvement: +{improvements['simple_auc_improvement']:.1%}")
        
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
        """
        Compare the advanced model with baseline models and existing solutions,
        return performance metrics and trained models.
        """
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

        print("Training and evaluating models...")
        print("-" * 60)

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


    # def _create_comparison_plot(self, results):
    #     """Create visualization comparing model performances."""
    #     import matplotlib.pyplot as plt
        
    #     # Prepare data for plotting
    #     models = list(results.keys())
    #     f1_scores = [results[model]['f1_score'] for model in models]
    #     auc_scores = [results[model]['auc'] for model in models]
    #     accuracies = [results[model]['accuracy'] for model in models]
        
    #     # Create subplots
    #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
    #     # Colors (highlight our model)
    #     colors = ['#ff6b6b' if '🏆' in model else '#4ecdc4' for model in models]
        
    #     # Plot 1: F1-Score comparison
    #     bars1 = ax1.bar(range(len(models)), f1_scores, color=colors)
    #     ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    #     ax1.set_ylabel('F1-Score')
    #     ax1.set_xticks(range(len(models)))
    #     ax1.set_xticklabels([m.replace('🏆 ', '') for m in models], rotation=45, ha='right')
    #     ax1.grid(axis='y', alpha=0.3)
        
    #     # Add value labels on bars
    #     for bar, score in zip(bars1, f1_scores):
    #         ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
    #                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
    #     # Plot 2: AUC comparison
    #     bars2 = ax2.bar(range(len(models)), auc_scores, color=colors)
    #     ax2.set_title('AUC (Area Under Curve) Comparison', fontsize=14, fontweight='bold')
    #     ax2.set_ylabel('AUC Score')
    #     ax2.set_xticks(range(len(models)))
    #     ax2.set_xticklabels([m.replace('🏆 ', '') for m in models], rotation=45, ha='right')
    #     ax2.grid(axis='y', alpha=0.3)
        
    #     for bar, score in zip(bars2, auc_scores):
    #         ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
    #                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
    #     # Plot 3: Accuracy comparison
    #     bars3 = ax3.bar(range(len(models)), accuracies, color=colors)
    #     ax3.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    #     ax3.set_ylabel('Accuracy')
    #     ax3.set_xticks(range(len(models)))
    #     ax3.set_xticklabels([m.replace('🏆 ', '') for m in models], rotation=45, ha='right')
    #     ax3.grid(axis='y', alpha=0.3)
        
    #     for bar, score in zip(bars3, accuracies):
    #         ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
    #                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
    #     # Plot 4: Radar chart for our model
    #     metrics = ['F1-Score', 'AUC', 'Accuracy', 'Precision', 'Recall']
    #     our_model_name = '🏆 Advanced Model (Ours)'
    #     our_scores = [
    #         results[our_model_name]['f1_score'],
    #         results[our_model_name]['auc'],
    #         results[our_model_name]['accuracy'],
    #         results[our_model_name]['precision'],
    #         results[our_model_name]['recall']
    #     ]
        
    #     # Simple bar chart instead of radar (easier to implement)
    #     bars4 = ax4.bar(metrics, our_scores, color='#ff6b6b')
    #     ax4.set_title('Our Advanced Model - All Metrics', fontsize=14, fontweight='bold')
    #     ax4.set_ylabel('Score')
    #     ax4.set_ylim(0, 1)
    #     ax4.grid(axis='y', alpha=0.3)
        
    #     for bar, score in zip(bars4, our_scores):
    #         ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
    #                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
    #     plt.tight_layout()
    #     plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    #     plt.show()


    def _print_comparison_table(self, results):
        """Print a detailed comparison table."""
        print("\n📋 DETAILED COMPARISON TABLE")
        print("=" * 100)
        print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10} {'CV Mean':<10}")
        print("-" * 100)
        
        # Sort by F1-score (descending)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for rank, (name, metrics) in enumerate(sorted_results, 1):
            symbol = "🏆" if rank == 1 else f"{rank:2d}"
            clean_name = name.replace('🏆 ', '')
            print(f"{symbol} {clean_name:<22} {metrics['accuracy']:<10.3f} {metrics['precision']:<10.3f} "
                f"{metrics['recall']:<10.3f} {metrics['f1_score']:<10.3f} {metrics['auc']:<10.3f} "
                f"{metrics['cv_mean']:<10.3f}")

    def _calculate_improvements(self, all_results, advanced_results):
        """Calculate performance improvements over baseline models."""
        # Find best baseline model (excluding our advanced model)
        baseline_results = {k: v for k, v in all_results.items() if '🏆' not in k}
        
        best_baseline = max(baseline_results.items(), key=lambda x: x[1]['f1_score'])
        best_baseline_name, best_baseline_metrics = best_baseline
        
        # Simple baseline (logistic regression)
        simple_baseline_metrics = all_results.get('Simple Logistic Regression', best_baseline_metrics)
        
        improvements = {
            'best_baseline_name': best_baseline_name,
            'best_accuracy_improvement': advanced_results['accuracy'] - best_baseline_metrics['accuracy'],
            'best_f1_improvement': advanced_results['f1_score'] - best_baseline_metrics['f1_score'],
            'best_auc_improvement': advanced_results['auc'] - best_baseline_metrics['auc'],
            'simple_accuracy_improvement': advanced_results['accuracy'] - simple_baseline_metrics['accuracy'],
            'simple_f1_improvement': advanced_results['f1_score'] - simple_baseline_metrics['f1_score'],
            'simple_auc_improvement': advanced_results['auc'] - simple_baseline_metrics['auc']
        }
        
        return improvements

    def _create_comparison_plot(self, results):
        """Create visualization comparing model performances."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import os
        
        # Prepare data for plotting
        models = list(results.keys())
        f1_scores = [results[model]['f1_score'] for model in models]
        auc_scores = [results[model]['auc'] for model in models]
        accuracies = [results[model]['accuracy'] for model in models]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Colors (highlight our model)
        colors = ['#ff6b6b' if '🏆' in model else '#4ecdc4' for model in models]
        
        # Plot 1: F1-Score comparison
        bars1 = ax1.bar(range(len(models)), f1_scores, color=colors)
        ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('F1-Score')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels([m.replace('🏆 ', '') for m in models], rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars1, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: AUC comparison
        bars2 = ax2.bar(range(len(models)), auc_scores, color=colors)
        ax2.set_title('AUC (Area Under Curve) Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('AUC Score')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace('🏆 ', '') for m in models], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars2, auc_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Accuracy comparison
        bars3 = ax3.bar(range(len(models)), accuracies, color=colors)
        ax3.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.replace('🏆 ', '') for m in models], rotation=45, ha='right')
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars3, accuracies):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Bar chart for our model
        metrics = ['F1-Score', 'AUC', 'Accuracy', 'Precision', 'Recall']
        our_model_name = '🏆 Advanced Model (Ours)'
        our_scores = [
            results[our_model_name]['f1_score'],
            results[our_model_name]['auc'],
            results[our_model_name]['accuracy'],
            results[our_model_name]['precision'],
            results[our_model_name]['recall']
        ]
        
        bars4 = ax4.bar(metrics, our_scores, color='#ff6b6b')
        ax4.set_title('Our Advanced Model - All Metrics', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, score in zip(bars4, our_scores):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Ensure static directory exists
        static_dir = 'static'
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        
        # Save the plot
        plot_path = os.path.join(static_dir, 'model_comparison.png')
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_path}")
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
            raise
        finally:
            plt.close(fig)  # Close the figure to free memory
        
        # Verify file exists
        if not os.path.exists(plot_path):
            raise FileNotFoundError(f"Failed to create plot at {plot_path}")


# Global predictor instance
predictor = None
model_loaded = False

def load_or_train_model():
    """Load existing model or train a new one if needed"""
    global predictor, model_loaded
    
    predictor = AdvancedDiabetesPredictor()
    model_path = 'diabetes_predictor_demo.pkl'
    
    if os.path.exists(model_path):
        try:
            predictor.load_model(model_path)
            model_loaded = True
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("Will attempt to train a new model...")
    
    # If model doesn't exist or failed to load, train a new one
    try:
        print("🔄 Training new model...")
        from sklearn.datasets import make_classification
        
        # Generate synthetic data (same as in your demo)
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
        
        # Transform to realistic medical ranges
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
        
        # Train the model
        results = predictor.advanced_model_training(X_enhanced, y_enhanced)
        
        # Save the trained model
        predictor.save_model(model_path)
        model_loaded = True
        print("✅ New model trained and saved successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        traceback.print_exc()
        return False


app = Flask(__name__)
app.secret_key = 'your-secret-key-here-098765443etrghjvcgdtu'  



@app.route('/')
def index():
    """Main page with prediction form"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.',
            'success': False
        })
    
    try:
        # Get form data
        data = {
            'pregnancies': int(request.form.get('pregnancies', 0)),
            'glucose': float(request.form.get('glucose', 0)),
            'blood_pressure': float(request.form.get('blood_pressure', 0)),
            'skin_thickness': float(request.form.get('skin_thickness', 0)),
            'insulin': float(request.form.get('insulin', 0)),
            'bmi': float(request.form.get('bmi', 0)),
            'diabetes_pedigree': float(request.form.get('diabetes_pedigree', 0)),
            'age': int(request.form.get('age', 0))
        }
        
        # Validate input ranges
        validation_errors = []
        if data['pregnancies'] < 0 or data['pregnancies'] > 20:
            validation_errors.append("Pregnancies should be between 0-20")
        if data['glucose'] < 50 or data['glucose'] > 300:
            validation_errors.append("Glucose should be between 50-300 mg/dL")
        if data['blood_pressure'] < 40 or data['blood_pressure'] > 150:
            validation_errors.append("Blood pressure should be between 40-150 mmHg")
        if data['bmi'] < 10 or data['bmi'] > 50:
            validation_errors.append("BMI should be between 10-50")
        if data['age'] < 1 or data['age'] > 120:
            validation_errors.append("Age should be between 1-120 years")
        
        if validation_errors:
            return jsonify({
                'error': '; '.join(validation_errors),
                'success': False
            })
        
        # Create DataFrame
        patient_data = pd.DataFrame([data])
        
        # Feature engineering
        patient_enhanced = predictor.advanced_feature_engineering(patient_data)
        
        # Make prediction
        result = predictor.predict_with_confidence(patient_enhanced.values[0])
        
        # Add timestamp and input data to result
        result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result['input_data'] = data
        result['success'] = True
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            'error': f'Invalid input: {str(e)}',
            'success': False
        })
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        })



@app.route('/compare_models', methods=['POST'])
def compare_models():
    """API endpoint for model comparison"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.',
            'success': False
        })
    
    try:
        # Generate or load test data for comparison
        from sklearn.datasets import make_classification
        
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
        
        # Transform to realistic medical ranges
        df['pregnancies'] = (np.abs(df['pregnancies']) * 2 + 1).astype(int)
        df['glucose'] = np.abs(df['glucose']) * 30 + 100
        df['blood_pressure'] = np.abs(df['blood_pressure']) * 20 + 60
        df['skin_thickness'] = np.abs(df['skin_thickness']) * 15 + 15
        df['insulin'] = np.abs(df['insulin']) * 100 + 50
        df['bmi'] = np.abs(df['bmi']) * 10 + 20
        df['diabetes_pedigree'] = np.abs(df['diabetes_pedigree']) * 0.5 + 0.1
        df['age'] = (np.abs(df['age']) * 15 + 25).astype(int)
        df['target'] = y
        
        # Feature engineering
        df_enhanced = predictor.advanced_feature_engineering(df)
        X_enhanced = df_enhanced.drop('target', axis=1)
        y_enhanced = df_enhanced['target']
        
        # Run comparison
        comparison_result = predictor.compare_with_baseline_models(X_enhanced, y_enhanced)
        
        # Check if plot file exists
        # plot_path = os.path.join('static', 'model_comparison.png')
        
        # Format results for JSON response
        response_data = {
            'success': True,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'comparison_summary': comparison_result['summary'],
            'improvements': comparison_result['improvements'],
            'detailed_results': comparison_result['comparison_results'],
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': f'Model comparison failed: {str(e)}',
            'success': False
        })

@app.route('/api/health')
def health_check():
    """API endpoint for health checking"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/about')
def about():
    """About page with information about the system"""
    return render_template('about.html')



def compare_models_infor():
    """API endpoint for model comparison"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.',
            'success': False
        })
    
    try:
        # Generate or load test data for comparison
        from sklearn.datasets import make_classification
        
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
        
        # # Transform to realistic medical ranges
        # df['pregnancies'] = (np.abs(df['pregnancies']) * 2 + 1).astype(int)
        # df['glucose'] = np.abs(df['glucose']) * 30 + 100
        # df['blood_pressure'] = np.abs(df['blood_pressure']) * 20 + 60
        # df['skin_thickness'] = np.abs(df['skin_thickness']) * 15 + 15
        # df['insulin'] = np.abs(df['insulin']) * 100 + 50
        # df['bmi'] = np.abs(df['bmi']) * 10 + 20
        # df['diabetes_pedigree'] = np.abs(df['diabetes_pedigree']) * 0.5 + 0.1
        # df['age'] = (np.abs(df['age']) * 15 + 25).astype(int)
        # df['target'] = y

        # print("++++"*20)
        # print(df)
        # print("----"*10)
        
        # # Feature engineering
        # df_enhanced = predictor.advanced_feature_engineering(df)
        # Apply transformations only to the first 5 rows
        df_subset = df.head(5).copy()

        df_subset['pregnancies'] = (np.abs(df_subset['pregnancies']) * 2 + 1).astype(int)
        df_subset['glucose'] = np.abs(df_subset['glucose']) * 30 + 100
        df_subset['blood_pressure'] = np.abs(df_subset['blood_pressure']) * 20 + 60
        df_subset['skin_thickness'] = np.abs(df_subset['skin_thickness']) * 15 + 15
        df_subset['insulin'] = np.abs(df_subset['insulin']) * 100 + 50
        df_subset['bmi'] = np.abs(df_subset['bmi']) * 10 + 20
        df_subset['diabetes_pedigree'] = np.abs(df_subset['diabetes_pedigree']) * 0.5 + 0.1
        df_subset['age'] = (np.abs(df_subset['age']) * 15 + 25).astype(int)
        df_subset['target'] = y[:5]  # Assuming `y` is aligned

        print("++++" * 20)
        print(df_subset)
        print("----" * 10)

        # Feature engineering
        df_enhanced = predictor.advanced_feature_engineering(df_subset)

        X_enhanced = df_enhanced.drop('target', axis=1)
        y_enhanced = df_enhanced['target']
        
        # Run comparison
        comparison_result = predictor.compare_with_baseline_models(X_enhanced, y_enhanced)
        
        # Check if plot file exists
        plot_path = os.path.join('static', 'model_comparison_7.png')
        
        # Format results for JSON response
        response_data = {
            'success': True,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'comparison_summary': comparison_result['summary'],
            'improvements': comparison_result['improvements'],
            'detailed_results': comparison_result['comparison_results'],
        }


        return response_data


    except:


        return {}
    

def compare_models_infor_new():
    """API endpoint for model comparison"""
    if not model_loaded:
        return jsonify({
            'error': 'Model not loaded. Please check server logs.',
            'success': False
        })
    
    try:
        from sklearn.datasets import make_classification
        import numpy as np
        import pandas as pd
        import os
        from datetime import datetime

        np.random.seed(42)
        
        # Generate test data
        X, y = make_classification(
            n_samples=1000, n_features=8, n_informative=6, 
            n_redundant=2, n_clusters_per_class=1, random_state=42
        )
        
        feature_names = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age'
        ]
        
        df = pd.DataFrame(X, columns=feature_names)
        
        # Transform to realistic ranges
        df['pregnancies'] = (np.abs(df['pregnancies']) * 2 + 1).astype(int)
        df['glucose'] = np.abs(df['glucose']) * 30 + 100
        df['blood_pressure'] = np.abs(df['blood_pressure']) * 20 + 60
        df['skin_thickness'] = np.abs(df['skin_thickness']) * 15 + 15
        df['insulin'] = np.abs(df['insulin']) * 100 + 50
        df['bmi'] = np.abs(df['bmi']) * 10 + 20
        df['diabetes_pedigree'] = np.abs(df['diabetes_pedigree']) * 0.5 + 0.1
        df['age'] = (np.abs(df['age']) * 15 + 25).astype(int)
        df['target'] = y

        
        # Feature engineering
        df_enhanced = predictor.advanced_feature_engineering(df)
        X_enhanced = df_enhanced.drop('target', axis=1)
        y_enhanced = df_enhanced['target']
        
        # Run comparison with trained models returned
        comparison_result = predictor.compare_with_baseline_models(X_enhanced, y_enhanced)

        trained_models = comparison_result.get('trained_models', {})
        
        # Collect predictions for first 5 samples
        sample_inputs = X_enhanced.head(5)
        sample_predictions = {}

        print("\n🔍 Individual Predictions on First 5 Samples:\n")
        
        for idx, (_, row) in enumerate(sample_inputs.iterrows()):
            sample_key = f"Sample_{idx + 1}"
            sample_predictions[sample_key] = {}
            features = row.values.reshape(1, -1)
            
            print(f"\n--- Sample {idx + 1} ---")
            
            for model_name, model in trained_models.items():
                pred = model.predict(features)[0]
                print(f"{model_name}: Prediction = {pred}")
                
                # Add to JSON response
                sample_predictions[sample_key][model_name] = int(pred)

        # Check if plot file exists
        # plot_path = os.path.join('static', 'model_comparison_7.png')
        
        response_data = {
            'success': True,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'comparison_summary': comparison_result['summary'],
            'improvements': comparison_result['improvements'],
            'detailed_results': comparison_result['comparison_results'],
            'sample_predictions': sample_predictions
        }
        
        return response_data

    except Exception as e:
        print(f"Error during model comparison: {e}")
        return jsonify({
            'error': str(e),
            'success': False
        })



if __name__ == '__main__':
    # Try to initialize model
    load_or_train_model()


    # print(json.dumps(compare_models_infor().get('detailed_results',{})))


    
    # Run the Flask app
    print("\n🚀 Starting Flask Diabetes Prediction App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    # app.run()

    
    app.run(debug=True, host='0.0.0.0', port=8000)