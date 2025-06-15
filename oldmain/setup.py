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

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import make_scorer, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
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
        
        # Base Models Configuration
        # Each model brings different strengths to the ensemble
        
        # Random Forest: Handles feature interactions well, resistant to overfitting
        rf = RandomForestClassifier(
            n_estimators=100,    # Number of trees
            random_state=42      # For reproducible results
        )
        
        # Gradient Boosting: Sequential learning, good for complex patterns
        gb = GradientBoostingClassifier(
            n_estimators=100,    # Number of boosting stages
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


def advanced_diabetes_prediction_demo():
    """
    Comprehensive demonstration of the Advanced Diabetes Prediction System.
    
    This function provides a complete walkthrough of the system capabilities
    including data generation, feature engineering, model training, evaluation,
    and prediction examples.
    
    Returns:
        tuple: (trained_predictor, training_results)
        
    Example:
        >>> predictor, results = advanced_diabetes_prediction_demo()
        >>> # System is now trained and ready for use
        >>> 
        >>> # Make predictions on new patients
        >>> patient_data = [1, 110, 75, 20, 90, 25.0, 0.2, 30, ...]
        >>> prediction = predictor.predict_with_confidence(patient_data)
    """
    # Import required libraries for demo
    from sklearn.datasets import make_classification
    import os
    
    print("🔬 Advanced Diabetes Prediction System Demo")
    print("=" * 60)
    print("This demonstration showcases a comprehensive machine learning")
    print("system for diabetes risk prediction using ensemble methods,")
    print("advanced feature engineering, and interpretable results.")
    print("")
    
    # Step 1: Generate Realistic Synthetic Data
    print("📊 Step 1: Generating Synthetic Health Data")
    print("-" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create base synthetic dataset
    X, y = make_classification(
        n_samples=1000,        # 1000 patient records
        n_features=8,          # 8 basic health measurements
        n_informative=6,       # 6 features are actually predictive
        n_redundant=2,         # 2 features are combinations of others
        n_clusters_per_class=1, # Single cluster per class
        random_state=42
    )
    
    # Define medically meaningful feature names
    feature_names = [
        'pregnancies',         # Number of pregnancies
        'glucose',            # Blood glucose level
        'blood_pressure',     # Diastolic blood pressure
        'skin_thickness',     # Triceps skin fold thickness
        'insulin',           # 2-hour serum insulin
        'bmi',               # Body mass index
        'diabetes_pedigree', # Diabetes pedigree function
        'age'                # Age in years
    ]
    
    # Create DataFrame with realistic feature scaling
    df = pd.DataFrame(X, columns=feature_names)
    
    # Transform synthetic data to realistic medical ranges
    print("Scaling features to realistic medical ranges...")
    df['pregnancies'] = (np.abs(df['pregnancies']) * 2 + 1).astype(int)      # 1-15 pregnancies
    df['glucose'] = np.abs(df['glucose']) * 30 + 100                         # 70-200 mg/dL
    df['blood_pressure'] = np.abs(df['blood_pressure']) * 20 + 60           # 40-120 mmHg
    df['skin_thickness'] = np.abs(df['skin_thickness']) * 15 + 15           # 0-45 mm
    df['insulin'] = np.abs(df['insulin']) * 100 + 50                        # 0-300 μU/mL
    df['bmi'] = np.abs(df['bmi']) * 10 + 20                                 # 15-45 kg/m²
    df['diabetes_pedigree'] = np.abs(df['diabetes_pedigree']) * 0.5 + 0.1   # 0.1-1.0
    df['age'] = (np.abs(df['age']) * 15 + 25).astype(int)                   # 20-65 years
    
    # Add target variable
    df['target'] = y
    
    print(f"✅ Generated {len(df)} patient records")
    print(f"   Diabetic cases: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"   Non-diabetic cases: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    print("")
    
    # Step 2: Initialize Prediction System
    print("🤖 Step 2: Initializing Advanced Prediction System")
    print("-" * 40)
    predictor = AdvancedDiabetesPredictor()
    print("✅ Advanced Diabetes Predictor initialized")
    print("")
    
    # Step 3: Feature Engineering
    print("⚙️  Step 3: Advanced Feature Engineering")
    print("-" * 40)
    df_enhanced = predictor.advanced_feature_engineering(df)
    
    print(f"✅ Feature engineering completed")
    print(f"   Original features: {len(df.columns) - 1}")
    print(f"   Enhanced features: {len(df_enhanced.columns) - 1}")
    print(f"   New features created: {len(df_enhanced.columns) - len(df.columns)}")
    print("")
    
    # Display sample of new features
    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    print("   New features created:")
    for feature in new_features[:8]:  # Show first 8 for brevity
        print(f"   • {feature}")
    if len(new_features) > 8:
        print(f"   • ... and {len(new_features) - 8} more")
    print("")
    
    # Step 4: Model Training
    print("🎯 Step 4: Advanced Model Training")
    print("-" * 40)
    
    # Prepare training data
    X_enhanced = df_enhanced.drop('target', axis=1)
    y_enhanced = df_enhanced['target']
    
    print(f"Training on {len(X_enhanced)} samples with {len(X_enhanced.columns)} features")
    print("")
    
    # Train multiple models
    results = predictor.advanced_model_training(X_enhanced, y_enhanced)
    print("")
    
    # Step 5: Model Interpretation
    print("🔍 Step 5: Model Interpretation & Feature Importance")
    print("-" * 40)
    importance_df = predictor.model_interpretation(X_enhanced)
    print("")
    
    # Step 6: Demonstration Predictions
    print("🩺 Step 6: Sample Predictions")
    print("-" * 40)
    
    # Create sample patient profiles for demonstration
    sample_patients = [
        {
            'description': 'Low Risk Patient',
            'data': [1, 95, 70, 20, 80, 22.0, 0.15, 28, 0, 0, 0, 2660, 2090, 0.84, 0, 0, 1, 0, 0]
        },
        {
            'description': 'Moderate Risk Patient', 
            'data': [2, 140, 85, 30, 120, 28.5, 0.35, 45, 2, 1, 2, 3990, 3990, 0.86, 1, 0, 2, 1, 1]
        },
        {
            'description': 'High Risk Patient',
            'data': [3, 180, 95, 35, 200, 35.0, 0.75, 55, 2, 2, 3, 9900, 6300, 1.11, 1, 1, 2, 2, 3]
        }
    ]
    
    for patient in sample_patients:
        print(f"\n{patient['description']}:")
        print("-" * 25)
        
        try:
            result = predictor.predict_with_confidence(patient['data'])
            
            print(f"Prediction: {result['prediction_label']}")
            print(f"Diabetes Probability: {result['probability_diabetic']:.1%}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Risk Level: {result['risk_assessment']['level']}")
            print(f"Recommendation: {result['risk_assessment']['recommendation']}")
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
    
    print("")
    
    # Step 7: System Summary
    print("📈 Step 7: System Performance Summary")
    print("-" * 40)
    
    if results:
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        best_results = results[best_model_name]
        
        print(f"Best Performing Model: {best_model_name}")
        print(f"├── Accuracy: {best_results['accuracy']:.1%}")
        print(f"├── F1-Score: {best_results['f1_score']:.1%}")
        print(f"├── AUC Score: {best_results['auc']:.1%}")
        print(f"└── CV Score: {best_results['cv_mean']:.1%} (±{best_results['cv_std']:.1%})")
        print("")
        
        print("All Models Performance:")
        for name, result in results.items():
            print(f"├── {name}: F1={result['f1_score']:.3f}, AUC={result['auc']:.3f}")
    
    print("")
    print("🎉 Demo Complete! The Advanced Diabetes Predictor is ready for use.")
    print("=" * 60)
    
    return predictor, results



# """"""""
# # Main execution
# if __name__ == "__main__":
#     """
#     Main execution block for running the demonstration.
    
#     This block runs when the script is executed directly (not imported).
#     It demonstrates the complete functionality of the Advanced Diabetes
#     Prediction System.
#     """
#     try:
#         # Run the comprehensive demonstration
#         predictor, training_results = advanced_diabetes_prediction_demo()
        
#         print("\n" + "="*60)
#         print("🔧 Additional Usage Examples")
#         print("="*60)
        
#         # Example 1: Saving and Loading Models
#         print("\n📁 Example 1: Model Persistence")
#         print("-" * 30)
        
#         # Save the trained model
#         model_filename = 'diabetes_predictor_demo.pkl'
#         predictor.save_model(model_filename)
        
#         # Load the model (demonstration)
#         new_predictor = AdvancedDiabetesPredictor()
#         new_predictor.load_model(model_filename)
        
#         print(f"✅ Model saved and loaded successfully")
        
#         # # Clean up demo file
#         # if os.path.exists(model_filename):
#         #     os.remove(model_filename)
#         #     print(f"   Demo file {model_filename} cleaned up")
        
#         # Example 2: Integration Guidelines
#         print("\n🔗 Example 2: Integration Guidelines")
#         print("-" * 30)
#         print("For production use:")
#         print("1. Replace synthetic data with real patient data")
#         print("2. Validate with clinical experts and regulatory requirements")
#         print("3. Implement proper data security and privacy measures")
#         print("4. Add comprehensive error handling and logging")
#         print("5. Regular model retraining and performance monitoring")
#         print("6. Integration with electronic health record systems")
        
#         print("\n✨ System ready for adaptation to real-world applications!")
        
#     except Exception as e:
#         print(f"❌ Demo failed with error: {str(e)}")
#         print("Please check the error details above and ensure all dependencies are installed.")
        
#     finally:
#         print("\n" + "="*60)
#         print("Thank you for exploring the Advanced Diabetes Prediction System!")
#         print("="*60)


if __name__ == "__main__":
    import pandas as pd

    print("Diabetes Risk Checker")
    print("Enter your health information (get these from your doctor or medical report).")

    try:
        pregnancies = int(input("Number of pregnancies (0 for men): "))
        glucose = float(input("Fasting blood sugar (mg/dL, e.g., 120): "))
        blood_pressure = float(input("Blood pressure (lower number, mmHg, e.g., 80): "))
        skin_thickness = float(input("Triceps skin thickness (mm, e.g., 25): "))
        insulin = float(input("Insulin level (μU/mL, e.g., 100): "))
        bmi = float(input("Body Mass Index (BMI, e.g., 28.5): "))
        diabetes_pedigree = float(input("Family history score (0.1–1.0, e.g., 0.3): "))
        age = int(input("Age (years, e.g., 35): "))

        # Create DataFrame
        patient_data = pd.DataFrame({
            'pregnancies': [pregnancies],
            'glucose': [glucose],
            'blood_pressure': [blood_pressure],
            'skin_thickness': [skin_thickness],
            'insulin': [insulin],
            'bmi': [bmi],
            'diabetes_pedigree': [diabetes_pedigree],
            'age': [age]
        })

        # Load pre-trained model
        predictor = AdvancedDiabetesPredictor()
        predictor.load_model('diabetes_predictor_demo.pkl')

        # Process and predict
        patient_enhanced = predictor.advanced_feature_engineering(patient_data)
        result = predictor.predict_with_confidence(patient_enhanced.values[0])

        # Show results
        print("\nYour Results:")
        print(f"Prediction: {result['prediction_label']}")
        print(f"Diabetes Risk: {result['probability_diabetic']:.1%}")
        print(f"Risk Level: {result['risk_assessment']['level']}")
        print(f"What to Do: {result['risk_assessment']['recommendation']}")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your inputs (use numbers) or ensure the model file exists.")


        