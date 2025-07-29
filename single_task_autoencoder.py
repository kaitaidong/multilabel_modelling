"""
Simplified Multi-Label Classification for Field Changes Prediction
================================================================

This script focuses exclusively on multi-label classification for predicting
field changes in cash reconciliation, implementing both:
- Option A: Independent Binary Classifiers (one model per field)
- Option B: Single Multi-Label Classifier (one model for all fields)

Includes comprehensive model exploration, hyperparameter tuning, and comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import (hamming_loss, accuracy_score, precision_recall_fscore_support, 
                           jaccard_score, classification_report, multilabel_confusion_matrix)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class MultiLabelFieldChangesBaseline:
    """
    Simplified baseline focused on multi-label field changes prediction
    """
    
    def __init__(self, field_names, random_state=42):
        self.field_names = field_names
        self.n_fields = len(field_names)
        self.random_state = random_state
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        
        # Model collections
        self.base_models = self._initialize_base_models()
        
        # Results storage
        self.results = {
            'option_a_independent': {},
            'option_b_multilabel': {},
            'neural_networks': {}
        }
        
    def _initialize_base_models(self):
        """
        Initialize base models for multi-label classification
        """
        return {
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state, n_estimators=100),
            'xgboost': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
            'svm_rbf': SVC(kernel='rbf', probability=True, random_state=self.random_state),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'decision_tree': DecisionTreeClassifier(random_state=self.random_state),
            'extra_trees': ExtraTreesClassifier(random_state=self.random_state, n_estimators=100)
        }
    
    def _get_hyperparameter_grids(self):
        """
        Hyperparameter grids optimized for multi-label tasks
        """
        return {
            'logistic_regression': {
                'estimator__C': [0.1, 1, 10],
                'estimator__penalty': ['l1', 'l2'],
                'estimator__solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__max_depth': [5, 10, None],
                'estimator__min_samples_split': [2, 5],
                'estimator__min_samples_leaf': [1, 2]
            },
            'xgboost': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__learning_rate': [0.01, 0.1, 0.2],
                'estimator__max_depth': [3, 5, 7],
                'estimator__subsample': [0.8, 1.0]
            },
            'lightgbm': {
                'estimator__n_estimators': [50, 100, 200],
                'estimator__learning_rate': [0.01, 0.1, 0.2],
                'estimator__max_depth': [3, 5, 7],
                'estimator__subsample': [0.8, 1.0]
            }
        }
    
    def preprocess_data(self, X, fit_transform=True):
        """
        Preprocess data for multi-label classification
        """
        X_processed = X.copy()
        
        # Identify feature types
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle categorical features
        for col in categorical_features:
            if col in X_processed.columns:
                if fit_transform:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        X_processed[col] = self.label_encoders[col].fit_transform(
                            X_processed[col].astype(str).fillna('missing')
                        )
                    else:
                        X_processed[col] = self.label_encoders[col].transform(
                            X_processed[col].astype(str).fillna('missing')
                        )
                else:
                    X_processed[col] = self.label_encoders[col].transform(
                        X_processed[col].astype(str).fillna('missing')
                    )
        
        # Handle numerical features
        if numerical_features:
            if fit_transform:
                X_processed[numerical_features] = self.imputer_num.fit_transform(X_processed[numerical_features])
                X_processed[numerical_features] = self.scaler.fit_transform(X_processed[numerical_features])
            else:
                X_processed[numerical_features] = self.imputer_num.transform(X_processed[numerical_features])
                X_processed[numerical_features] = self.scaler.transform(X_processed[numerical_features])
        
        return X_processed
    
    def calculate_multilabel_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate comprehensive multi-label metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        metrics['exact_match_ratio'] = accuracy_score(y_true, y_pred)
        metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average='samples')
        
        # Per-field metrics
        field_metrics = {}
        for i, field_name in enumerate(self.field_names):
            field_metrics[field_name] = {
                'accuracy': accuracy_score(y_true[:, i], y_pred[:, i]),
                'precision': precision_recall_fscore_support(y_true[:, i], y_pred[:, i], average='binary')[0],
                'recall': precision_recall_fscore_support(y_true[:, i], y_pred[:, i], average='binary')[1],
                'f1_score': precision_recall_fscore_support(y_true[:, i], y_pred[:, i], average='binary')[2]
            }
        
        metrics['field_metrics'] = field_metrics
        
        # Average metrics across fields
        metrics['macro_accuracy'] = np.mean([field_metrics[field]['accuracy'] for field in self.field_names])
        metrics['macro_precision'] = np.mean([field_metrics[field]['precision'] for field in self.field_names])
        metrics['macro_recall'] = np.mean([field_metrics[field]['recall'] for field in self.field_names])
        metrics['macro_f1'] = np.mean([field_metrics[field]['f1_score'] for field in self.field_names])
        
        return metrics
    
    def train_option_a_independent(self, X, y, tune_hyperparameters=True):
        """
        Option A: Train independent binary classifiers for each field
        """
        print("\n" + "="*60)
        print("OPTION A: INDEPENDENT BINARY CLASSIFIERS")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        results = {}
        
        for model_name, base_model in self.base_models.items():
            print(f"\nTraining {model_name} (Independent Classifiers)...")
            
            try:
                # Create individual models for each field
                field_models = {}
                field_predictions = np.zeros((len(X_test), self.n_fields))
                field_probabilities = np.zeros((len(X_test), self.n_fields))
                
                for i, field_name in enumerate(self.field_names):
                    # Clone the base model
                    if hasattr(base_model, 'set_params'):
                        field_model = base_model.__class__(**base_model.get_params())
                    else:
                        field_model = base_model.__class__()
                    
                    # Hyperparameter tuning for each field
                    if tune_hyperparameters and model_name in ['random_forest', 'xgboost', 'lightgbm']:
                        try:
                            param_grid = {
                                'n_estimators': [50, 100],
                                'max_depth': [3, 5, None] if model_name == 'random_forest' else [3, 5],
                                'learning_rate': [0.1, 0.2] if model_name in ['xgboost', 'lightgbm'] else []
                            }
                            
                            # Remove learning_rate for random_forest
                            if model_name == 'random_forest':
                                param_grid.pop('learning_rate', None)
                            
                            search = GridSearchCV(field_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
                            search.fit(X_train, y_train[:, i])
                            field_model = search.best_estimator_
                        except:
                            field_model.fit(X_train, y_train[:, i])
                    else:
                        field_model.fit(X_train, y_train[:, i])
                    
                    # Predictions
                    field_predictions[:, i] = field_model.predict(X_test)
                    
                    # Probabilities if available
                    if hasattr(field_model, 'predict_proba'):
                        try:
                            proba = field_model.predict_proba(X_test)
                            field_probabilities[:, i] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                        except:
                            field_probabilities[:, i] = field_predictions[:, i]
                    else:
                        field_probabilities[:, i] = field_predictions[:, i]
                    
                    field_models[field_name] = field_model
                
                # Calculate metrics
                metrics = self.calculate_multilabel_metrics(y_test, field_predictions.astype(int), field_probabilities)
                
                results[model_name] = {
                    'models': field_models,
                    'predictions': field_predictions,
                    'probabilities': field_probabilities,
                    'metrics': metrics
                }
                
                print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
                print(f"  Exact Match Ratio: {metrics['exact_match_ratio']:.4f}")
                print(f"  Macro F1: {metrics['macro_f1']:.4f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        self.results['option_a_independent'] = results
        return results
    
    def train_option_b_multilabel(self, X, y, tune_hyperparameters=True):
        """
        Option B: Train single multi-label classifier
        """
        print("\n" + "="*60)
        print("OPTION B: SINGLE MULTI-LABEL CLASSIFIER")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        results = {}
        
        for model_name, base_model in self.base_models.items():
            print(f"\nTraining {model_name} (Multi-Label)...")
            
            try:
                # Use MultiOutputClassifier for multi-label
                model = MultiOutputClassifier(base_model)
                
                # Hyperparameter tuning
                if tune_hyperparameters and model_name in self._get_hyperparameter_grids():
                    param_grid = self._get_hyperparameter_grids()[model_name]
                    try:
                        search = RandomizedSearchCV(
                            model, param_grid, cv=3, scoring='accuracy', 
                            n_jobs=-1, n_iter=10, random_state=self.random_state
                        )
                        search.fit(X_train, y_train)
                        model = search.best_estimator_
                        print(f"  Best parameters: {search.best_params_}")
                    except:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # Predictions
                predictions = model.predict(X_test)
                
                # Probabilities if available
                try:
                    probabilities = model.predict_proba(X_test)
                    # Extract positive class probabilities
                    prob_matrix = np.zeros((len(X_test), self.n_fields))
                    for i in range(self.n_fields):
                        if probabilities[i].shape[1] > 1:
                            prob_matrix[:, i] = probabilities[i][:, 1]
                        else:
                            prob_matrix[:, i] = probabilities[i][:, 0]
                except:
                    prob_matrix = predictions.astype(float)
                
                # Calculate metrics
                metrics = self.calculate_multilabel_metrics(y_test, predictions, prob_matrix)
                
                results[model_name] = {
                    'model': model,
                    'predictions': predictions,
                    'probabilities': prob_matrix,
                    'metrics': metrics
                }
                
                print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
                print(f"  Exact Match Ratio: {metrics['exact_match_ratio']:.4f}")
                print(f"  Macro F1: {metrics['macro_f1']:.4f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        self.results['option_b_multilabel'] = results
        return results
    
    def create_neural_networks(self, input_dim):
        """
        Create neural network models for multi-label classification
        """
        models = {}
        
        # Simple neural network
        model_simple = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(self.n_fields, activation='sigmoid')  # Multi-label output
        ])
        
        model_simple.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        models['neural_simple'] = model_simple
        
        # Deeper neural network
        model_deep = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.n_fields, activation='sigmoid')
        ])
        
        model_deep.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        models['neural_deep'] = model_deep
        
        # Wide neural network
        model_wide = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(self.n_fields, activation='sigmoid')
        ])
        
        model_wide.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        models['neural_wide'] = model_wide
        
        return models
    
    def train_neural_networks(self, X, y):
        """
        Train neural network models for multi-label classification
        """
        print("\n" + "="*60)
        print("NEURAL NETWORK MODELS")
        print("="*60)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Additional scaling for neural networks
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        neural_models = self.create_neural_networks(X_train_scaled.shape[1])
        results = {}
        
        for model_name, model in neural_models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Callbacks
                early_stopping = EarlyStopping(
                    monitor='val_loss', patience=15, restore_best_weights=True
                )
                
                # Train model
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Predictions
                predictions_prob = model.predict(X_test_scaled)
                predictions = (predictions_prob > 0.5).astype(int)
                
                # Calculate metrics
                metrics = self.calculate_multilabel_metrics(y_test, predictions, predictions_prob)
                
                results[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'predictions': predictions,
                    'probabilities': predictions_prob,
                    'metrics': metrics,
                    'history': history
                }
                
                print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
                print(f"  Exact Match Ratio: {metrics['exact_match_ratio']:.4f}")
                print(f"  Macro F1: {metrics['macro_f1']:.4f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        self.results['neural_networks'] = results
        return results
    
    def compare_approaches(self):
        """
        Compare Option A vs Option B vs Neural Networks
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPARISON OF ALL APPROACHES")
        print("="*70)
        
        all_results = []
        
        # Extract results from all approaches
        for approach_name, approach_results in self.results.items():
            for model_name, result in approach_results.items():
                if 'metrics' in result:
                    metrics = result['metrics']
                    all_results.append({
                        'Approach': approach_name.replace('_', ' ').title(),
                        'Model': model_name,
                        'Hamming_Loss': metrics['hamming_loss'],
                        'Exact_Match_Ratio': metrics['exact_match_ratio'],
                        'Macro_F1': metrics['macro_f1'],
                        'Macro_Accuracy': metrics['macro_accuracy']
                    })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        
        if len(comparison_df) == 0:
            print("No results to compare!")
            return None
        
        # Sort by macro F1 score
        comparison_df = comparison_df.sort_values('Macro_F1', ascending=False)
        
        print("\nTop 10 Models by Macro F1 Score:")
        print(comparison_df.head(10).to_string(index=False))
        
        # Create visualizations
        plt.figure(figsize=(20, 12))
        
        # 1. Performance by approach
        plt.subplot(2, 3, 1)
        approach_performance = comparison_df.groupby('Approach')['Macro_F1'].agg(['mean', 'std'])
        plt.bar(approach_performance.index, approach_performance['mean'], 
                yerr=approach_performance['std'], capsize=5)
        plt.title('Mean Macro F1 by Approach')
        plt.ylabel('Macro F1 Score')
        plt.xticks(rotation=45)
        
        # 2. Hamming Loss comparison
        plt.subplot(2, 3, 2)
        approach_hamming = comparison_df.groupby('Approach')['Hamming_Loss'].agg(['mean', 'std'])
        plt.bar(approach_hamming.index, approach_hamming['mean'], 
                yerr=approach_hamming['std'], capsize=5)
        plt.title('Mean Hamming Loss by Approach')
        plt.ylabel('Hamming Loss')
        plt.xticks(rotation=45)
        
        # 3. Exact match ratio
        plt.subplot(2, 3, 3)
        approach_exact = comparison_df.groupby('Approach')['Exact_Match_Ratio'].agg(['mean', 'std'])
        plt.bar(approach_exact.index, approach_exact['mean'], 
                yerr=approach_exact['std'], capsize=5)
        plt.title('Mean Exact Match Ratio by Approach')
        plt.ylabel('Exact Match Ratio')
        plt.xticks(rotation=45)
        
        # 4. Model performance distribution
        plt.subplot(2, 3, 4)
        for approach in comparison_df['Approach'].unique():
            subset = comparison_df[comparison_df['Approach'] == approach]
            plt.hist(subset['Macro_F1'], alpha=0.6, label=approach, bins=10)
        plt.xlabel('Macro F1 Score')
        plt.ylabel('Frequency')
        plt.title('F1 Score Distribution by Approach')
        plt.legend()
        
        # 5. Scatter plot of metrics
        plt.subplot(2, 3, 5)
        colors = {'Option A Independent': 'blue', 'Option B Multilabel': 'red', 'Neural Networks': 'green'}
        for approach in comparison_df['Approach'].unique():
            subset = comparison_df[comparison_df['Approach'] == approach]
            plt.scatter(subset['Hamming_Loss'], subset['Macro_F1'], 
                       label=approach, alpha=0.7, c=colors.get(approach, 'gray'))
        plt.xlabel('Hamming Loss')
        plt.ylabel('Macro F1 Score')
        plt.title('Hamming Loss vs Macro F1')
        plt.legend()
        
        # 6. Best models comparison
        plt.subplot(2, 3, 6)
        top_models = comparison_df.head(10)
        plt.barh(range(len(top_models)), top_models['Macro_F1'])
        plt.yticks(range(len(top_models)), 
                  [f"{row['Model']} ({row['Approach']})" for _, row in top_models.iterrows()])
        plt.xlabel('Macro F1 Score')
        plt.title('Top 10 Models')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def analyze_field_specific_performance(self):
        """
        Analyze performance for each field individually
        """
        print("\n" + "="*70)
        print("FIELD-SPECIFIC PERFORMANCE ANALYSIS")
        print("="*70)
        
        # Get best model from each approach
        best_models = {}
        for approach_name, approach_results in self.results.items():
            best_f1 = 0
            best_model_name = None
            best_result = None
            
            for model_name, result in approach_results.items():
                if 'metrics' in result and result['metrics']['macro_f1'] > best_f1:
                    best_f1 = result['metrics']['macro_f1']
                    best_model_name = model_name
                    best_result = result
            
            if best_result:
                best_models[approach_name] = {
                    'name': best_model_name,
                    'result': best_result
                }
        
        # Create field performance comparison
        field_performance_data = []
        
        for approach_name, model_info in best_models.items():
            if 'metrics' in model_info['result']:
                field_metrics = model_info['result']['metrics']['field_metrics']
                for field_name in self.field_names:
                    field_performance_data.append({
                        'Approach': approach_name.replace('_', ' ').title(),
                        'Model': model_info['name'],
                        'Field': field_name,
                        'Accuracy': field_metrics[field_name]['accuracy'],
                        'Precision': field_metrics[field_name]['precision'],
                        'Recall': field_metrics[field_name]['recall'],
                        'F1_Score': field_metrics[field_name]['f1_score']
                    })
        
        field_df = pd.DataFrame(field_performance_data)
        
        if len(field_df) == 0:
            print("No field-specific results to analyze!")
            return None
        
        # Visualize field-specific performance
        plt.figure(figsize=(15, 10))
        
        # Heatmap of F1 scores by field and approach
        plt.subplot(2, 2, 1)
        pivot_f1 = field_df.pivot_table(values='F1_Score', index='Field', columns='Approach')
        sns.heatmap(pivot_f1, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('F1 Score by Field and Approach')
        
        # Accuracy comparison
        plt.subplot(2, 2, 2)
        pivot_acc = field_df.pivot_table(values='Accuracy', index='Field', columns='Approach')
        sns.heatmap(pivot_acc, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Accuracy by Field and Approach')
        
        # Field difficulty analysis
        plt.subplot(2, 2, 3)
        field_difficulty = field_df.groupby('Field')['F1_Score'].agg(['mean', 'std']).sort_values('mean')
        plt.barh(range(len(field_difficulty)), field_difficulty['mean'], 
                xerr=field_difficulty['std'], capsize=5)
        plt.yticks(range(len(field_difficulty)), field_difficulty.index)
        plt.xlabel('Mean F1 Score')
        plt.title('Field Difficulty (Lower F1 = Harder)')
        
        # Best approach per field
        plt.subplot(2, 2, 4)
        best_per_field = field_df.loc[field_df.groupby('Field')['F1_Score'].idxmax()]
        approach_counts = best_per_field['Approach'].value_counts()
        plt.pie(approach_counts.values, labels=approach_counts.index, autopct='%1.1f%%')
        plt.title('Best Approach per Field')
        
        plt.tight_layout()
        plt.show()
        
        print("\nField Performance Summary:")
        print(field_df.groupby(['Approach', 'Field'])['F1_Score'].mean().unstack().round(3))
        
        return field_df
    
    def run_complete_analysis(self, X, y, tune_hyperparameters=True):
        """
        Run complete multi-label classification analysis
        """
        print("MULTI-LABEL FIELD CHANGES PREDICTION ANALYSIS")
        print("=" * 60)
        print(f"Dataset shape: {X.shape}")
        print(f"Number of fields: {self.n_fields}")
        print(f"Field names: {self.field_names}")
        print(f"Field change rates: {y.mean(axis=0).round(3)}")
        
        # Preprocess data
        print("\nPreprocessing data...")
        X_processed = self.preprocess_data(X)
        print(f"Processed dataset shape: {X_processed.shape}")
        
        # Train all approaches
        option_a_results = self.train_option_a_independent(X_processed, y, tune_hyperparameters)
        option_b_results = self.train_option_b_multilabel(X_processed, y, tune_hyperparameters)
        neural_results = self.train_neural_networks(X_processed, y)
        
        # Compare approaches
        comparison_df = self.compare_approaches()
        
        # Field-specific analysis
        field_analysis = self.analyze_field_specific_performance()
        
        # Final recommendations
        self.print_final_recommendations(comparison_df)
        
        return {
            'option_a': option_a_results,
            'option_b': option_b_results,
            'neural': neural_results,
            'comparison': comparison_df,
            'field_analysis': field_analysis
        }
    
    def print_final_recommendations(self, comparison_df):
        """
        Print final recommendations based on analysis
        """
        print("\n" + "="*70)
        print("FINAL RECOMMENDATIONS")
        print("="*70)
        
        if comparison_df is not None and len(comparison_df) > 0:
            best_overall = comparison_df.iloc[0]
            
            print(f"\n🏆 BEST OVERALL MODEL:")
            print(f"   Model: {best_overall['Model']}")
            print(f"   Approach: {best_overall['Approach']}")
            print(f"   Macro F1: {best_overall['Macro_F1']:.4f}")
            print(f"   Hamming Loss: {best_overall['Hamming_Loss']:.4f}")
            print(f"   Exact Match Ratio: {best_overall['Exact_Match_Ratio']:.4f}")
            
            # Best by approach
            best_by_approach = comparison_df.groupby('Approach').first()
            
            print(f"\n📊 BEST MODEL PER APPROACH:")
            for approach, row in best_by_approach.iterrows():
                print(f"   {approach}: {row['Model']} (F1: {row['Macro_F1']:.4f})")
            
            # Approach comparison
            approach_stats = comparison_df.groupby('Approach')[['Macro_F1', 'Hamming_Loss', 'Exact_Match_Ratio']].mean()
            best_approach = approach_stats['Macro_F1'].idxmax()
            
            print(f"\n🎯 BEST APPROACH OVERALL: {best_approach}")
            print(f"   Average Macro F1: {approach_stats.loc[best_approach, 'Macro_F1']:.4f}")
            print(f"   Average Hamming Loss: {approach_stats.loc[best_approach, 'Hamming_Loss']:.4f}")
        
        print(f"\n💡 KEY INSIGHTS:")
        print("   1. Option A (Independent): Good for uncorrelated field changes")
        print("   2. Option B (Multi-label): Better when field changes are correlated")
        print("   3. Neural Networks: May need more data/tuning for optimal performance")
        print("   4. Tree-based models typically perform well on tabular financial data")
        
        print(f"\n🚀 NEXT STEPS:")
        print("   1. Use best performing model as baseline")
        print("   2. Consider ensemble methods combining top approaches")
        print("   3. Investigate feature engineering based on domain knowledge")
        print("   4. Analyze prediction errors to identify improvement opportunities")
        print("   5. Validate on out-of-time test data for production readiness")

def create_simplified_demo_data():
    """
    Create demonstration data focused on field changes prediction
    """
    np.random.seed(42)
    
    n_samples = 3000
    
    # Financial features relevant to field changes
    numerical_features = pd.DataFrame({
        'transaction_amount': np.random.lognormal(10, 1.5, n_samples),
        'account_balance': np.random.lognormal(12, 1, n_samples),
        'interest_rate': np.random.normal(0.03, 0.01, n_samples),
        'days_to_maturity': np.random.exponential(200, n_samples),
        'volatility': np.random.exponential(0.1, n_samples),
        'price_change': np.random.normal(0, 0.02, n_samples),
        'volume_ratio': np.random.lognormal(0, 0.5, n_samples),
        'credit_rating_score': np.random.uniform(1, 10, n_samples),
        'liquidity_ratio': np.random.gamma(2, 0.5, n_samples),
        'market_cap_log': np.random.normal(18, 2, n_samples)
    })
    
    categorical_features = pd.DataFrame({
        'security_type': np.random.choice(['bond', 'equity', 'derivative'], n_samples, p=[0.5, 0.3, 0.2]),
        'currency': np.random.choice(['USD', 'EUR', 'GBP', 'JPY'], n_samples, p=[0.5, 0.25, 0.15, 0.1]),
        'custodian': np.random.choice(['BNY', 'SSGA', 'JPM', 'CITI'], n_samples, p=[0.3, 0.3, 0.25, 0.15]),
        'transaction_type': np.random.choice(['BUY', 'SELL', 'DIVIDEND', 'COUPON'], n_samples, p=[0.35, 0.35, 0.15, 0.15]),
        'portfolio_type': np.random.choice(['equity_fund', 'bond_fund', 'mixed_fund'], n_samples, p=[0.4, 0.4, 0.2]),
        'market_sector': np.random.choice(['FINANCIAL', 'TECH', 'HEALTHCARE', 'ENERGY'], n_samples, p=[0.3, 0.25, 0.25, 0.2])
    })
    
    # Combine features
    X = pd.concat([numerical_features, categorical_features], axis=1)
    
    # Introduce missing values
    missing_cols = ['days_to_maturity', 'volatility', 'market_cap_log']
    for col in missing_cols:
        missing_mask = np.random.random(n_samples) < 0.15
        X.loc[missing_mask, col] = np.nan
    
    # Create correlated field changes
    field_names = ['previous_coupon', 'factor', 'previous_factor', 'interest', 'principles', 'date']
    
    # Base probabilities for each field
    base_probs = [0.12, 0.15, 0.10, 0.18, 0.08, 0.06]
    
    # Create correlations between fields and features
    y = np.zeros((n_samples, len(field_names)))
    
    for i, (field_name, base_prob) in enumerate(zip(field_names, base_probs)):
        # Field-specific logic
        field_logits = np.log(base_prob / (1 - base_prob))  # Base odds
        
        # Common factors affecting all fields
        field_logits += 0.8 * (X['transaction_amount'] > X['transaction_amount'].quantile(0.9))
        field_logits += 0.6 * (X['security_type'] == 'derivative')
        field_logits += 0.4 * (X['transaction_type'].isin(['DIVIDEND', 'COUPON']))
        
        # Field-specific factors
        if field_name in ['previous_coupon', 'interest']:
            field_logits += 0.7 * (X['security_type'] == 'bond')
            field_logits += 0.5 * (X['interest_rate'] > 0.04)
        
        if field_name in ['factor', 'previous_factor']:
            field_logits += 0.6 * (X['volatility'] > X['volatility'].quantile(0.8, interpolation='nearest'))
            field_logits += 0.4 * (X['security_type'] == 'derivative')
        
        if field_name == 'principles':
            field_logits += 0.5 * (X['transaction_type'] == 'SELL')
        
        if field_name == 'date':
            field_logits += 0.3 * (X['days_to_maturity'] < 30)
        
        # Add some correlation between fields
        if i > 0:
            field_logits += 0.4 * y[:, i-1]  # Correlation with previous field
        
        # Add noise
        field_logits += np.random.normal(0, 0.3, n_samples)
        
        # Convert to probabilities and sample
        field_prob = 1 / (1 + np.exp(-field_logits))
        y[:, i] = np.random.binomial(1, field_prob)
    
    return X, y, field_names

def main():
    """
    Main function to run simplified multi-label analysis
    """
    print("SIMPLIFIED MULTI-LABEL FIELD CHANGES PREDICTION")
    print("=" * 60)
    
    # Create demonstration data
    print("Creating demonstration data...")
    X, y, field_names = create_simplified_demo_data()
    
    print(f"Dataset created with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Field names: {field_names}")
    print(f"Field change rates:")
    for i, field_name in enumerate(field_names):
        print(f"  {field_name}: {y[:, i].mean():.3f}")
    
    # Check correlations between fields
    field_corr = np.corrcoef(y.T)
    print(f"\nField correlation matrix:")
    correlation_df = pd.DataFrame(field_corr, index=field_names, columns=field_names)
    print(correlation_df.round(3))
    
    # Initialize baseline
    baseline = MultiLabelFieldChangesBaseline(field_names, random_state=42)
    
    # Run complete analysis
    results = baseline.run_complete_analysis(X, y, tune_hyperparameters=True)
    
    # Additional insights
    print("\n" + "="*70)
    print("ADDITIONAL INSIGHTS")
    print("="*70)
    
    # Analyze label co-occurrence
    print("\nLabel Co-occurrence Analysis:")
    label_combinations = {}
    for i in range(len(y)):
        combination = tuple(y[i])
        label_combinations[combination] = label_combinations.get(combination, 0) + 1
    
    # Show most common combinations
    sorted_combinations = sorted(label_combinations.items(), key=lambda x: x[1], reverse=True)
    print("Most common label combinations:")
    for combination, count in sorted_combinations[:10]:
        active_fields = [field_names[j] for j, val in enumerate(combination) if val == 1]
        print(f"  {active_fields if active_fields else 'No changes'}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Label frequency
    print(f"\nLabel Statistics:")
    print(f"  Average fields changed per sample: {y.sum(axis=1).mean():.2f}")
    print(f"  Samples with no field changes: {(y.sum(axis=1) == 0).sum()} ({(y.sum(axis=1) == 0).mean()*100:.1f}%)")
    print(f"  Samples with all fields changed: {(y.sum(axis=1) == len(field_names)).sum()}")
    
    # Feature importance for best model
    if 'comparison' in results and results['comparison'] is not None:
        best_model_info = results['comparison'].iloc[0]
        print(f"\nBest performing model: {best_model_info['Model']} ({best_model_info['Approach']})")
        
        # Try to extract feature importance if it's a tree-based model
        best_approach = best_model_info['Approach'].lower().replace(' ', '_')
        best_model_name = best_model_info['Model']
        
        if best_approach in baseline.results and best_model_name in baseline.results[best_approach]:
            result = baseline.results[best_approach][best_model_name]
            
            if 'model' in result:
                model = result['model']
                
                # Try to get feature importance
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
                        # For MultiOutputClassifier, average importance across estimators
                        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
                    else:
                        importances = None
                    
                    if importances is not None:
                        X_processed = baseline.preprocess_data(X)
                        feature_names_processed = X_processed.columns
                        
                        importance_df = pd.DataFrame({
                            'Feature': feature_names_processed,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                        
                        print(f"\nTop 10 Most Important Features ({best_model_name}):")
                        print(importance_df.head(10).to_string(index=False))
                
                except Exception as e:
                    print(f"Could not extract feature importance: {e}")
    
    print(f"\n🎉 Analysis Complete!")
    print(f"Check the visualizations and results above for detailed insights.")

if __name__ == "__main__":
    main()
