"""
Multi-Task Learning Framework for Cash Reconciliation
====================================================

This script implements the proposed multi-task neural network architecture using TensorFlow.
It includes:
1. Feature engineering pipeline for mixed data types
2. Multi-task architecture with shared layers and task-specific heads
3. Both approaches for field changes prediction
4. Attention mechanisms for interpretability
5. Training and evaluation pipeline
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """
    Feature engineering pipeline for mixed data types with missing values
    """
    
    def __init__(self, embedding_dims: Dict[str, int] = None):
        self.numerical_scaler = StandardScaler()
        self.label_encoders = {}
        self.embedding_dims = embedding_dims or {}
        self.feature_stats = {}
        
    def fit_transform_numerical(self, X_num: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process numerical features with missingness indicators
        """
        # Create missingness indicators
        missing_indicators = X_num.isnull().astype(int)
        
        # Fill missing values with median
        X_num_filled = X_num.fillna(X_num.median())
        
        # Scale features
        X_num_scaled = self.numerical_scaler.fit_transform(X_num_filled)
        
        # Store statistics for interpretability
        self.feature_stats['numerical'] = {
            'means': X_num.mean().to_dict(),
            'stds': X_num.std().to_dict(),
            'missing_rates': X_num.isnull().mean().to_dict()
        }
        
        return X_num_scaled, missing_indicators.values
    
    def fit_transform_categorical(self, X_cat: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Process categorical features with embeddings
        """
        encoded_features = {}
        
        for col in X_cat.columns:
            # Handle missing values by treating them as a separate category
            X_cat_filled = X_cat[col].fillna('MISSING').astype(str)
            
            # Fit label encoder
            self.label_encoders[col] = LabelEncoder()
            encoded = self.label_encoders[col].fit_transform(X_cat_filled)
            
            # Store vocabulary size for embedding layer
            vocab_size = len(self.label_encoders[col].classes_)
            self.embedding_dims[col] = min(50, (vocab_size + 1) // 2)  # Rule of thumb
            
            encoded_features[col] = encoded
            
            # Store statistics
            if 'categorical' not in self.feature_stats:
                self.feature_stats['categorical'] = {}
            
            self.feature_stats['categorical'][col] = {
                'vocab_size': vocab_size,
                'categories': self.label_encoders[col].classes_.tolist(),
                'missing_rate': X_cat[col].isnull().mean()
            }
        
        return encoded_features

class AttentionLayer(layers.Layer):
    """
    Custom attention layer for interpretability
    """
    
    def __init__(self, attention_dim=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.attention_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.attention_dim,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.attention_dim,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Calculate attention scores
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=-1)
        
        # Apply attention weights
        ait = tf.expand_dims(ait, -1)
        weighted_input = inputs * ait
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output, ait

class MultiTaskCashReconciliation:
    """
    Multi-task learning framework for cash reconciliation
    """
    
    def __init__(self, 
                 field_names: List[str],
                 shared_layers_config: List[int] = [256, 128, 64],
                 task_head_config: List[int] = [32, 16],
                 use_attention: bool = True,
                 field_prediction_approach: str = 'multilabel'):
        
        self.field_names = field_names
        self.shared_layers_config = shared_layers_config
        self.task_head_config = task_head_config
        self.use_attention = use_attention
        self.field_prediction_approach = field_prediction_approach  # 'independent' or 'multilabel'
        
        self.feature_engineering = FeatureEngineering()
        self.model = None
        self.history = None
        
    def build_model(self, 
                   numerical_features: int,
                   categorical_features_info: Dict[str, int],
                   missing_indicators: int) -> Model:
        """
        Build the multi-task neural network architecture
        """
        
        # Input layers
        inputs = {}
        embeddings = []
        
        # Numerical features input
        numerical_input = Input(shape=(numerical_features,), name='numerical_features')
        inputs['numerical'] = numerical_input
        
        # Missing indicators input
        missing_input = Input(shape=(missing_indicators,), name='missing_indicators')
        inputs['missing'] = missing_input
        
        # Categorical features inputs and embeddings
        for feature_name, vocab_size in categorical_features_info.items():
            cat_input = Input(shape=(1,), name=f'cat_{feature_name}')
            inputs[f'cat_{feature_name}'] = cat_input
            
            embedding_dim = self.feature_engineering.embedding_dims[feature_name]
            embedding = layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name=f'embedding_{feature_name}'
            )(cat_input)
            embedding = layers.Flatten()(embedding)
            embeddings.append(embedding)
        
        # Concatenate all features
        if embeddings:
            concatenated = layers.Concatenate(name='feature_concatenation')([
                numerical_input, missing_input
            ] + embeddings)
        else:
            concatenated = layers.Concatenate(name='feature_concatenation')([
                numerical_input, missing_input
            ])
        
        # Shared layers (trunk of the model)
        x = concatenated
        for i, units in enumerate(self.shared_layers_config):
            x = layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.001),
                name=f'shared_dense_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'shared_bn_{i}')(x)
            x = layers.Dropout(0.3, name=f'shared_dropout_{i}')(x)
        
        shared_representation = x
        
        # Optional attention mechanism
        if self.use_attention:
            attended_features, attention_weights = AttentionLayer(
                attention_dim=64, 
                name='attention_layer'
            )(tf.expand_dims(shared_representation, 1))
            shared_representation = attended_features
        
        # Task 1: Cash break prediction (binary classification)
        cash_break_head = shared_representation
        for i, units in enumerate(self.task_head_config):
            cash_break_head = layers.Dense(
                units,
                activation='relu',
                name=f'cash_break_dense_{i}'
            )(cash_break_head)
            cash_break_head = layers.Dropout(0.2)(cash_break_head)
        
        cash_break_output = layers.Dense(
            1,
            activation='sigmoid',
            name='cash_break_output'
        )(cash_break_head)
        
        # Task 2: Field changes prediction
        field_change_head = shared_representation
        for i, units in enumerate(self.task_head_config):
            field_change_head = layers.Dense(
                units,
                activation='relu',
                name=f'field_change_dense_{i}'
            )(field_change_head)
            field_change_head = layers.Dropout(0.2)(field_change_head)
        
        if self.field_prediction_approach == 'independent':
            # Option A: Multiple independent binary classifiers
            field_outputs = {}
            for field_name in self.field_names:
                field_output = layers.Dense(
                    1,
                    activation='sigmoid',
                    name=f'field_change_{field_name}'
                )(field_change_head)
                field_outputs[f'field_change_{field_name}'] = field_output
            
            all_outputs = {'cash_break_output': cash_break_output}
            all_outputs.update(field_outputs)
            
        else:
            # Option B: Single multi-label classifier
            field_changes_output = layers.Dense(
                len(self.field_names),
                activation='sigmoid',
                name='field_changes_output'
            )(field_change_head)
            
            all_outputs = {
                'cash_break_output': cash_break_output,
                'field_changes_output': field_changes_output
            }
        
        # Create model
        model = Model(inputs=list(inputs.values()), outputs=list(all_outputs.values()))
        
        return model, inputs, all_outputs
    
    def compile_model(self, model: Model, outputs_info: Dict):
        """
        Compile the model with appropriate loss functions and metrics
        """
        losses = {}
        metrics = {}
        loss_weights = {}
        
        if self.field_prediction_approach == 'independent':
            # Cash break task
            losses['cash_break_output'] = 'binary_crossentropy'
            metrics['cash_break_output'] = ['accuracy']
            loss_weights['cash_break_output'] = 1.0
            
            # Individual field tasks
            for field_name in self.field_names:
                output_name = f'field_change_{field_name}'
                losses[output_name] = 'binary_crossentropy'
                metrics[output_name] = ['accuracy']
                loss_weights[output_name] = 0.5  # Lower weight for individual fields
        else:
            # Multi-label approach
            losses = {
                'cash_break_output': 'binary_crossentropy',
                'field_changes_output': 'binary_crossentropy'
            }
            metrics = {
                'cash_break_output': ['accuracy'],
                'field_changes_output': ['accuracy']
            }
            loss_weights = {
                'cash_break_output': 1.0,
                'field_changes_output': 1.0
            }
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )
        
        return model
    
    def prepare_data(self, X: pd.DataFrame, 
                    y_cash_break: np.ndarray,
                    y_field_changes: np.ndarray) -> Tuple:
        """
        Prepare data for training
        """
        # Separate numerical and categorical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        X_num = X[numerical_cols] if numerical_cols else pd.DataFrame()
        X_cat = X[categorical_cols] if categorical_cols else pd.DataFrame()
        
        # Feature engineering
        if not X_num.empty:
            X_num_processed, missing_indicators = self.feature_engineering.fit_transform_numerical(X_num)
        else:
            X_num_processed = np.empty((len(X), 0))
            missing_indicators = np.empty((len(X), 0))
        
        if not X_cat.empty:
            X_cat_processed = self.feature_engineering.fit_transform_categorical(X_cat)
        else:
            X_cat_processed = {}
        
        # Prepare inputs dictionary
        inputs_data = {
            'numerical_features': X_num_processed,
            'missing_indicators': missing_indicators
        }
        
        # Add categorical inputs
        for feature_name, encoded_data in X_cat_processed.items():
            inputs_data[f'cat_{feature_name}'] = encoded_data.reshape(-1, 1)
        
        # Prepare outputs
        if self.field_prediction_approach == 'independent':
            outputs_data = {'cash_break_output': y_cash_break}
            for i, field_name in enumerate(self.field_names):
                outputs_data[f'field_change_{field_name}'] = y_field_changes[:, i]
        else:
            outputs_data = {
                'cash_break_output': y_cash_break,
                'field_changes_output': y_field_changes
            }
        
        return inputs_data, outputs_data, X_cat_processed
    
    def train(self, X: pd.DataFrame,
             y_cash_break: np.ndarray,
             y_field_changes: np.ndarray,
             validation_split: float = 0.2,
             epochs: int = 100,
             batch_size: int = 32) -> Model:
        """
        Train the multi-task model
        """
        print("Preparing data...")
        inputs_data, outputs_data, cat_features_info = self.prepare_data(X, y_cash_break, y_field_changes)
        
        # Get feature dimensions
        numerical_features = inputs_data['numerical_features'].shape[1]
        missing_indicators = inputs_data['missing_indicators'].shape[1]
        categorical_features_vocab = {}
        
        for feature_name in cat_features_info.keys():
            vocab_size = len(self.feature_engineering.label_encoders[feature_name].classes_)
            categorical_features_vocab[feature_name] = vocab_size
        
        print("Building model...")
        model, input_layers, output_layers = self.build_model(
            numerical_features=numerical_features,
            categorical_features_info=categorical_features_vocab,
            missing_indicators=missing_indicators
        )
        
        print("Compiling model...")
        model = self.compile_model(model, output_layers)
        
        print(f"Model architecture:")
        model.summary()
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        print("Training model...")
        history = model.fit(
            x=list(inputs_data.values()),
            y=list(outputs_data.values()),
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = model
        self.history = history
        
        return model
    
    def evaluate_model(self, X_test: pd.DataFrame,
                      y_cash_break_test: np.ndarray,
                      y_field_changes_test: np.ndarray):
        """
        Evaluate the trained model
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Prepare test data
        inputs_test, outputs_test, _ = self.prepare_data(X_test, y_cash_break_test, y_field_changes_test)
        
        # Make predictions
        predictions = self.model.predict(list(inputs_test.values()))
        
        if self.field_prediction_approach == 'independent':
            cash_break_pred = predictions[0]
            field_change_preds = predictions[1:]
            
            # Evaluate cash break prediction
            cash_break_pred_binary = (cash_break_pred > 0.5).astype(int).flatten()
            cash_break_accuracy = accuracy_score(y_cash_break_test, cash_break_pred_binary)
            
            print(f"Cash Break Prediction Accuracy: {cash_break_accuracy:.4f}")
            print("\nCash Break Classification Report:")
            print(classification_report(y_cash_break_test, cash_break_pred_binary))
            
            # Evaluate individual field predictions
            print("\nField Change Predictions (Independent):")
            for i, field_name in enumerate(self.field_names):
                field_pred_binary = (field_change_preds[i] > 0.5).astype(int).flatten()
                field_accuracy = accuracy_score(y_field_changes_test[:, i], field_pred_binary)
                print(f"{field_name}: Accuracy = {field_accuracy:.4f}")
        
        else:
            cash_break_pred = predictions[0]
            field_changes_pred = predictions[1]
            
            # Evaluate cash break prediction
            cash_break_pred_binary = (cash_break_pred > 0.5).astype(int).flatten()
            cash_break_accuracy = accuracy_score(y_cash_break_test, cash_break_pred_binary)
            
            print(f"Cash Break Prediction Accuracy: {cash_break_accuracy:.4f}")
            print("\nCash Break Classification Report:")
            print(classification_report(y_cash_break_test, cash_break_pred_binary))
            
            # Evaluate multi-label field changes
            field_changes_pred_binary = (field_changes_pred > 0.5).astype(int)
            hamming = hamming_loss(y_field_changes_test, field_changes_pred_binary)
            
            print(f"\nField Changes Hamming Loss: {hamming:.4f}")
            print("\nField-wise Accuracies:")
            for i, field_name in enumerate(self.field_names):
                field_accuracy = accuracy_score(
                    y_field_changes_test[:, i], 
                    field_changes_pred_binary[:, i]
                )
                print(f"{field_name}: {field_accuracy:.4f}")
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Cash break accuracy
        if 'cash_break_output_accuracy' in self.history.history:
            axes[0, 1].plot(self.history.history['cash_break_output_accuracy'], label='Training Accuracy')
            axes[0, 1].plot(self.history.history['val_cash_break_output_accuracy'], label='Validation Accuracy')
            axes[0, 1].set_title('Cash Break Prediction Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
        
        # Field changes accuracy
        if self.field_prediction_approach == 'multilabel':
            if 'field_changes_output_accuracy' in self.history.history:
                axes[1, 0].plot(self.history.history['field_changes_output_accuracy'], label='Training Accuracy')
                axes[1, 0].plot(self.history.history['val_field_changes_output_accuracy'], label='Validation Accuracy')
                axes[1, 0].set_title('Field Changes Prediction Accuracy')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].legend()
        
        # Learning rate
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self, X_sample: pd.DataFrame = None):
        """
        Extract feature importance using attention weights (if available)
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if not self.use_attention:
            print("Attention mechanism not enabled. Cannot extract attention-based importance.")
            return None
        
        # Get attention layer
        attention_layer = None
        for layer in self.model.layers:
            if isinstance(layer, AttentionLayer):
                attention_layer = layer
                break
        
        if attention_layer is None:
            print("Attention layer not found in model.")
            return None
        
        print("Feature importance analysis requires sample data for attention weights extraction.")
        print("Use SHAP or LIME for comprehensive post-hoc explanations.")
        
        return self.feature_engineering.feature_stats

def create_demo_data():
    """
    Create demonstration data for the multi-task learning framework
    """
    np.random.seed(42)
    
    n_samples = 5000
    
    # Create numerical features (representing transaction amounts, ratios, etc.)
    numerical_features = {
        'transaction_amount': np.random.lognormal(10, 1, n_samples),
        'account_balance': np.random.lognormal(12, 1.5, n_samples),
        'settlement_days': np.random.poisson(3, n_samples),
        'interest_rate': np.random.normal(0.03, 0.01, n_samples),
        'exchange_rate': np.random.normal(1.0, 0.1, n_samples),
        'previous_amount': np.random.lognormal(9.8, 1.2, n_samples),
        'factor_ratio': np.random.normal(1.0, 0.05, n_samples),
        'days_to_maturity': np.random.exponential(100, n_samples),
        'coupon_rate': np.random.uniform(0, 0.08, n_samples),
        'price_volatility': np.random.exponential(0.1, n_samples)
    }
    
    # Create categorical features
    categorical_features = {
        'security_type': np.random.choice(['bond', 'equity', 'derivative', 'cash'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'currency': np.random.choice(['USD', 'EUR', 'GBP', 'JPY', 'CHF'], n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
        'custodian': np.random.choice(['BNY', 'SSGA', 'JPM', 'CITI', 'BNP'], n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'portfolio': np.random.choice([f'PF_{i}' for i in range(1, 21)], n_samples),
        'trader': np.random.choice([f'TRADER_{i}' for i in range(1, 11)], n_samples),
        'settlement_status': np.random.choice(['pending', 'settled', 'failed'], n_samples, p=[0.1, 0.85, 0.05]),
        'transaction_type': np.random.choice(['buy', 'sell', 'dividend', 'interest', 'fx'], n_samples, p=[0.3, 0.3, 0.2, 0.15, 0.05])
    }
    
    # Introduce missing values
    for feature_name, feature_data in numerical_features.items():
        missing_rate = np.random.uniform(0.05, 0.25)  # 5-25% missing
        missing_mask = np.random.random(n_samples) < missing_rate
        feature_data[missing_mask] = np.nan
    
    for feature_name, feature_data in categorical_features.items():
        missing_rate = np.random.uniform(0.02, 0.15)  # 2-15% missing
        missing_mask = np.random.random(n_samples) < missing_rate
        feature_data[missing_mask] = None
    
    # Create DataFrame
    data = {}
    data.update(numerical_features)
    data.update(categorical_features)
    X = pd.DataFrame(data)
    
    # Create realistic target variables with some correlation to features
    # Cash break probability influenced by transaction amount, settlement status, etc.
    cash_break_prob = (
        0.1 +  # Base probability
        0.15 * (X['transaction_amount'] > X['transaction_amount'].quantile(0.9)) +  # Large transactions
        0.3 * (X['settlement_status'] == 'failed') +  # Failed settlements
        0.1 * (X['security_type'] == 'derivative') +  # Complex securities
        0.05 * np.random.random(n_samples)  # Random component
    )
    cash_break_prob = np.clip(cash_break_prob, 0, 1)
    y_cash_break = np.random.binomial(1, cash_break_prob)
    
    # Field changes with some correlation to cash breaks and features
    field_names = ['previous_coupon', 'factor', 'previous_factor', 'interest', 'principles', 'date']
    y_field_changes = np.zeros((n_samples, len(field_names)))
    
    for i, field_name in enumerate(field_names):
        field_prob = (
            0.05 +  # Base probability
            0.2 * y_cash_break +  # Higher if cash break occurs
            0.1 * (X['security_type'].isin(['bond', 'derivative'])) +  # Bond-related fields
            0.05 * np.random.random(n_samples)  # Random component
        )
        field_prob = np.clip(field_prob, 0, 1)
        y_field_changes[:, i] = np.random.binomial(1, field_prob)
    
    return X, y_cash_break, y_field_changes, field_names

def main():
    """
    Main demonstration of the multi-task learning framework
    """
    print("Multi-Task Learning for Cash Reconciliation")
    print("=" * 50)
    
    # Create demonstration data
    print("Creating demonstration data...")
    X, y_cash_break, y_field_changes, field_names = create_demo_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Cash break rate: {y_cash_break.mean():.3f}")
    print(f"Field change rates: {y_field_changes.mean(axis=0)}")
    print(f"Missing data rates:")
    print(X.isnull().mean().sort_values(ascending=False).head(10))
    
    # Split data
    X_train, X_test, y_cash_train, y_cash_test, y_field_train, y_field_test = train_test_split(
        X, y_cash_break, y_field_changes, test_size=0.2, random_state=42, stratify=y_cash_break
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Option A: Independent binary classifiers approach
    print("\n" + "=" * 60)
    print("TRAINING MODEL - INDEPENDENT APPROACH")
    print("=" * 60)
    
    mtl_independent = MultiTaskCashReconciliation(
        field_names=field_names,
        shared_layers_config=[256, 128, 64],
        task_head_config=[32, 16],
        use_attention=True,
        field_prediction_approach='independent'
    )
    
    model_independent = mtl_independent.train(
        X_train, y_cash_train, y_field_train,
        epochs=50, batch_size=64
    )
    
    print("\nEvaluating Independent Approach...")
    mtl_independent.evaluate_model(X_test, y_cash_test, y_field_test)
    
    # Option B: Multi-label classifier approach
    print("\n" + "=" * 60)
    print("TRAINING MODEL - MULTI-LABEL APPROACH")
    print("=" * 60)
    
    mtl_multilabel = MultiTaskCashReconciliation(
        field_names=field_names,
        shared_layers_config=[256, 128, 64],
        task_head_config=[32, 16],
        use_attention=True,
        field_prediction_approach='multilabel'
    )
    
    model_multilabel = mtl_multilabel.train(
        X_train, y_cash_train, y_field_train,
        epochs=50, batch_size=64
    )
    
    print("\nEvaluating Multi-label Approach...")
    mtl_multilabel.evaluate_model(X_test, y_cash_test, y_field_test)
    
    # Feature importance analysis
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    feature_stats = mtl_multilabel.get_feature_importance()
    if feature_stats:
        print("Feature statistics stored for interpretability analysis.")
        print("Use SHAP or LIME for detailed feature importance analysis.")
    
    # Plot training histories
    print("\nPlotting training histories...")
    mtl_independent.plot_training_history()
    mtl_multilabel.plot_training_history()
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("1. The multi-label approach typically performs better when field changes are correlated")
    print("2. Use attention weights for interpretability in production")
    print("3. Implement SHAP values for detailed feature explanations")
    print("4. Consider ensemble methods combining both approaches")
    print("5. Monitor model performance on new data and retrain periodically")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main()
