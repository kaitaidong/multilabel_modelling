"""
Alternative Architectures for Cash Reconciliation Task
=====================================================

This script proposes and implements several alternative architectures
that could work well with the cash reconciliation problem:

1. Transformer-based Multi-Task Architecture
2. Gradient Boosting Multi-Output Ensemble
3. Hierarchical Task Learning
4. Graph Neural Network for Transaction Relationships
5. Hybrid CNN-LSTM for Sequential Transaction Data
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
import networkx as nx
from typing import Dict, List, Tuple, Optional

# Alternative Architecture 1: Transformer-based Multi-Task Learning
class TransformerMultiTask:
    """
    Transformer-based architecture for handling sequential transaction data
    and complex feature interactions
    """
    
    def __init__(self, 
                 field_names: List[str],
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dff: int = 512):
        
        self.field_names = field_names
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
    
    def positional_encoding(self, position, d_model):
        """
        Create positional encoding for transformer
        """
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                   np.arange(d_model)[np.newaxis, :],
                                   d_model)
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
        Calculate the attention weights and apply to values
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add the mask to the scaled tensor
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def multi_head_attention(self, v, k, q, mask=None):
        """
        Multi-head attention mechanism
        """
        batch_size = tf.shape(q)[0]
        
        # Linear layers for Q, K, V
        q = layers.Dense(self.d_model)(q)
        k = layers.Dense(self.d_model)(k)
        v = layers.Dense(self.d_model)(v)
        
        # Split into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Apply attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        # Concatenate heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))
        
        # Final linear layer
        output = layers.Dense(self.d_model)(concat_attention)
        
        return output, attention_weights
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.d_model // self.num_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def build_model(self, input_dim: int, sequence_length: int = 10):
        """
        Build transformer-based multi-task model
        """
        # Input layer
        inputs = Input(shape=(sequence_length, input_dim))
        
        # Input projection to d_model
        x = layers.Dense(self.d_model)(inputs)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding(sequence_length, self.d_model)
        x += pos_encoding[:, :sequence_length, :]
        
        # Transformer layers
        for _ in range(self.num_layers):
            # Multi-head attention
            attn_output, _ = self.multi_head_attention(x, x, x)
            attn_output = layers.Dropout(0.1)(attn_output)
            out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
            
            # Feed forward network
            ffn_output = layers.Dense(self.dff, activation='relu')(out1)
            ffn_output = layers.Dense(self.d_model)(ffn_output)
            ffn_output = layers.Dropout(0.1)(ffn_output)
            x = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        # Global average pooling
        pooled = layers.GlobalAveragePooling1D()(x)
        
        # Task-specific heads
        # Cash break prediction
        cash_break_head = layers.Dense(64, activation='relu')(pooled)
        cash_break_head = layers.Dropout(0.3)(cash_break_head)
        cash_break_output = layers.Dense(1, activation='sigmoid', name='cash_break')(cash_break_head)
        
        # Field changes prediction
        field_changes_head = layers.Dense(64, activation='relu')(pooled)
        field_changes_head = layers.Dropout(0.3)(field_changes_head)
        field_changes_output = layers.Dense(
            len(self.field_names), 
            activation='sigmoid', 
            name='field_changes'
        )(field_changes_head)
        
        model = Model(inputs=inputs, outputs=[cash_break_output, field_changes_output])
        
        return model

# Alternative Architecture 2: Gradient Boosting Multi-Output Ensemble
class GradientBoostingMultiTask:
    """
    Ensemble of gradient boosting models for multi-task learning
    """
    
    def __init__(self, field_names: List[str]):
        self.field_names = field_names
        self.cash_break_model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Individual models for each field
        self.field_models = {}
        for field_name in field_names:
            self.field_models[field_name] = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        # Multi-output model for field changes
        self.field_changes_model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        )
    
    def create_ensemble_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create ensemble features using feature interactions
        """
        X_ensemble = X.copy()
        
        # Add feature interactions
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            for i, col1 in enumerate(numerical_cols[:5]):  # Limit to prevent explosion
                for col2 in numerical_cols[i+1:6]:
                    X_ensemble[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                    X_ensemble[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
        
        # Add statistical features
        X_ensemble['num_missing'] = X.isnull().sum(axis=1)
        X_ensemble['num_features_mean'] = X[numerical_cols].mean(axis=1)
        X_ensemble['num_features_std'] = X[numerical_cols].std(axis=1)
        
        return X_ensemble
    
    def train(self, X: pd.DataFrame, y_cash_break: np.ndarray, y_field_changes: np.ndarray):
        """
        Train ensemble of gradient boosting models
        """
        # Create ensemble features
        X_ensemble = self.create_ensemble_features(X)
        
        # Handle categorical variables
        X_processed = pd.get_dummies(X_ensemble, drop_first=True)
        X_processed = X_processed.fillna(X_processed.median())
        
        # Train cash break model
        print("Training cash break model...")
        self.cash_break_model.fit(X_processed, y_cash_break)
        
        # Get cash break predictions as features for field models
        cash_break_probs = self.cash_break_model.predict_proba(X_processed)[:, 1]
        X_with_cash_pred = X_processed.copy()
        X_with_cash_pred['cash_break_prob'] = cash_break_probs
        
        # Train individual field models
        print("Training individual field models...")
        for i, field_name in enumerate(self.field_names):
            self.field_models[field_name].fit(X_with_cash_pred, y_field_changes[:, i])
        
        # Train multi-output field changes model
        print("Training multi-output field changes model...")
        self.field_changes_model.fit(X_with_cash_pred, y_field_changes)
        
        self.X_processed = X_processed
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the ensemble
        """
        # Create ensemble features and process
        X_ensemble = self.create_ensemble_features(X)
        X_processed = pd.get_dummies(X_ensemble, drop_first=True)
        
        # Align columns with training data
        for col in self.X_processed.columns:
            if col not in X_processed.columns:
                X_processed[col] = 0
        X_processed = X_processed[self.X_processed.columns].fillna(0)
        
        # Predict cash breaks
        cash_break_probs = self.cash_break_model.predict_proba(X_processed)[:, 1]
        
        # Add cash break predictions as features
        X_with_cash_pred = X_processed.copy()
        X_with_cash_pred['cash_break_prob'] = cash_break_probs
        
        # Predict field changes using both approaches
        field_predictions_individual = np.zeros((len(X), len(self.field_names)))
        for i, field_name in enumerate(self.field_names):
            field_predictions_individual[:, i] = self.field_models[field_name].predict_proba(
                X_with_cash_pred)[:, 1]
        
        field_predictions_multioutput = self.field_changes_model.predict(X_with_cash_pred)
        
        # Ensemble the field predictions
        field_predictions = (field_predictions_individual + field_predictions_multioutput) / 2
        
        return cash_break_probs, field_predictions

# Alternative Architecture 3: Hierarchical Task Learning
class HierarchicalTaskLearning:
    """
    Hierarchical approach where cash break prediction informs field change predictions
    """
    
    def __init__(self, field_names: List[str]):
        self.field_names = field_names
        self.level1_model = None  # Cash break prediction
        self.level2_models = {}   # Field-specific models conditioned on cash break
    
    def build_level1_model(self, input_dim: int) -> Model:
        """
        Build Level 1 model for cash break prediction
        """
        inputs = Input(shape=(input_dim,))
        
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output cash break probability and learned representation
        cash_break_output = layers.Dense(1, activation='sigmoid', name='cash_break')(x)
        representation_output = layers.Dense(32, activation='relu', name='representation')(x)
        
        model = Model(inputs=inputs, outputs=[cash_break_output, representation_output])
        
        return model
    
    def build_level2_model(self, input_dim: int, representation_dim: int = 32) -> Model:
        """
        Build Level 2 model for field changes prediction, conditioned on Level 1
        """
        # Original features
        feature_inputs = Input(shape=(input_dim,), name='features')
        
        # Cash break probability from Level 1
        cash_break_input = Input(shape=(1,), name='cash_break_prob')
        
        # Learned representation from Level 1
        representation_input = Input(shape=(representation_dim,), name='representation')
        
        # Combine all inputs
        combined = layers.Concatenate()([feature_inputs, cash_break_input, representation_input])
        
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Field-specific outputs
        field_outputs = []
        for field_name in self.field_names:
            field_head = layers.Dense(32, activation='relu')(x)
            field_output = layers.Dense(1, activation='sigmoid', name=f'field_{field_name}')(field_head)
            field_outputs.append(field_output)
        
        model = Model(
            inputs=[feature_inputs, cash_break_input, representation_input],
            outputs=field_outputs
        )
        
        return model
    
    def train_hierarchical(self, X: pd.DataFrame, 
                          y_cash_break: np.ndarray, 
                          y_field_changes: np.ndarray):
        """
        Train hierarchical models
        """
        # Prepare data
        X_processed = pd.get_dummies(X, drop_first=True).fillna(0)
        
        # Split data
        X_train, X_val, y_cash_train, y_cash_val, y_field_train, y_field_val = train_test_split(
            X_processed, y_cash_break, y_field_changes, test_size=0.2, random_state=42
        )
        
        # Train Level 1 model
        print("Training Level 1 model (Cash Break Prediction)...")
        self.level1_model = self.build_level1_model(X_processed.shape[1])
        self.level1_model.compile(
            optimizer='adam',
            loss={'cash_break': 'binary_crossentropy', 'representation': 'mse'},
            metrics={'cash_break': 'accuracy'},
            loss_weights={'cash_break': 1.0, 'representation': 0.1}
        )
        
        # Create dummy target for representation (self-supervised)
        representation_target = np.random.randn(len(X_train), 32) * 0.1
        representation_target_val = np.random.randn(len(X_val), 32) * 0.1
        
        self.level1_model.fit(
            X_train,
            {'cash_break': y_cash_train, 'representation': representation_target},
            validation_data=(X_val, {'cash_break': y_cash_val, 'representation': representation_target_val}),
            epochs=50,
            batch_size=64,
            verbose=1
        )
        
        # Get Level 1 predictions for Level 2 training
        print("Generating Level 1 predictions for Level 2...")
        cash_break_pred_train, representation_train = self.level1_model.predict(X_train)
        cash_break_pred_val, representation_val = self.level1_model.predict(X_val)
        
        # Train Level 2 model
        print("Training Level 2 model (Field Changes Prediction)...")
        self.level2_model = self.build_level2_model(X_processed.shape[1])
        
        # Prepare Level 2 targets
        level2_targets_train = {f'field_{field_name}': y_field_train[:, i] 
                               for i, field_name in enumerate(self.field_names)}
        level2_targets_val = {f'field_{field_name}': y_field_val[:, i] 
                             for i, field_name in enumerate(self.field_names)}
        
        self.level2_model.compile(
            optimizer='adam',
            loss={f'field_{field_name}': 'binary_crossentropy' for field_name in self.field_names},
            metrics={f'field_{field_name}': 'accuracy' for field_name in self.field_names}
        )
        
        self.level2_model.fit(
            [X_train, cash_break_pred_train, representation_train],
            level2_targets_train,
            validation_data=([X_val, cash_break_pred_val, representation_val], level2_targets_val),
            epochs=50,
            batch_size=64,
            verbose=1
        )
        
        self.X_columns = X_processed.columns

# Alternative Architecture 4: Graph Neural Network for Transaction Relationships
class GraphNeuralNetworkMultiTask:
    """
    Graph Neural Network that captures relationships between transactions,
    securities, and entities in the cash reconciliation process
    """
    
    def __init__(self, field_names: List[str]):
        self.field_names = field_names
        self.graph = None
        self.node_features = None
    
    def create_transaction_graph(self, X: pd.DataFrame) -> nx.Graph:
        """
        Create a graph where nodes are transactions and edges represent relationships
        """
        G = nx.Graph()
        
        # Add transaction nodes
        for idx in X.index:
            G.add_node(f"txn_{idx}", type='transaction', **X.loc[idx].to_dict())
        
        # Add security nodes
        if 'security_id' in X.columns:
            securities = X['security_id'].unique()
            for security in securities:
                if pd.notna(security):
                    G.add_node(f"sec_{security}", type='security')
        
        # Add custodian nodes
        if 'custodian' in X.columns:
            custodians = X['custodian'].unique()
            for custodian in custodians:
                if pd.notna(custodian):
                    G.add_node(f"cust_{custodian}", type='custodian')
        
        # Add portfolio nodes
        if 'portfolio' in X.columns:
            portfolios = X['portfolio'].unique()
            for portfolio in portfolios:
                if pd.notna(portfolio):
                    G.add_node(f"port_{portfolio}", type='portfolio')
        
        # Create edges based on relationships
        for idx, row in X.iterrows():
            txn_node = f"txn_{idx}"
            
            # Connect transactions to securities
            if 'security_id' in row and pd.notna(row['security_id']):
                G.add_edge(txn_node, f"sec_{row['security_id']}", relation='trades')
            
            # Connect transactions to custodians
            if 'custodian' in row and pd.notna(row['custodian']):
                G.add_edge(txn_node, f"cust_{row['custodian']}", relation='custodial')
            
            # Connect transactions to portfolios
            if 'portfolio' in row and pd.notna(row['portfolio']):
                G.add_edge(txn_node, f"port_{row['portfolio']}", relation='belongs_to')
        
        # Add edges between transactions with similar characteristics
        transaction_nodes = [n for n in G.nodes() if n.startswith('txn_')]
        
        for i, txn1 in enumerate(transaction_nodes[:100]):  # Limit for performance
            for txn2 in transaction_nodes[i+1:101]:
                # Calculate similarity based on features
                idx1 = int(txn1.split('_')[1])
                idx2 = int(txn2.split('_')[1])
                
                if idx1 in X.index and idx2 in X.index:
                    row1, row2 = X.loc[idx1], X.loc[idx2]
                    
                    # Same security type
                    if ('security_type' in row1 and 'security_type' in row2 and 
                        row1['security_type'] == row2['security_type'] and pd.notna(row1['security_type'])):
                        G.add_edge(txn1, txn2, relation='similar_type', weight=0.5)
                    
                    # Same currency
                    if ('currency' in row1 and 'currency' in row2 and 
                        row1['currency'] == row2['currency'] and pd.notna(row1['currency'])):
                        G.add_edge(txn1, txn2, relation='same_currency', weight=0.3)
        
        return G
    
    def build_gnn_model(self, node_feature_dim: int, num_nodes: int):
        """
        Build Graph Neural Network model using message passing
        """
        # Node features input
        node_features = Input(shape=(num_nodes, node_feature_dim), name='node_features')
        
        # Adjacency matrix input
        adjacency = Input(shape=(num_nodes, num_nodes), name='adjacency_matrix')
        
        # Graph convolution layers
        x = node_features
        for i in range(3):
            # Message passing: aggregate neighbor features
            messages = layers.Dense(64, activation='relu', name=f'message_{i}')(x)
            
            # Apply adjacency matrix to aggregate messages
            aggregated = tf.matmul(adjacency, messages)
            
            # Update node representations
            x = layers.Dense(64, activation='relu', name=f'update_{i}')(
                layers.Concatenate()([x, aggregated])
            )
            x = layers.Dropout(0.2)(x)
        
        # Global pooling to get graph-level representation
        graph_representation = layers.GlobalAveragePooling1D()(x)
        
        # Task-specific heads
        # Cash break prediction
        cash_break_head = layers.Dense(64, activation='relu')(graph_representation)
        cash_break_output = layers.Dense(1, activation='sigmoid', name='cash_break')(cash_break_head)
        
        # Field changes prediction
        field_changes_head = layers.Dense(64, activation='relu')(graph_representation)
        field_changes_output = layers.Dense(
            len(self.field_names), 
            activation='sigmoid', 
            name='field_changes'
        )(field_changes_head)
        
        model = Model(
            inputs=[node_features, adjacency],
            outputs=[cash_break_output, field_changes_output]
        )
        
        return model

# Alternative Architecture 5: Hybrid CNN-LSTM for Sequential Transaction Data
class CNNLSTMMultiTask:
    """
    Hybrid CNN-LSTM architecture for capturing both local patterns and temporal dependencies
    """
    
    def __init__(self, field_names: List[str]):
        self.field_names = field_names
    
    def prepare_sequential_data(self, X: pd.DataFrame, sequence_length: int = 10):
        """
        Prepare data in sequential format for time-series modeling
        """
        # Sort by timestamp if available, otherwise by index
        if 'timestamp' in X.columns:
            X_sorted = X.sort_values('timestamp')
        else:
            X_sorted = X.sort_index()
        
        # Create sequences
        sequences = []
        for i in range(len(X_sorted) - sequence_length + 1):
            sequence = X_sorted.iloc[i:i+sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def build_cnn_lstm_model(self, input_shape: Tuple[int, int]):
        """
        Build hybrid CNN-LSTM model
        """
        inputs = Input(shape=input_shape)
        
        # 1D CNN layers for local pattern detection
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.Dropout(0.3)(x)
        
        # LSTM layers for temporal dependencies
        x = layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
        x = layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
        
        # Shared dense layers
        shared = layers.Dense(128, activation='relu')(x)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)
        
        shared = layers.Dense(64, activation='relu')(shared)
        shared = layers.Dropout(0.2)(shared)
        
        # Task-specific heads
        # Cash break prediction
        cash_break_head = layers.Dense(32, activation='relu')(shared)
        cash_break_output = layers.Dense(1, activation='sigmoid', name='cash_break')(cash_break_head)
        
        # Field changes prediction with attention
        field_attention = layers.Dense(len(self.field_names), activation='softmax')(shared)
        field_context = layers.Multiply()([shared, field_attention])
        
        field_changes_head = layers.Dense(32, activation='relu')(field_context)
        field_changes_output = layers.Dense(
            len(self.field_names), 
            activation='sigmoid', 
            name='field_changes'
        )(field_changes_head)
        
        model = Model(inputs=inputs, outputs=[cash_break_output, field_changes_output])
        
        return model

# Alternative Architecture 6: Meta-Learning Approach
class MetaLearningMultiTask:
    """
    Meta-learning approach that learns to quickly adapt to new data patterns
    """
    
    def __init__(self, field_names: List[str]):
        self.field_names = field_names
        self.meta_model = None
        self.task_models = {}
    
    def build_meta_model(self, input_dim: int):
        """
        Build meta-model that generates parameters for task-specific models
        """
        inputs = Input(shape=(input_dim,))
        
        # Meta-learner network
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Generate parameters for cash break model
        cash_break_params = layers.Dense(64, activation='tanh', name='cash_break_params')(x)
        
        # Generate parameters for field change models
        field_change_params = {}
        for field_name in self.field_names:
            params = layers.Dense(32, activation='tanh', name=f'field_params_{field_name}')(x)
            field_change_params[field_name] = params
        
        # Task predictions using generated parameters
        cash_break_features = layers.Dense(32, activation='relu')(inputs)
        cash_break_weighted = layers.Multiply()([cash_break_features, cash_break_params[:, :32]])
        cash_break_output = layers.Dense(1, activation='sigmoid', name='cash_break')(cash_break_weighted)
        
        field_outputs = []
        for field_name in self.field_names:
            field_features = layers.Dense(16, activation='relu')(inputs)
            field_weighted = layers.Multiply()([field_features, field_change_params[field_name][:, :16]])
            field_output = layers.Dense(1, activation='sigmoid', name=f'field_{field_name}')(field_weighted)
            field_outputs.append(field_output)
        
        outputs = [cash_break_output] + field_outputs
        model = Model(inputs=inputs, outputs=outputs)
        
        return model

# Ensemble Architecture: Combining Multiple Approaches
class EnsembleMultiTask:
    """
    Ensemble that combines predictions from multiple architectures
    """
    
    def __init__(self, field_names: List[str]):
        self.field_names = field_names
        self.models = {}
        self.weights = {}
    
    def add_model(self, name: str, model, weight: float = 1.0):
        """
        Add a model to the ensemble
        """
        self.models[name] = model
        self.weights[name] = weight
    
    def train_ensemble(self, X: pd.DataFrame, 
                      y_cash_break: np.ndarray, 
                      y_field_changes: np.ndarray):
        """
        Train all models in the ensemble
        """
        for name, model in self.models.items():
            print(f"Training {name}...")
            if hasattr(model, 'train'):
                model.train(X, y_cash_break, y_field_changes)
            elif hasattr(model, 'fit'):
                X_processed = pd.get_dummies(X, drop_first=True).fillna(0)
                model.fit(X_processed, np.column_stack([y_cash_break, y_field_changes]))
    
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions
        """
        cash_break_predictions = []
        field_change_predictions = []
        
        total_weight = sum(self.weights.values())
        
        for name, model in self.models.items():
            weight = self.weights[name] / total_weight
            
            if hasattr(model, 'predict'):
                if name == 'gradient_boosting':
                    cash_pred, field_pred = model.predict(X)
                else:
                    X_processed = pd.get_dummies(X, drop_first=True).fillna(0)
                    predictions = model.predict(X_processed)
                    cash_pred = predictions[:, 0]
                    field_pred = predictions[:, 1:]
                
                cash_break_predictions.append(cash_pred * weight)
                field_change_predictions.append(field_pred * weight)
        
        ensemble_cash_pred = np.sum(cash_break_predictions, axis=0)
        ensemble_field_pred = np.sum(field_change_predictions, axis=0)
        
        return ensemble_cash_pred, ensemble_field_pred

# Demonstration and Comparison Function
def compare_architectures():
    """
    Compare different architectures on synthetic data
    """
    print("Comparing Alternative Architectures for Cash Reconciliation")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 2000
    n_features = 50
    
    # Generate mixed data
    X_num = np.random.randn(n_samples, 30)
    X_cat = pd.DataFrame({
        'security_type': np.random.choice(['bond', 'equity', 'derivative'], n_samples),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], n_samples),
        'custodian': np.random.choice(['BNY', 'SSGA', 'JPM'], n_samples),
    })
    
    X = pd.concat([
        pd.DataFrame(X_num, columns=[f'num_feature_{i}' for i in range(30)]),
        X_cat
    ], axis=1)
    
    field_names = ['previous_coupon', 'factor', 'previous_factor', 'interest', 'principles', 'date']
    
    # Create target variables
    y_cash_break = np.random.binomial(1, 0.2, n_samples)
    y_field_changes = np.random.binomial(1, 0.15, (n_samples, len(field_names)))
    
    # Split data
    X_train, X_test, y_cash_train, y_cash_test, y_field_train, y_field_test = train_test_split(
        X, y_cash_break, y_field_changes, test_size=0.2, random_state=42
    )
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Initialize architectures
    architectures = {
        'Gradient Boosting': GradientBoostingMultiTask(field_names),
        'Hierarchical': HierarchicalTaskLearning(field_names),
    }
    
    results = {}
    
    # Train and evaluate each architecture
    for name, model in architectures.items():
        print(f"\n{'='*40}")
        print(f"Training {name} Architecture")
        print(f"{'='*40}")
        
        try:
            if name == 'Gradient Boosting':
                model.train(X_train, y_cash_train, y_field_train)
                cash_pred, field_pred = model.predict(X_test)
                
                # Evaluate
                cash_accuracy = accuracy_score(y_cash_test, (cash_pred > 0.5).astype(int))
                field_hamming = hamming_loss(y_field_test, (field_pred > 0.5).astype(int))
                
                results[name] = {
                    'cash_accuracy': cash_accuracy,
                    'field_hamming_loss': field_hamming
                }
                
            elif name == 'Hierarchical':
                model.train_hierarchical(X_train, y_cash_train, y_field_train)
                # Note: Prediction method would need to be implemented
                results[name] = {'status': 'trained successfully'}
            
            print(f"{name} training completed.")
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            results[name] = {'error': str(e)}
    
    # Print results summary
    print(f"\n{'='*60}")
    print("ARCHITECTURE COMPARISON RESULTS")
    print(f"{'='*60}")
    
    for name, result in results.items():
        print(f"\n{name}:")
        if 'cash_accuracy' in result:
            print(f"  Cash Break Accuracy: {result['cash_accuracy']:.4f}")
            print(f"  Field Changes Hamming Loss: {result['field_hamming_loss']:.4f}")
        elif 'status' in result:
            print(f"  Status: {result['status']}")
        elif 'error' in result:
            print(f"  Error: {result['error']}")
    
    print(f"\n{'='*60}")
    print("ARCHITECTURE RECOMMENDATIONS")
    print(f"{'='*60}")
    print("1. Transformer Architecture: Best for complex feature interactions")
    print("2. Gradient Boosting: Strong baseline with good interpretability")
    print("3. Hierarchical Learning: When cash breaks strongly influence field changes")
    print("4. Graph Neural Networks: When transaction relationships are important")
    print("5. CNN-LSTM: For temporal patterns in transaction sequences")
    print("6. Meta-Learning: For adaptation to new data patterns")
    print("7. Ensemble: Combines strengths of multiple approaches")
    
    return results

if __name__ == "__main__":
    compare_architectures()
