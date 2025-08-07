"""
The Core Problem
You're absolutely right that a naive multi-output wrapper wouldn't help. If you just train:

Model 1: Predicts cash_break
Model 2: Predicts is_factor_changed
Model 3: Predicts is_principle_changed

These models don't talk to each other, and the change predictions don't inform the cash break prediction. We need architectures where information flows between tasks.
Implementation Strategy 1: Stacked Generalization with Shared Learning
Architecture

# Pseudo-architecture
Stage 1: Base Models for Change Predictions
├── Model_factor_change: X → P(factor_changed)
├── Model_principle_change: X → P(principle_changed)
├── Model_coupon_change: X → P(coupon_changed)
└── ... other change models

Stage 2: Meta Model for Cash Break
Input: [X, P(factor_changed), P(principle_changed), P(coupon_changed), ...]
Output: P(cash_break)
"""

# implementation details
class StackedCashBreakPredictor:
    def __init__(self):
        # Stage 1: Change predictors
        self.change_predictors = {
            'factor': XGBClassifier(n_estimators=100),
            'principle': XGBClassifier(n_estimators=100),
            'coupon': XGBClassifier(n_estimators=100),
            # ... other fields
        }
        
        # Stage 2: Meta-model
        self.cash_break_model = XGBClassifier(n_estimators=200)
        
    def fit(self, X, y_cashbreak, y_changes_dict):
        # Step 1: Train change predictors using cross-validation
        # to avoid overfitting
        change_predictions = {}
        
        for field, model in self.change_predictors.items():
            # Use out-of-fold predictions to avoid leakage
            oof_predictions = cross_val_predict(
                model, X, y_changes_dict[field],
                cv=5, method='predict_proba'
            )[:, 1]
            change_predictions[field] = oof_predictions
            
            # Fit on full data for later use
            model.fit(X, y_changes_dict[field])
        
        # Step 2: Create augmented features
        X_augmented = np.hstack([
            X,
            np.column_stack(list(change_predictions.values()))
        ])
        
        # Step 3: Train cash break model with augmented features
        self.cash_break_model.fit(X_augmented, y_cashbreak)

"""
Key Insight: The predicted probabilities of changes become features for the cash break model, creating an information flow.

Implementation Strategy 2: Joint Neural Network Architecture
Architecture
"""
class JointCashBreakNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        
        # Shared representation layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Task-specific heads for changes
        self.factor_head = nn.Linear(hidden_dim, 1)
        self.principle_head = nn.Linear(hidden_dim, 1)
        self.coupon_head = nn.Linear(hidden_dim, 1)
        
        # Cash break head that takes both shared features 
        # AND change predictions
        self.cash_break_combiner = nn.Linear(hidden_dim + 3, 32)
        self.cash_break_head = nn.Linear(32, 1)
        
    def forward(self, x):
        # Generate shared representations
        shared = self.shared_layers(x)
        
        # Predict changes
        factor_pred = torch.sigmoid(self.factor_head(shared))
        principle_pred = torch.sigmoid(self.principle_head(shared))
        coupon_pred = torch.sigmoid(self.coupon_head(shared))
        
        # Combine for cash break prediction
        # This is the KEY: change predictions inform cash break
        cash_break_input = torch.cat([
            shared,
            factor_pred,
            principle_pred,
            coupon_pred
        ], dim=1)
        
        cash_break_hidden = torch.relu(self.cash_break_combiner(cash_break_input))
        cash_break_pred = torch.sigmoid(self.cash_break_head(cash_break_hidden))
        
        return {
            'cash_break': cash_break_pred,
            'factor_change': factor_pred,
            'principle_change': principle_pred,
            'coupon_change': coupon_pred
        }

"""
Training with Weighted Multi-Task Loss
"""

class WeightedMultiTaskLoss:
    def __init__(self, task_weights=None):
        self.task_weights = task_weights or {
            'cash_break': 0.5,  # Primary task gets highest weight
            'factor_change': 0.2,
            'principle_change': 0.2,
            'coupon_change': 0.1
        }
        self.bce = nn.BCELoss()
        
    def __call__(self, predictions, targets):
        total_loss = 0
        losses = {}
        
        for task, pred in predictions.items():
            task_loss = self.bce(pred, targets[task])
            weighted_loss = self.task_weights[task] * task_loss
            total_loss += weighted_loss
            losses[task] = task_loss.item()
            
        return total_loss, losses


"""
Implementation Strategy 3: Gradient Boosting with Chained Models
"""

class ChainedGradientBoostingMultiTask:
    def __init__(self):
        # Important: models are trained sequentially
        # Each model can use previous models' predictions
        self.model_chain = []
        self.feature_names = []
        
    def fit(self, X, y_all_targets):
        X_current = X.copy()
        
        # Step 1: Train change predictors first
        change_columns = [col for col in y_all_targets.columns 
                         if col != 'was_exception']
        
        for change_col in change_columns:
            model = LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.05
            )
            
            model.fit(X_current, y_all_targets[change_col])
            
            # Add predictions as features for next models
            change_probs = model.predict_proba(X_current)[:, 1].reshape(-1, 1)
            X_current = np.hstack([X_current, change_probs])
            
            self.model_chain.append({
                'name': change_col,
                'model': model,
                'type': 'auxiliary'
            })
        
        # Step 2: Train cash break model with all features + predictions
        final_model = LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            # This ensures it focuses on the augmented features
            feature_fraction=0.8,
            # Can set higher importance to predicted change features
            # through feature engineering
        )
        
        final_model.fit(X_current, y_all_targets['was_exception'])
        
        self.model_chain.append({
            'name': 'cash_break',
            'model': final_model,
            'type': 'primary'
        })
    
    def predict_with_explanations(self, X):
        X_current = X.copy()
        predictions = {}
        feature_contributions = {}
        
        # Forward pass through the chain
        for step in self.model_chain:
            if step['type'] == 'auxiliary':
                pred_proba = step['model'].predict_proba(X_current)[:, 1]
                predictions[step['name']] = pred_proba
                X_current = np.hstack([X_current, pred_proba.reshape(-1, 1)])
            else:
                # For final model, we can extract feature importance
                predictions['cash_break'] = step['model'].predict_proba(X_current)[:, 1]
                
                # Get feature importance for the augmented features
                importance = step['model'].feature_importances_
                # Last N features are our change predictions
                n_change_features = len(self.model_chain) - 1
                change_importance = importance[-n_change_features:]
                
                feature_contributions = dict(zip(
                    [m['name'] for m in self.model_chain[:-1]], 
                    change_importance
                ))
        
        return predictions, feature_contributions


"""
Implementation Strategy 4: Custom XGBoost with Multi-Task Objective
Custom Objective Function
"""
import xgboost as xgb
import numpy as np

class MultiTaskXGBoost:
    def __init__(self, n_tasks, task_weights=None):
        self.n_tasks = n_tasks
        self.task_weights = task_weights or [1.0] * n_tasks
        
    def multi_task_objective(self, preds, dtrain):
        """
        Custom objective that handles multiple tasks
        preds: [n_samples * n_tasks] flattened predictions
        """
        labels = dtrain.get_label()  # [n_samples * n_tasks] flattened
        n_samples = len(labels) // self.n_tasks
        
        # Reshape for easier handling
        preds = preds.reshape(n_samples, self.n_tasks)
        labels = labels.reshape(n_samples, self.n_tasks)
        
        # Gradient and Hessian for each task
        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)
        
        for task_idx in range(self.n_tasks):
            # Logistic loss gradient for each task
            task_preds = 1.0 / (1.0 + np.exp(-preds[:, task_idx]))
            grad[:, task_idx] = (task_preds - labels[:, task_idx]) * self.task_weights[task_idx]
            hess[:, task_idx] = task_preds * (1.0 - task_preds) * self.task_weights[task_idx]
            
            # Special handling for cash break task (task 0)
            if task_idx == 0:
                # Add gradient contribution from predicted changes
                # This creates dependency between tasks
                for change_idx in range(1, self.n_tasks):
                    change_pred = 1.0 / (1.0 + np.exp(-preds[:, change_idx]))
                    # If change is predicted, increase cash break gradient
                    grad[:, task_idx] += 0.1 * change_pred * (1 - labels[:, task_idx])
        
        return grad.flatten(), hess.flatten()
    
    def fit(self, X, y_multi):
        """
        y_multi: DataFrame with columns [was_exception, is_factor_changed, ...]
        """
        # Flatten labels for multi-task learning
        y_flat = y_multi.values.flatten()
        
        dtrain = xgb.DMatrix(X, label=y_flat)
        
        params = {
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.05,
            'seed': 42
        }
        
        # Train with custom objective
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            obj=self.multi_task_objective
        )


"""
Strategy 5: Hierarchical Multi-Task Learning (Most Sophisticated)
"""
class HierarchicalCashBreakModel:
    """
    Creates a hierarchy:
    Level 1: Group related changes (e.g., payment-related, security-related)
    Level 2: Individual change predictions
    Level 3: Cash break prediction using all information
    """
    
    def __init__(self):
        self.hierarchy = {
            'payment_changes': {
                'children': ['factor_change', 'principle_change', 'coupon_change'],
                'model': None,
                'combiner': None
            },
            'security_changes': {
                'children': ['rating_change', 'sector_change'],
                'model': None,
                'combiner': None
            }
        }
        self.final_model = None
        
    def fit(self, X, y_targets):
        all_group_features = []
        
        # Level 1 & 2: Train hierarchical change predictors
        for group_name, group_info in self.hierarchy.items():
            group_predictions = []
            
            # Train individual change models
            for child in group_info['children']:
                model = XGBClassifier(n_estimators=50)
                model.fit(X, y_targets[child])
                pred = model.predict_proba(X)[:, 1]
                group_predictions.append(pred)
            
            # Create group-level features
            group_array = np.column_stack(group_predictions)
            
            # Train group combiner (learns patterns in changes)
            group_info['combiner'] = XGBClassifier(n_estimators=30)
            # Create synthetic group target (any change in group)
            group_target = (group_array.max(axis=1) > 0.5).astype(int)
            group_info['combiner'].fit(group_array, group_target)
            
            # Generate group-level representation
            group_feature = group_info['combiner'].predict_proba(group_array)[:, 1]
            all_group_features.append(group_feature)
            all_group_features.extend(group_predictions)
        
        # Level 3: Final cash break model
        X_final = np.hstack([
            X,
            np.column_stack(all_group_features)
        ])
        
        self.final_model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.03
        )
        self.final_model.fit(X_final, y_targets['was_exception'])


"""
Key Insights for Root Cause Identification
1. Feature Attribution in Multi-Task Context
"""
def get_root_causes(model, X, prediction_threshold=0.7):
    """
    Identifies root causes by analyzing which change predictions
    most influenced the cash break prediction
    """
    predictions = model.predict_all_tasks(X)
    
    root_causes = []
    for i, row in enumerate(X):
        if predictions['cash_break'][i] > prediction_threshold:
            # Find which changes are predicted
            active_changes = {
                task: prob for task, prob in predictions.items()
                if task != 'cash_break' and prob[i] > 0.5
            }
            
            # Use SHAP to understand contribution
            explainer = shap.TreeExplainer(model.final_model)
            shap_values = explainer.shap_values(row)
            
            # Map SHAP values to change predictions
            cause_ranking = sorted(
                active_changes.items(),
                key=lambda x: abs(shap_values[feature_index_map[x[0]]]),
                reverse=True
            )
            
            root_causes.append({
                'record_id': i,
                'cash_break_prob': predictions['cash_break'][i],
                'primary_cause': cause_ranking[0] if cause_ranking else None,
                'all_causes': cause_ranking
            })
    
    return root_causes


"""
2. Conditional Dependencies
The multi-task approach naturally captures conditional dependencies:

"Cash breaks happen when factor_change=1 AND principle_change=1"
"Coupon changes only matter when payment_date is near"

These patterns emerge naturally in the shared representations or through the augmented features.
3. Uncertainty Quantification
"""
class UncertaintyAwareMultiTask:
    def predict_with_confidence(self, X):
        # Run multiple forward passes with dropout
        predictions = []
        for _ in range(100):
            pred = self.model.predict(X, training=True)  # Keep dropout on
            predictions.append(pred)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # High uncertainty in change predictions = unclear root cause
        root_cause_confidence = 1.0 / (1.0 + std_pred.mean())
        
        return mean_pred, root_cause_confidence


"""
Practical Recommendations

- Start with Strategy 1 (Stacked Generalization): It's interpretable and works with any base algorithm.
- For production, consider Strategy 3 (Chained Models): It's robust and provides clear feature importance.
- If you have lots of data, Strategy 2 (Neural Network): Offers the most flexibility for complex patterns.
- For maximum interpretability: Combine any strategy with rule extraction:

# Extract rules from trained model
rules = extract_decision_rules(model, max_depth=3)
# "IF factor_change_prob > 0.8 AND principle_change_prob > 0.6 THEN cash_break_prob = 0.92"
"""


