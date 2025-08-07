"""
Rule Extraction: Converting Complex Models into Interpretable Logic
Rule extraction is the process of converting complex ML models (like neural networks or ensemble methods) into human-readable IF-THEN rules. This is crucial for your cash reconciliation use case because it helps explain to the Cash team exactly why a break is predicted.
Why Rule Extraction Matters for Your Use Case
Instead of saying "the model predicts 0.89 probability of cash break," you can say:
"Cash break likely because:
- Factor payment changed in last 3 days (happened in 87% of similar cases)
- AND principle amount > $1M 
- AND custodian is 'XYZ Bank'
→ Historical accuracy of this pattern: 92%"


Implementation Strategies
Strategy 1: Direct Rule Extraction from Decision Trees
The simplest approach - train a decision tree to mimic your complex model, then extract its rules.
"""

from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np

class DecisionTreeRuleExtractor:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def extract_rules_from_tree(self, tree, min_samples=50, min_confidence=0.8):
        """
        Extract human-readable rules from a decision tree
        
        Args:
            tree: Trained DecisionTreeClassifier
            min_samples: Minimum samples in leaf for rule to be considered
            min_confidence: Minimum confidence (purity) of the rule
        """
        tree_ = tree.tree_
        feature_names = self.feature_names
        
        rules = []
        
        def recurse(node, path, path_directions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                # Internal node
                feature_id = tree_.feature[node]
                threshold = tree_.threshold[node]
                feature_name = feature_names[feature_id]
                
                # Go left (<=)
                left_path = path + [(feature_name, '<=', threshold)]
                left_dirs = path_directions + ['left']
                recurse(tree_.children_left[node], left_path, left_dirs)
                
                # Go right (>)
                right_path = path + [(feature_name, '>', threshold)]
                right_dirs = path_directions + ['right']
                recurse(tree_.children_right[node], right_path, right_dirs)
            else:
                # Leaf node - extract rule if it meets criteria
                samples = tree_.n_node_samples[node]
                value = tree_.value[node][0]
                class_probability = value[1] / value.sum()  # Probability of positive class
                
                if samples >= min_samples and class_probability >= min_confidence:
                    rule = {
                        'conditions': path,
                        'confidence': class_probability,
                        'support': samples,
                        'prediction': 1 if class_probability > 0.5 else 0
                    }
                    rules.append(rule)
        
        recurse(0, [], [])
        return rules
    
    def format_rule(self, rule):
        """Convert rule dict to readable string"""
        conditions_str = " AND ".join([
            f"{feat} {op} {val:.3f}" if isinstance(val, float) 
            else f"{feat} {op} {val}"
            for feat, op, val in rule['conditions']
        ])
        
        return (f"IF {conditions_str} "
                f"THEN cash_break = {rule['prediction']} "
                f"(confidence: {rule['confidence']:.2%}, "
                f"support: {rule['support']} samples)")

# Usage example
def extract_rules_from_complex_model(complex_model, X_train, y_train, feature_names):
    """
    Train a decision tree to mimic complex model, then extract rules
    """
    # Get predictions from complex model
    complex_predictions = complex_model.predict_proba(X_train)[:, 1]
    
    # Train decision tree to mimic these predictions
    # Using soft labels (probabilities) instead of hard labels
    mimic_tree = DecisionTreeClassifier(
        max_depth=5,  # Keep it simple for interpretability
        min_samples_leaf=50,  # Ensure rules have support
        min_samples_split=100
    )
    
    # Discretize predictions for tree training
    mimic_labels = (complex_predictions > 0.5).astype(int)
    mimic_tree.fit(X_train, mimic_labels)
    
    # Extract rules
    extractor = DecisionTreeRuleExtractor(feature_names)
    rules = extractor.extract_rules_from_tree(mimic_tree)
    
    # Sort by confidence * support (importance)
    rules.sort(key=lambda x: x['confidence'] * x['support'], reverse=True)
    
    return rules, mimic_tree


"""
Strategy 2: Rule Extraction via LIME/Anchors
Anchors is specifically designed for rule extraction and works with any black-box model.
"""

from anchor import anchor_tabular
import numpy as np

class AnchorRuleExtractor:
    def __init__(self, model, X_train, feature_names, categorical_features=None):
        """
        Initialize Anchor explainer for rule extraction
        """
        self.model = model
        self.feature_names = feature_names
        
        # Initialize Anchors
        self.explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=['no_cash_break', 'cash_break'],
            feature_names=feature_names,
            train_data=X_train,
            categorical_names=categorical_features or {}
        )
        
    def extract_rule_for_instance(self, x_instance, threshold=0.9):
        """
        Extract rule that explains why this instance has its prediction
        """
        # Get model prediction
        prediction = self.model.predict([x_instance])[0]
        
        # Get anchor (rule) explanation
        explanation = self.explainer.explain_instance(
            x_instance,
            self.model.predict,
            threshold=threshold,  # Minimum precision of anchor
            max_anchor_size=5  # Maximum number of conditions
        )
        
        return {
            'rule': ' AND '.join(explanation.names()),
            'precision': explanation.precision(),
            'coverage': explanation.coverage(),
            'prediction': prediction
        }
    
    def extract_global_rules(self, X_sample, n_rules=20):
        """
        Extract global rules by finding anchors for diverse instances
        """
        # Select diverse instances using clustering
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_rules, random_state=42)
        clusters = kmeans.fit_predict(X_sample)
        
        rules = []
        for cluster_id in range(n_rules):
            # Get representative instance from each cluster
            cluster_mask = clusters == cluster_id
            cluster_instances = X_sample[cluster_mask]
            
            # Use centroid or high-confidence prediction instance
            representative = cluster_instances[0]
            
            try:
                rule = self.extract_rule_for_instance(representative)
                rule['cluster_size'] = cluster_mask.sum()
                rules.append(rule)
            except:
                continue
                
        return rules


"""
Strategy 3: Association Rule Mining for Pattern Discovery
This finds combinations of conditions that frequently lead to cash breaks.
"""

from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

class AssociationRuleMiner:
    def __init__(self, min_support=0.01, min_confidence=0.8):
        self.min_support = min_support
        self.min_confidence = min_confidence
        
    def discretize_features(self, X, feature_names, n_bins=3):
        """
        Convert continuous features to discrete for rule mining
        """
        X_discrete = pd.DataFrame()
        
        for i, fname in enumerate(feature_names):
            if X[:, i].dtype == 'float':
                # Discretize continuous features
                bins = pd.qcut(X[:, i], q=n_bins, duplicates='drop')
                for bin_val in bins.unique():
                    col_name = f"{fname}_{bin_val}"
                    X_discrete[col_name] = (bins == bin_val).astype(int)
            else:
                # Keep categorical as is
                unique_vals = pd.Series(X[:, i]).unique()
                for val in unique_vals:
                    col_name = f"{fname}={val}"
                    X_discrete[col_name] = (X[:, i] == val).astype(int)
                    
        return X_discrete
    
    def mine_cash_break_rules(self, X, y, feature_names):
        """
        Find association rules for cash breaks
        """
        # Discretize features
        X_discrete = self.discretize_features(X, feature_names)
        
        # Add target
        X_discrete['cash_break'] = y
        
        # Find frequent itemsets
        frequent_itemsets = apriori(
            X_discrete, 
            min_support=self.min_support, 
            use_colnames=True
        )
        
        # Generate rules
        rules = association_rules(
            frequent_itemsets, 
            metric="confidence", 
            min_threshold=self.min_confidence
        )
        
        # Filter rules where consequent is cash_break
        cash_break_rules = rules[
            rules['consequents'].apply(lambda x: 'cash_break' in str(x))
        ]
        
        # Format rules
        formatted_rules = []
        for _, rule in cash_break_rules.iterrows():
            formatted_rules.append({
                'conditions': list(rule['antecedents']),
                'prediction': 'cash_break',
                'confidence': rule['confidence'],
                'support': rule['support'],
                'lift': rule['lift']  # How much more likely than random
            })
            
        return formatted_rules


"""
Strategy 4: Rule Extraction from Gradient Boosting Models
Extract rules directly from XGBoost/LightGBM trees.
"""

import xgboost as xgb
from collections import defaultdict

class GBMRuleExtractor:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def extract_path_rules(self, sample_X, threshold=0.7):
        """
        Extract the decision path for high-confidence predictions
        """
        # Get predictions
        predictions = self.model.predict_proba(sample_X)[:, 1]
        high_conf_indices = predictions > threshold
        
        # For XGBoost
        if isinstance(self.model, xgb.XGBClassifier):
            booster = self.model.get_booster()
            
            # Get decision paths
            paths = []
            for idx in np.where(high_conf_indices)[0]:
                sample = sample_X[idx:idx+1]
                
                # Get leaf indices for all trees
                leaf_indices = booster.predict(
                    xgb.DMatrix(sample), 
                    pred_leaf=True
                )
                
                # Extract path for each tree
                trees = booster.get_dump(with_stats=True)
                sample_paths = self._extract_paths_from_trees(
                    trees, leaf_indices[0], sample[0]
                )
                
                paths.append({
                    'prediction': predictions[idx],
                    'rules': sample_paths
                })
                
        return paths
    
    def _extract_paths_from_trees(self, trees, leaf_indices, sample):
        """
        Extract decision paths from tree dumps
        """
        all_conditions = defaultdict(list)
        
        for tree_idx, (tree_str, leaf_idx) in enumerate(zip(trees, leaf_indices)):
            # Parse tree string to find path to leaf
            conditions = self._parse_tree_path(tree_str, leaf_idx, sample)
            
            for feat, op, val in conditions:
                all_conditions[feat].append((op, val))
        
        # Consolidate conditions
        consolidated = []
        for feat, conditions in all_conditions.items():
            # Find range for numerical features
            less_than = [v for op, v in conditions if op == '<']
            greater_than = [v for op, v in conditions if op == '>=']
            
            if less_than and greater_than:
                consolidated.append(
                    f"{max(greater_than):.3f} <= {feat} < {min(less_than):.3f}"
                )
            elif less_than:
                consolidated.append(f"{feat} < {min(less_than):.3f}")
            elif greater_than:
                consolidated.append(f"{feat} >= {max(greater_than):.3f}")
                
        return consolidated
    
    def get_important_rules(self, X, y, top_k=10):
        """
        Extract the most important rules based on feature interactions
        """
        # Get feature interactions from model
        importance_scores = self.model.feature_importances_
        
        # For XGBoost, get interaction scores
        if hasattr(self.model, 'get_score'):
            scores = self.model.get_score(importance_type='gain')
            
            # Get top interactions
            top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            rules = []
            for feat, score in top_features:
                # Find typical values for this feature in positive cases
                feature_idx = self.feature_names.index(feat)
                positive_values = X[y == 1, feature_idx]
                
                if len(positive_values) > 0:
                    percentiles = np.percentile(positive_values, [25, 75])
                    
                    rule = {
                        'feature': feat,
                        'importance': score,
                        'typical_range': f"{percentiles[0]:.3f} - {percentiles[1]:.3f}",
                        'positive_mean': positive_values.mean(),
                        'negative_mean': X[y == 0, feature_idx].mean() if len(X[y == 0]) > 0 else 0
                    }
                    rules.append(rule)
                    
        return rules


"""
Strategy 5: Symbolic Regression for Mathematical Rules
Discover mathematical relationships that lead to cash breaks.
"""

from gplearn.genetic import SymbolicClassifier
import sympy as sp

class SymbolicRuleExtractor:
    def __init__(self, population_size=500, generations=20):
        self.model = SymbolicClassifier(
            population_size=population_size,
            generations=generations,
            tournament_size=20,
            stopping_criteria=0.95,  # Stop if 95% accuracy reached
            parsimony_coefficient=0.01,  # Prefer simpler formulas
            max_samples=0.9,
            verbose=1,
            random_state=42
        )
        
    def fit_and_extract(self, X, y, feature_names):
        """
        Discover symbolic rules for cash breaks
        """
        # Fit symbolic regressor
        self.model.fit(X, y)
        
        # Get the best program (formula)
        best_program = self.model._program
        
        # Convert to human-readable formula
        formula_str = str(best_program)
        
        # Parse and simplify using SymPy
        simplified = self._simplify_formula(formula_str, feature_names)
        
        return {
            'formula': simplified,
            'fitness': best_program.fitness_,
            'length': len(best_program.program),
            'raw': formula_str
        }
    
    def _simplify_formula(self, formula_str, feature_names):
        """
        Simplify the formula using symbolic math
        """
        # Replace X0, X1, etc. with actual feature names
        for i, name in enumerate(feature_names):
            formula_str = formula_str.replace(f'X{i}', name)
            
        try:
            # Parse and simplify
            expr = sp.sympify(formula_str)
            simplified = sp.simplify(expr)
            return str(simplified)
        except:
            return formula_str


"""
Strategy 6: Bayesian Rule Lists
Create probabilistic rule lists that are inherently interpretable.

"""

from sklearn.tree import DecisionTreeClassifier
import numpy as np

class BayesianRuleListExtractor:
    """
    Simplified Bayesian Rule List implementation
    Creates an ordered list of if-then-else rules
    """
    
    def __init__(self, max_rules=10, min_support=0.01):
        self.max_rules = max_rules
        self.min_support = min_support
        self.rule_list = []
        
    def fit(self, X, y, feature_names):
        """
        Build a rule list using greedy approach
        """
        remaining_X = X.copy()
        remaining_y = y.copy()
        remaining_indices = np.arange(len(y))
        
        for rule_idx in range(self.max_rules):
            if len(remaining_y) < len(y) * self.min_support:
                break
                
            # Find best rule for remaining data
            best_rule = self._find_best_rule(
                remaining_X, remaining_y, feature_names
            )
            
            if best_rule is None:
                break
                
            # Apply rule and get remaining data
            rule_applies = self._apply_rule(remaining_X, best_rule)
            
            # Store rule with statistics
            rule_stats = {
                'conditions': best_rule['conditions'],
                'prediction': best_rule['prediction'],
                'confidence': best_rule['confidence'],
                'support': rule_applies.sum(),
                'order': rule_idx
            }
            self.rule_list.append(rule_stats)
            
            # Remove covered instances
            remaining_X = remaining_X[~rule_applies]
            remaining_y = remaining_y[~rule_applies]
            remaining_indices = remaining_indices[~rule_applies]
            
        # Add default rule for remaining cases
        if len(remaining_y) > 0:
            default_pred = int(remaining_y.mean() > 0.5)
            self.rule_list.append({
                'conditions': 'ELSE',
                'prediction': default_pred,
                'confidence': remaining_y.mean() if default_pred == 1 else 1 - remaining_y.mean(),
                'support': len(remaining_y),
                'order': len(self.rule_list)
            })
            
        return self
    
    def _find_best_rule(self, X, y, feature_names):
        """
        Find the best rule for current data using information gain
        """
        if len(np.unique(y)) == 1:
            return None
            
        # Use decision tree to find best split
        dt = DecisionTreeClassifier(max_depth=1, min_samples_leaf=int(len(y) * 0.05))
        dt.fit(X, y)
        
        # Extract the rule
        tree = dt.tree_
        if tree.feature[0] == -2:  # No split found
            return None
            
        feature_idx = tree.feature[0]
        threshold = tree.threshold[0]
        feature_name = feature_names[feature_idx]
        
        # Determine which side has higher positive rate
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        left_pos_rate = y[left_mask].mean() if left_mask.any() else 0
        right_pos_rate = y[right_mask].mean() if right_mask.any() else 0
        
        if left_pos_rate > right_pos_rate:
            condition = (feature_name, '<=', threshold)
            mask = left_mask
            confidence = left_pos_rate
        else:
            condition = (feature_name, '>', threshold)
            mask = right_mask
            confidence = right_pos_rate
            
        return {
            'conditions': [condition],
            'prediction': 1 if confidence > 0.5 else 0,
            'confidence': confidence,
            'mask': mask
        }
    
    def _apply_rule(self, X, rule):
        """
        Check which instances satisfy the rule
        """
        return rule['mask']
    
    def predict_with_explanation(self, X, feature_names):
        """
        Predict using rule list and provide explanation
        """
        predictions = []
        explanations = []
        
        for instance in X:
            for rule in self.rule_list:
                if rule['conditions'] == 'ELSE':
                    predictions.append(rule['prediction'])
                    explanations.append(f"Default rule: predict {rule['prediction']}")
                    break
                    
                # Check if rule applies
                applies = True
                for feat, op, val in rule['conditions']:
                    feat_idx = feature_names.index(feat)
                    if op == '<=':
                        applies = applies and (instance[feat_idx] <= val)
                    else:
                        applies = applies and (instance[feat_idx] > val)
                        
                if applies:
                    predictions.append(rule['prediction'])
                    rule_str = " AND ".join([f"{f} {op} {v:.3f}" for f, op, v in rule['conditions']])
                    explanations.append(f"Rule {rule['order']}: IF {rule_str} THEN {rule['prediction']}")
                    break
                    
        return predictions, explanations


"""
Practical Implementation for Your Cash Break Use Case
Here's how to combine these approaches specifically for your problem:
"""

class CashBreakRuleSystem:
    def __init__(self, ml_model, X_train, y_train, feature_names, change_columns):
        self.ml_model = ml_model
        self.feature_names = feature_names
        self.change_columns = change_columns
        
        # Initialize multiple rule extractors
        self.tree_extractor = DecisionTreeRuleExtractor(feature_names)
        self.association_miner = AssociationRuleMiner()
        self.gbm_extractor = GBMRuleExtractor(ml_model, feature_names)
        
    def extract_comprehensive_rules(self, X, y):
        """
        Extract rules using multiple methods and combine
        """
        all_rules = {}
        
        # 1. Tree-based rules (most interpretable)
        print("Extracting tree-based rules...")
        tree_rules, mimic_tree = self.extract_tree_rules(X, y)
        all_rules['tree_rules'] = tree_rules
        
        # 2. Association rules (finds patterns)
        print("Mining association rules...")
        association_rules = self.association_miner.mine_cash_break_rules(X, y, self.feature_names)
        all_rules['association_rules'] = association_rules
        
        # 3. Change-specific rules
        print("Extracting change-specific rules...")
        change_rules = self.extract_change_rules(X, y)
        all_rules['change_rules'] = change_rules
        
        # 4. Combine and rank rules
        ranked_rules = self.rank_and_combine_rules(all_rules)
        
        return ranked_rules
    
    def extract_change_rules(self, X, y):
        """
        Focus on rules involving your change indicators
        """
        change_rules = []
        
        for change_col in self.change_columns:
            if change_col in self.feature_names:
                col_idx = self.feature_names.index(change_col)
                
                # Find when this change leads to cash break
                change_mask = X[:, col_idx] == 1
                break_rate_with_change = y[change_mask].mean() if change_mask.any() else 0
                break_rate_without = y[~change_mask].mean() if (~change_mask).any() else 0
                
                if break_rate_with_change > 0.7:  # High confidence rule
                    # Look for additional conditions
                    X_with_change = X[change_mask]
                    y_with_change = y[change_mask]
                    
                    # Find complementary conditions
                    dt = DecisionTreeClassifier(max_depth=2)
                    dt.fit(X_with_change, y_with_change)
                    
                    sub_rules = self.tree_extractor.extract_rules_from_tree(dt, min_samples=10)
                    
                    for sub_rule in sub_rules:
                        combined_rule = {
                            'primary_condition': f"{change_col} = 1",
                            'additional_conditions': sub_rule['conditions'],
                            'confidence': sub_rule['confidence'],
                            'lift': break_rate_with_change / (y.mean() + 1e-10),
                            'support': sub_rule['support']
                        }
                        change_rules.append(combined_rule)
                        
        return change_rules
    
    def rank_and_combine_rules(self, all_rules):
        """
        Combine rules from different methods and rank by importance
        """
        combined = []
        
        # Score each rule
        for method, rules in all_rules.items():
            for rule in rules:
                score = self.calculate_rule_score(rule)
                rule['method'] = method
                rule['score'] = score
                combined.append(rule)
        
        # Sort by score
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove redundant rules
        final_rules = self.remove_redundant_rules(combined)
        
        return final_rules
    
    def calculate_rule_score(self, rule):
        """
        Score based on confidence, support, and simplicity
        """
        confidence = rule.get('confidence', 0)
        support = rule.get('support', 0)
        lift = rule.get('lift', 1)
        
        # Normalize support (assume max 1000 samples)
        normalized_support = min(support / 1000, 1)
        
        # Penalize complex rules
        n_conditions = len(rule.get('conditions', [])) if rule.get('conditions') != 'ELSE' else 1
        simplicity = 1 / (1 + n_conditions)
        
        # Combined score
        score = (confidence * 0.4 + 
                normalized_support * 0.3 + 
                min(lift / 3, 1) * 0.2 + 
                simplicity * 0.1)
        
        return score
    
    def remove_redundant_rules(self, rules, similarity_threshold=0.8):
        """
        Remove rules that are too similar
        """
        final_rules = []
        
        for rule in rules:
            is_redundant = False
            
            for existing in final_rules:
                if self.calculate_rule_similarity(rule, existing) > similarity_threshold:
                    is_redundant = True
                    break
                    
            if not is_redundant:
                final_rules.append(rule)
                
        return final_rules
    
    def format_rules_for_ops_team(self, rules, top_k=10):
        """
        Format top rules for the operations team
        """
        report = []
        
        for i, rule in enumerate(rules[:top_k], 1):
            formatted = f"\n{'='*50}\n"
            formatted += f"RULE #{i} (Confidence: {rule.get('confidence', 0):.1%})\n"
            formatted += f"{'='*50}\n"
            
            # Format conditions
            if 'primary_condition' in rule:
                formatted += f"WHEN: {rule['primary_condition']}\n"
                if rule.get('additional_conditions'):
                    formatted += "AND: " + " AND ".join([str(c) for c in rule['additional_conditions']]) + "\n"
            elif 'conditions' in rule and rule['conditions'] != 'ELSE':
                formatted += "WHEN: " + " AND ".join([str(c) for c in rule['conditions']]) + "\n"
            else:
                formatted += "DEFAULT RULE\n"
            
            formatted += f"THEN: Cash Break Likely\n"
            formatted += f"Support: {rule.get('support', 0)} historical cases\n"
            
            if 'lift' in rule:
                formatted += f"Lift: {rule['lift']:.2f}x more likely than baseline\n"
            
            formatted += f"Detection Method: {rule.get('method', 'unknown')}\n"
            
            report.append(formatted)
            
        return "\n".join(report)


"""
Key Takeaways for Your Implementation

- Start Simple: Begin with decision tree rule extraction - it's fast and interpretable
- Combine Methods: Different extraction methods find different types of patterns
- Focus on Change Rules: Since your change indicators are key, prioritize rules involving them
- Validate with Domain Experts: Show extracted rules to the Cash team for validation
- Track Rule Performance: Monitor which rules actually predict breaks in production
- Create Rule Templates: Common patterns like "When X changes near month-end"
- Build a Rule Library: Accumulate validated rules over time

The beauty of rule extraction is that it bridges the gap between ML accuracy and business understanding, making your predictions actionable for the operations team.
"""
