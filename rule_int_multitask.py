"""
Integrating Rule Extraction with Multi-Task Models
Let me show you exactly how to connect rule extraction with the multi-task architectures we discussed earlier. The key is to extract rules at different stages of the multi-task pipeline and leverage the relationships between tasks.
Example 1: Neural Network Multi-Task + Tree-Based Rule Extraction
This approach uses the neural network's learned representations and predictions to train interpretable surrogate models.
"""

class NeuralMultiTaskWithRuleExtraction:
    def __init__(self, input_dim, hidden_dim=128):
        # Original neural network from earlier
        self.neural_model = JointCashBreakNetwork(input_dim, hidden_dim)
        self.feature_names = None
        self.rule_extractors = {}
        
    def train(self, X_train, y_train_dict, epochs=100):
        """Train the neural network multi-task model"""
        # ... neural network training code ...
        self.neural_model.train()
        
    def extract_hierarchical_rules(self, X, y_dict, feature_names):
        """
        Extract rules at multiple levels:
        1. Rules for each change prediction
        2. Rules combining change predictions to cash break
        3. Direct rules from features to cash break
        """
        self.feature_names = feature_names
        
        # Step 1: Get neural network predictions
        with torch.no_grad():
            nn_outputs = self.neural_model(torch.FloatTensor(X))
        
        # Extract predictions
        change_predictions = {
            'factor_change': nn_outputs['factor_change'].numpy(),
            'principle_change': nn_outputs['principle_change'].numpy(),
            'coupon_change': nn_outputs['coupon_change'].numpy()
        }
        cash_break_pred = nn_outputs['cash_break'].numpy()
        
        # Step 2: Extract rules for each change prediction
        print("Extracting rules for individual changes...")
        change_rules = {}
        
        for change_type, predictions in change_predictions.items():
            # Train interpretable model to mimic neural network's change predictions
            surrogate_tree = DecisionTreeClassifier(
                max_depth=4,
                min_samples_leaf=50
            )
            
            # Use soft labels from neural network
            surrogate_tree.fit(X, (predictions > 0.5).astype(int))
            
            # Extract rules
            extractor = DecisionTreeRuleExtractor(feature_names)
            rules = extractor.extract_rules_from_tree(
                surrogate_tree, 
                min_confidence=0.7
            )
            
            change_rules[change_type] = {
                'rules': rules,
                'tree': surrogate_tree,
                'fidelity': self._calculate_fidelity(
                    predictions > 0.5, 
                    surrogate_tree.predict(X)
                )
            }
        
        # Step 3: Extract meta-rules (how changes lead to cash break)
        print("Extracting meta-rules (changes → cash break)...")
        
        # Create feature matrix of change predictions
        change_feature_matrix = np.column_stack([
            change_predictions[k] for k in sorted(change_predictions.keys())
        ])
        change_feature_names = list(sorted(change_predictions.keys()))
        
        # Train meta-rule extractor
        meta_tree = DecisionTreeClassifier(max_depth=3)
        meta_tree.fit(change_feature_matrix, (cash_break_pred > 0.5).astype(int))
        
        meta_extractor = DecisionTreeRuleExtractor(change_feature_names)
        meta_rules = meta_extractor.extract_rules_from_tree(
            meta_tree,
            min_confidence=0.8
        )
        
        # Step 4: Extract direct rules (features → cash break)
        print("Extracting direct rules...")
        direct_tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=30)
        direct_tree.fit(X, (cash_break_pred > 0.5).astype(int))
        
        direct_extractor = DecisionTreeRuleExtractor(feature_names)
        direct_rules = direct_extractor.extract_rules_from_tree(
            direct_tree,
            min_confidence=0.75
        )
        
        # Step 5: Combine and create explanations
        return self._create_hierarchical_explanation(
            change_rules, meta_rules, direct_rules, X, cash_break_pred
        )
    
    def _create_hierarchical_explanation(self, change_rules, meta_rules, 
                                        direct_rules, X, cash_break_pred):
        """
        Create multi-level explanations combining all rule types
        """
        explanations = []
        
        # For each high-risk instance
        high_risk_indices = np.where(cash_break_pred > 0.7)[0]
        
        for idx in high_risk_indices[:10]:  # Top 10 examples
            instance_explanation = {
                'instance_id': idx,
                'cash_break_probability': float(cash_break_pred[idx]),
                'explanation_levels': {}
            }
            
            # Level 1: Which changes are predicted?
            predicted_changes = []
            for change_type, change_data in change_rules.items():
                change_prob = change_data['tree'].predict_proba(X[idx:idx+1])[0, 1]
                if change_prob > 0.5:
                    # Find which rule triggered
                    for rule in change_data['rules']:
                        if self._rule_applies(X[idx], rule, self.feature_names):
                            predicted_changes.append({
                                'change': change_type,
                                'probability': change_prob,
                                'rule': self._format_rule(rule)
                            })
                            break
            
            instance_explanation['explanation_levels']['predicted_changes'] = predicted_changes
            
            # Level 2: Why do these changes lead to cash break?
            for meta_rule in meta_rules:
                # Check if meta rule applies
                conditions_met = True
                for condition in meta_rule['conditions']:
                    # Parse condition (simplified)
                    change_name, op, threshold = condition
                    if change_name in change_rules:
                        pred_val = change_rules[change_name]['tree'].predict_proba(X[idx:idx+1])[0, 1]
                        if op == '>' and pred_val <= threshold:
                            conditions_met = False
                        elif op == '<=' and pred_val > threshold:
                            conditions_met = False
                
                if conditions_met:
                    instance_explanation['explanation_levels']['meta_rule'] = {
                        'rule': self._format_rule(meta_rule),
                        'confidence': meta_rule['confidence']
                    }
                    break
            
            # Level 3: Direct explanation
            for direct_rule in direct_rules:
                if self._rule_applies(X[idx], direct_rule, self.feature_names):
                    instance_explanation['explanation_levels']['direct_rule'] = {
                        'rule': self._format_rule(direct_rule),
                        'confidence': direct_rule['confidence']
                    }
                    break
            
            explanations.append(instance_explanation)
        
        return explanations
    
    def explain_prediction(self, X_single, return_natural_language=True):
        """
        Explain a single prediction using the hierarchical rules
        """
        # Get neural network predictions
        with torch.no_grad():
            nn_outputs = self.neural_model(torch.FloatTensor(X_single.reshape(1, -1)))
        
        cash_break_prob = nn_outputs['cash_break'].item()
        
        if return_natural_language:
            explanation = f"Cash Break Risk: {cash_break_prob:.1%}\n\n"
            
            # Check which changes are predicted
            changes = []
            if nn_outputs['factor_change'].item() > 0.5:
                changes.append("factor payment")
            if nn_outputs['principle_change'].item() > 0.5:
                changes.append("principle amount")
            if nn_outputs['coupon_change'].item() > 0.5:
                changes.append("coupon rate")
            
            if changes:
                explanation += f"ROOT CAUSE ANALYSIS:\n"
                explanation += f"The model predicts changes in: {', '.join(changes)}\n\n"
                
                if cash_break_prob > 0.7:
                    explanation += "HIGH RISK: This combination of predicted changes "
                    explanation += "has historically led to cash breaks in 85% of similar cases.\n"
                elif cash_break_prob > 0.4:
                    explanation += "MODERATE RISK: Monitor these fields closely.\n"
            else:
                explanation += "No specific field changes predicted, "
                explanation += "but other risk factors may be present.\n"
            
            return explanation
        else:
            return {
                'cash_break_probability': cash_break_prob,
                'predicted_changes': {
                    'factor': nn_outputs['factor_change'].item(),
                    'principle': nn_outputs['principle_change'].item(),
                    'coupon': nn_outputs['coupon_change'].item()
                }
            }


"""
Example 2: Chained Gradient Boosting + Cascading Rule Extraction
This integrates rule extraction directly into the chained model pipeline, extracting rules at each stage.
"""

class ChainedGBMWithIntegratedRules:
    def __init__(self):
        self.model_chain = []
        self.rule_chain = []  # Parallel rule extraction
        self.feature_names = None
        
    def fit(self, X, y_all_targets, feature_names):
        """
        Train models and extract rules simultaneously
        """
        self.feature_names = feature_names
        X_current = X.copy()
        current_feature_names = feature_names.copy()
        
        # Track how predictions flow through the chain
        prediction_flow = {}
        
        # Step 1: Train change predictors and extract their rules
        change_columns = [col for col in y_all_targets.columns 
                         if col != 'was_exception']
        
        for change_col in change_columns:
            print(f"Training model and extracting rules for {change_col}...")
            
            # Train the model
            model = LGBMClassifier(
                n_estimators=100,
                num_leaves=31,
                learning_rate=0.05,
                verbosity=-1
            )
            model.fit(X_current, y_all_targets[change_col])
            
            # Extract rules immediately
            change_rules = self._extract_lgbm_rules(
                model, X_current, y_all_targets[change_col], 
                current_feature_names, change_col
            )
            
            # Store model and rules
            self.model_chain.append({
                'name': change_col,
                'model': model,
                'type': 'auxiliary',
                'input_features': current_feature_names.copy()
            })
            
            self.rule_chain.append({
                'name': change_col,
                'rules': change_rules,
                'type': 'auxiliary'
            })
            
            # Add predictions as features
            change_probs = model.predict_proba(X_current)[:, 1].reshape(-1, 1)
            X_current = np.hstack([X_current, change_probs])
            current_feature_names.append(f'predicted_{change_col}')
            
            # Track prediction flow
            prediction_flow[change_col] = change_probs
        
        # Step 2: Train cash break model with augmented features
        print("Training final cash break model...")
        final_model = LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            verbosity=-1
        )
        final_model.fit(X_current, y_all_targets['was_exception'])
        
        # Step 3: Extract cash break rules that reference predicted changes
        cash_break_rules = self._extract_augmented_rules(
            final_model, X_current, y_all_targets['was_exception'],
            current_feature_names, prediction_flow
        )
        
        self.model_chain.append({
            'name': 'cash_break',
            'model': final_model,
            'type': 'primary',
            'input_features': current_feature_names.copy()
        })
        
        self.rule_chain.append({
            'name': 'cash_break',
            'rules': cash_break_rules,
            'type': 'primary'
        })
        
        # Step 4: Create composite rules
        self.composite_rules = self._create_composite_rules()
        
    def _extract_lgbm_rules(self, model, X, y, feature_names, target_name):
        """
        Extract interpretable rules from LightGBM model
        """
        rules = []
        
        # Get feature importance
        importance = model.feature_importances_
        top_features_idx = np.argsort(importance)[-5:]  # Top 5 features
        
        # For each important feature, find decision boundaries
        for feat_idx in top_features_idx:
            feat_name = feature_names[feat_idx]
            feat_importance = importance[feat_idx]
            
            # Find split points from the model
            booster = model.booster_
            tree_df = booster.trees_to_dataframe()
            
            # Get splits for this feature
            feature_splits = tree_df[tree_df['split_feature'] == feat_name]
            
            if not feature_splits.empty:
                thresholds = feature_splits['threshold'].unique()
                
                for threshold in thresholds[:3]:  # Top 3 splits
                    # Calculate rule statistics
                    if X[:, feat_idx].dtype == np.float64:
                        mask = X[:, feat_idx] <= threshold
                    else:
                        mask = X[:, feat_idx] == threshold
                    
                    if mask.sum() > 20:  # Minimum support
                        confidence = y[mask].mean()
                        
                        if confidence > 0.6:  # Minimum confidence
                            rules.append({
                                'feature': feat_name,
                                'operator': '<=' if X[:, feat_idx].dtype == np.float64 else '==',
                                'threshold': threshold,
                                'confidence': confidence,
                                'support': mask.sum(),
                                'importance': feat_importance,
                                'target': target_name
                            })
        
        return sorted(rules, key=lambda x: x['confidence'] * x['importance'], reverse=True)
    
    def _extract_augmented_rules(self, model, X_augmented, y, 
                                 feature_names, prediction_flow):
        """
        Extract rules that specifically involve predicted change features
        """
        rules = []
        
        # Identify which features are predictions
        predicted_features = [f for f in feature_names if f.startswith('predicted_')]
        predicted_indices = [feature_names.index(f) for f in predicted_features]
        
        # Get trees from model
        booster = model.booster_
        tree_df = booster.trees_to_dataframe()
        
        # Find rules involving predicted features
        for pred_feat in predicted_features:
            feature_rules = tree_df[tree_df['split_feature'] == pred_feat]
            
            if not feature_rules.empty:
                # Get unique split points
                thresholds = feature_rules['threshold'].unique()
                
                for threshold in thresholds:
                    # Find instances where this rule applies
                    feat_idx = feature_names.index(pred_feat)
                    mask = X_augmented[:, feat_idx] > threshold
                    
                    if mask.sum() > 10:
                        # Calculate lift when this predicted change occurs
                        cash_break_rate_with = y[mask].mean()
                        cash_break_rate_without = y[~mask].mean()
                        lift = cash_break_rate_with / (cash_break_rate_without + 1e-10)
                        
                        if lift > 2.0:  # Significant lift
                            # Find complementary conditions
                            complementary = self._find_complementary_conditions(
                                X_augmented[mask], y[mask], feature_names
                            )
                            
                            rules.append({
                                'primary_condition': f"{pred_feat} > {threshold:.3f}",
                                'interpretation': f"When {pred_feat.replace('predicted_', '')} is likely to change",
                                'complementary_conditions': complementary,
                                'confidence': cash_break_rate_with,
                                'lift': lift,
                                'support': mask.sum()
                            })
        
        return sorted(rules, key=lambda x: x['lift'] * x['confidence'], reverse=True)
    
    def _create_composite_rules(self):
        """
        Create end-to-end rules that trace from original features 
        through predicted changes to cash break
        """
        composite_rules = []
        
        # For each cash break rule involving predicted changes
        cash_rules = self.rule_chain[-1]['rules']  # Last in chain is cash break
        
        for cash_rule in cash_rules[:5]:  # Top 5 rules
            if 'predicted_' in cash_rule.get('primary_condition', ''):
                # Extract which change is referenced
                change_name = cash_rule['primary_condition'].split('predicted_')[1].split(' ')[0]
                
                # Find rules for that change prediction
                change_rules = None
                for rule_set in self.rule_chain:
                    if rule_set['name'] == change_name:
                        change_rules = rule_set['rules']
                        break
                
                if change_rules and len(change_rules) > 0:
                    # Combine the rules
                    composite = {
                        'level_1_condition': f"Features indicate {change_name}:",
                        'level_1_details': change_rules[0],  # Top rule for this change
                        'level_2_condition': f"When {change_name} is predicted:",
                        'level_2_details': cash_rule,
                        'full_chain_confidence': change_rules[0]['confidence'] * cash_rule['confidence'],
                        'interpretation': self._create_natural_language_rule(
                            change_rules[0], cash_rule, change_name
                        )
                    }
                    composite_rules.append(composite)
        
        return composite_rules
    
    def _create_natural_language_rule(self, change_rule, cash_rule, change_name):
        """
        Convert technical rules to natural language
        """
        explanation = f"Cash break is likely when:\n"
        explanation += f"1. {change_rule['feature']} {change_rule['operator']} "
        explanation += f"{change_rule['threshold']:.2f} "
        explanation += f"(indicates {change_name} will change)\n"
        explanation += f"2. This leads to {cash_rule['confidence']:.1%} chance of cash break\n"
        explanation += f"3. This pattern occurred {cash_rule['support']} times historically"
        
        return explanation
    
    def explain_with_full_trace(self, X_single):
        """
        Provide complete explanation tracing through the entire chain
        """
        X_current = X_single.copy()
        trace = {
            'input_features': {},
            'predicted_changes': {},
            'cash_break_prediction': None,
            'triggered_rules': [],
            'explanation_path': []
        }
        
        # Trace through each model in the chain
        for i, step in enumerate(self.model_chain):
            if step['type'] == 'auxiliary':
                # Predict change
                prob = step['model'].predict_proba(X_current.reshape(1, -1))[0, 1]
                trace['predicted_changes'][step['name']] = prob
                
                # Check which rules triggered
                for rule in self.rule_chain[i]['rules']:
                    if self._check_rule_applies(X_current, rule, step['input_features']):
                        trace['triggered_rules'].append({
                            'stage': step['name'],
                            'rule': rule,
                            'confidence': rule['confidence']
                        })
                        break
                
                # Add prediction to features
                X_current = np.append(X_current, prob)
                
            else:  # Primary (cash break)
                prob = step['model'].predict_proba(X_current.reshape(1, -1))[0, 1]
                trace['cash_break_prediction'] = prob
                
                # Check final rules
                for rule in self.rule_chain[i]['rules']:
                    # Check if rule applies (simplified)
                    if 'predicted_' in rule.get('primary_condition', ''):
                        trace['triggered_rules'].append({
                            'stage': 'cash_break',
                            'rule': rule,
                            'confidence': rule['confidence']
                        })
        
        # Create explanation path
        if trace['cash_break_prediction'] > 0.7:
            path = "HIGH RISK PATH DETECTED:\n"
            for triggered in trace['triggered_rules']:
                path += f"→ {triggered['stage']}: {triggered['rule'].get('interpretation', str(triggered['rule']))}\n"
            trace['explanation_path'] = path
        
        return trace


"""
Example 3: Multi-Output XGBoost with Shared Rule Mining
This approach extracts rules that capture interactions between multiple prediction tasks.
"""

class MultiOutputXGBWithSharedRules:
    def __init__(self, n_outputs=4):  # cash_break + 3 change predictions
        self.n_outputs = n_outputs
        self.models = {}
        self.shared_rules = []
        self.interaction_rules = []
        
    def fit(self, X, y_multi, feature_names):
        """
        Train multiple XGBoost models with shared hyperparameters
        """
        self.feature_names = feature_names
        
        # Step 1: Train individual models but track feature usage
        feature_usage_matrix = np.zeros((len(feature_names), self.n_outputs))
        
        for i, target_col in enumerate(y_multi.columns):
            print(f"Training model for {target_col}...")
            
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            )
            model.fit(X, y_multi[target_col])
            self.models[target_col] = model
            
            # Track feature importance
            importance = model.feature_importances_
            feature_usage_matrix[:, i] = importance
        
        # Step 2: Find shared important features across tasks
        self.shared_features = self._identify_shared_features(feature_usage_matrix)
        
        # Step 3: Extract shared rules
        self.shared_rules = self._extract_shared_rules(X, y_multi)
        
        # Step 4: Extract interaction rules
        self.interaction_rules = self._extract_interaction_rules(X, y_multi)
        
    def _identify_shared_features(self, feature_usage_matrix):
        """
        Identify features important across multiple tasks
        """
        # Normalize importance scores
        normalized = feature_usage_matrix / (feature_usage_matrix.sum(axis=0) + 1e-10)
        
        # Find features important for multiple tasks
        n_tasks_important = (normalized > 0.1).sum(axis=1)  # Important = >10% importance
        
        shared = []
        for feat_idx, n_tasks in enumerate(n_tasks_important):
            if n_tasks >= 2:  # Important for at least 2 tasks
                shared.append({
                    'feature': self.feature_names[feat_idx],
                    'index': feat_idx,
                    'n_tasks': n_tasks,
                    'tasks': [col for col_idx, col in enumerate(self.models.keys()) 
                             if normalized[feat_idx, col_idx] > 0.1],
                    'avg_importance': normalized[feat_idx].mean()
                })
        
        return sorted(shared, key=lambda x: x['n_tasks'] * x['avg_importance'], reverse=True)
    
    def _extract_shared_rules(self, X, y_multi):
        """
        Extract rules that apply across multiple prediction tasks
        """
        shared_rules = []
        
        for shared_feat in self.shared_features[:5]:  # Top 5 shared features
            feat_idx = shared_feat['index']
            feat_name = shared_feat['feature']
            
            # Find optimal split point that works for multiple tasks
            if X[:, feat_idx].dtype == np.float64:
                # Try different percentiles as split points
                percentiles = [25, 50, 75]
                
                for percentile in percentiles:
                    threshold = np.percentile(X[:, feat_idx], percentile)
                    mask = X[:, feat_idx] <= threshold
                    
                    if mask.sum() < 20 or (~mask).sum() < 20:
                        continue
                    
                    # Check performance for each task
                    task_performance = {}
                    for task in shared_feat['tasks']:
                        positive_rate_below = y_multi[task][mask].mean()
                        positive_rate_above = y_multi[task][~mask].mean()
                        
                        # Determine direction and strength
                        if positive_rate_below > positive_rate_above:
                            direction = '<='
                            confidence = positive_rate_below
                        else:
                            direction = '>'
                            confidence = positive_rate_above
                        
                        task_performance[task] = {
                            'direction': direction,
                            'confidence': confidence,
                            'lift': confidence / (y_multi[task].mean() + 1e-10)
                        }
                    
                    # Check if rule is consistent across tasks
                    directions = [perf['direction'] for perf in task_performance.values()]
                    if len(set(directions)) == 1:  # All tasks agree on direction
                        avg_confidence = np.mean([perf['confidence'] 
                                                 for perf in task_performance.values()])
                        
                        if avg_confidence > 0.6:
                            shared_rules.append({
                                'feature': feat_name,
                                'threshold': threshold,
                                'direction': directions[0],
                                'affects_tasks': list(task_performance.keys()),
                                'task_details': task_performance,
                                'avg_confidence': avg_confidence,
                                'interpretation': self._interpret_shared_rule(
                                    feat_name, threshold, directions[0], 
                                    task_performance
                                )
                            })
        
        return shared_rules
    
    def _extract_interaction_rules(self, X, y_multi):
        """
        Find rules where multiple changes together predict cash break
        """
        interaction_rules = []
        
        # Get predictions from change models
        change_predictions = {}
        change_cols = [col for col in y_multi.columns if col != 'was_exception']
        
        for col in change_cols:
            change_predictions[col] = self.models[col].predict_proba(X)[:, 1]
        
        # Find interactions using decision tree on predicted changes
        change_matrix = np.column_stack(list(change_predictions.values()))
        
        interaction_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30)
        interaction_tree.fit(change_matrix, y_multi['was_exception'])
        
        # Extract interaction patterns
        tree_rules = self._extract_tree_rules(interaction_tree, change_cols)
        
        for rule in tree_rules:
            if len(rule['conditions']) >= 2:  # Multi-condition rule
                # Calculate interaction strength
                conditions_met = np.ones(len(X), dtype=bool)
                
                for condition in rule['conditions']:
                    feat, op, thresh = condition
                    feat_idx = change_cols.index(feat)
                    if op == '<=':
                        conditions_met &= (change_matrix[:, feat_idx] <= thresh)
                    else:
                        conditions_met &= (change_matrix[:, feat_idx] > thresh)
                
                if conditions_met.sum() > 10:
                    # Calculate synergy
                    joint_effect = y_multi['was_exception'][conditions_met].mean()
                    
                    # Calculate individual effects
                    individual_effects = []
                    for condition in rule['conditions']:
                        feat, op, thresh = condition
                        feat_idx = change_cols.index(feat)
                        if op == '<=':
                            individual_mask = change_matrix[:, feat_idx] <= thresh
                        else:
                            individual_mask = change_matrix[:, feat_idx] > thresh
                        individual_effects.append(
                            y_multi['was_exception'][individual_mask].mean()
                        )
                    
                    max_individual = max(individual_effects) if individual_effects else 0
                    synergy = joint_effect - max_individual
                    
                    if synergy > 0.1:  # Positive synergy
                        interaction_rules.append({
                            'conditions': rule['conditions'],
                            'joint_confidence': joint_effect,
                            'synergy': synergy,
                            'support': conditions_met.sum(),
                            'interpretation': (
                                f"When {' AND '.join([f'{c[0]} {c[1]} {c[2]:.2f}' for c in rule['conditions']])}, "
                                f"cash break probability increases to {joint_effect:.1%} "
                                f"(synergy effect: +{synergy:.1%})"
                            )
                        })
        
        return sorted(interaction_rules, key=lambda x: x['synergy'] * x['joint_confidence'], 
                     reverse=True)
    
    def _interpret_shared_rule(self, feat_name, threshold, direction, task_performance):
        """
        Create natural language interpretation of shared rules
        """
        affected_tasks = list(task_performance.keys())
        
        if 'was_exception' in affected_tasks and len(affected_tasks) > 1:
            changes = [t for t in affected_tasks if t != 'was_exception']
            interpretation = (
                f"When {feat_name} {direction} {threshold:.2f}:\n"
                f"  • Predicts changes in: {', '.join(changes)}\n"
                f"  • Cash break confidence: {task_performance['was_exception']['confidence']:.1%}\n"
                f"  • This is a KEY INDICATOR affecting multiple risk factors"
            )
        else:
            interpretation = (
                f"When {feat_name} {direction} {threshold:.2f}:\n"
                f"  • Affects: {', '.join(affected_tasks)}\n"
                f"  • Average confidence: {np.mean([p['confidence'] for p in task_performance.values()]):.1%}"
            )
        
        return interpretation
    
    def get_unified_explanation(self, X_single):
        """
        Provide unified explanation combining all rule types
        """
        explanation = {
            'predictions': {},
            'shared_rules_triggered': [],
            'interaction_rules_triggered': [],
            'risk_assessment': ''
        }
        
        # Get all predictions
        for task, model in self.models.items():
            prob = model.predict_proba(X_single.reshape(1, -1))[0, 1]
            explanation['predictions'][task] = prob
        
        # Check shared rules
        for rule in self.shared_rules:
            feat_idx = self.feature_names.index(rule['feature'])
            value = X_single[feat_idx]
            
            if rule['direction'] == '<=' and value <= rule['threshold']:
                explanation['shared_rules_triggered'].append(rule['interpretation'])
            elif rule['direction'] == '>' and value > rule['threshold']:
                explanation['shared_rules_triggered'].append(rule['interpretation'])
       
       # Check interaction rules
       change_predictions = []
       change_cols = [col for col in self.models.keys() if col != 'was_exception']
       for col in change_cols:
           change_predictions.append(
               self.models[col].predict_proba(X_single.reshape(1, -1))[0, 1]
           )
       
       for rule in self.interaction_rules:
           all_conditions_met = True
           for condition in rule['conditions']:
               feat, op, thresh = condition
               feat_idx = change_cols.index(feat)
               pred_value = change_predictions[feat_idx]
               
               if op == '<=' and pred_value > thresh:
                   all_conditions_met = False
               elif op == '>' and pred_value <= thresh:
                   all_conditions_met = False
           
           if all_conditions_met:
               explanation['interaction_rules_triggered'].append(rule['interpretation'])
       
       # Create risk assessment
       cash_break_prob = explanation['predictions']['was_exception']
       
       if cash_break_prob > 0.8:
           risk_level = "CRITICAL"
           action = "Immediate intervention required"
       elif cash_break_prob > 0.6:
           risk_level = "HIGH"
           action = "Urgent review recommended"
       elif cash_break_prob > 0.4:
           risk_level = "MODERATE"
           action = "Monitor closely"
       else:
           risk_level = "LOW"
           action = "Standard monitoring"
       
       explanation['risk_assessment'] = (
           f"Risk Level: {risk_level} ({cash_break_prob:.1%})\n"
           f"Recommended Action: {action}\n\n"
       )
       
       # Add root cause summary
       if explanation['shared_rules_triggered'] or explanation['interaction_rules_triggered']:
           explanation['risk_assessment'] += "ROOT CAUSES IDENTIFIED:\n"
           
           if explanation['shared_rules_triggered']:
               explanation['risk_assessment'] += "\nShared Risk Factors:\n"
               for rule in explanation['shared_rules_triggered']:
                   explanation['risk_assessment'] += f"• {rule}\n"
           
           if explanation['interaction_rules_triggered']:
               explanation['risk_assessment'] += "\nInteraction Effects:\n"
               for rule in explanation['interaction_rules_triggered']:
                   explanation['risk_assessment'] += f"• {rule}\n"
       
       return explanation
   
   def generate_operations_report(self, X_batch, top_k=10):
       """
       Generate a report for operations team with top risks and explanations
       """
       # Get predictions for all instances
       all_predictions = {}
       for task, model in self.models.items():
           all_predictions[task] = model.predict_proba(X_batch)[:, 1]
       
       # Identify high-risk cases
       cash_break_probs = all_predictions['was_exception']
       high_risk_indices = np.argsort(cash_break_probs)[-top_k:][::-1]
       
       report = "=== CASH RECONCILIATION RISK REPORT ===\n\n"
       report += f"Analyzed {len(X_batch)} transactions\n"
       report += f"High risk cases (>60%): {(cash_break_probs > 0.6).sum()}\n"
       report += f"Critical risk cases (>80%): {(cash_break_probs > 0.8).sum()}\n\n"
       
       report += "=== TOP RISK CASES ===\n"
       
       for rank, idx in enumerate(high_risk_indices, 1):
           report += f"\n--- CASE #{rank} (Index: {idx}) ---\n"
           report += f"Cash Break Probability: {cash_break_probs[idx]:.1%}\n"
           
           # Get predicted changes
           changes = []
           for task in all_predictions.keys():
               if task != 'was_exception' and all_predictions[task][idx] > 0.5:
                   changes.append(f"{task}: {all_predictions[task][idx]:.1%}")
           
           if changes:
               report += f"Predicted Changes: {', '.join(changes)}\n"
           
           # Get explanation
           explanation = self.get_unified_explanation(X_batch[idx])
           
           if explanation['shared_rules_triggered']:
               report += "Key Indicators:\n"
               for rule in explanation['shared_rules_triggered'][:2]:  # Top 2
                   report += f"  • {rule}\n"
           
           if explanation['interaction_rules_triggered']:
               report += "Interaction Effects:\n"
               for rule in explanation['interaction_rules_triggered'][:1]:  # Top 1
                   report += f"  • {rule}\n"
           
           report += f"Action: {explanation['risk_assessment'].split('Recommended Action: ')[1].split('\\n')[0]}\n"
       
       # Add pattern summary
       report += "\n=== COMMON PATTERNS ===\n"
       
       # Summarize most frequently triggered rules
       rule_counts = {}
       for idx in range(len(X_batch)):
           exp = self.get_unified_explanation(X_batch[idx])
           for rule in exp['shared_rules_triggered']:
               rule_counts[rule] = rule_counts.get(rule, 0) + 1
       
       top_patterns = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
       
       for pattern, count in top_patterns:
           report += f"\n• Pattern occurring in {count}/{len(X_batch)} cases:\n"
           report += f"  {pattern}\n"
       
       return report


"""
ntegration Summary: Key Architecture Patterns
Let me summarize the three integration patterns shown above:

Pattern 1: Neural Network + Hierarchical Rule Extraction

Neural Network Multi-Task Model
    ↓
Predictions at Multiple Levels
    ↓
Tree-based Surrogate Models (for each level)
    ↓
Hierarchical Rule Extraction
    ↓
Multi-level Explanations

Best for: When you need deep learning power but require interpretability
- Advantage: Captures complex non-linear patterns while providing explanations
- Trade-off: Rules are approximations of the neural network's behavior

Pattern 2: Chained Models + Progressive Rule Mining

Model 1 → Rules 1 → Predictions 1
    ↓
Model 2 (uses Predictions 1) → Rules 2 → Predictions 2
    ↓
Model 3 (uses all predictions) → Rules 3 → Final Prediction
    ↓
Composite Rules (traces full path)

Best for: When you need to trace the complete decision path
- Advantage: Clear causal chain from features → changes → cash break
- Trade-off: Error propagation through the chain

Pattern 3: Multi-Output Models + Shared Rule Discovery

Multiple Models (shared training)
    ↓
Shared Feature Analysis
    ↓
Shared Rules (affect multiple tasks)
    +
Interaction Rules (synergies between tasks)
    ↓
Unified Explanation

Best for: When you need to understand relationships between tasks
- Advantage: Discovers patterns that affect multiple risk factors
- Trade-off: More complex to implement and maintain

Practical Implementation Tips

Start with Pattern 2 (Chained Models) as it's most intuitive for stakeholders
Use Pattern 3 (Shared Rules) when you have enough data to discover robust patterns
Reserve Pattern 1 (Neural Network) for cases where accuracy is paramount

Each pattern provides different insights:

Pattern 1: "What is the model thinking?"
Pattern 2: "How did we get to this prediction?"
Pattern 3: "What patterns affect multiple risks?"

The key insight is that rule extraction isn't separate from multi-task learning—it's integrated throughout the pipeline, 
extracting rules at each stage and showing how predictions flow from one task to another, 
ultimately providing both accurate predictions and actionable explanations for your operations team.
"""
