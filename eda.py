"""
1. Pairwise Interaction Analysis

- Chi-square test for combinations: For your 20 binary change indicators, test all pairwise combinations against the break target:
- Interaction strength metrics: Calculate lift or odds ratios for combinations:
"""

from scipy.stats import chi2_contingency
import itertools

# Create contingency tables for each pair
for col1, col2 in itertools.combinations(binary_change_cols, 2):
    # Create 2x2x2 contingency table: (col1, col2, target)
    contingency = pd.crosstab([df[col1], df[col2]], df['target'])
    chi2, p_value = chi2_contingency(contingency)[:2]

# Example: Lift calculation
def calculate_lift(df, col1, col2, target):
    # P(break | both changes) / P(break | individual changes)
    both_change = df[(df[col1]==1) & (df[col2]==1)][target].mean()
    col1_only = df[df[col1]==1][target].mean()
    col2_only = df[df[col2]==1][target].mean()
    
    expected = col1_only + col2_only - (col1_only * col2_only)
    lift = both_change / expected if expected > 0 else 0
    return lift


"""
Method 4: Association Rule Mining
Association rule mining finds patterns like "if changes A and B occur together, then a break happens with X% confidence." 
This is particularly powerful for understanding which combinations of field changes reliably predict breaks.

Key Metrics Explained:

- Support: How frequently the combination appears in your data (e.g., 0.05 = 5% of cases)
- Confidence: When the antecedent occurs, what's the probability of the consequent? (e.g., 0.8 = 80% of the time when fields A+B change, a break occurs)
- Lift: How much more likely is the consequent given the antecedent compared to random? (lift > 1 means positive association)
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class BreakAssociationAnalyzer:
    def __init__(self, df, binary_change_cols, target_col):
        self.df = df
        self.binary_change_cols = binary_change_cols
        self.target_col = target_col
        self.rules_df = None
        self.frequent_itemsets = None
    
    def prepare_transaction_data(self):
        """
        Prepare data for association rule mining.
        Each row becomes a 'transaction' containing the changes that occurred.
        """
        transactions = []
        
        for idx, row in self.df.iterrows():
            transaction = []
            
            # Add field changes to transaction
            for col in self.binary_change_cols:
                if row[col] == 1:
                    transaction.append(f"{col}_changed")
            
            # Add outcome to transaction
            if row[self.target_col] == 1:
                transaction.append("BREAK_OCCURRED")
            else:
                transaction.append("NO_BREAK")
            
            transactions.append(transaction)
        
        return transactions
    
    def find_association_rules(self, min_support=0.01, min_confidence=0.5, min_lift=1.0):
        """
        Find association rules using Apriori algorithm.
        
        Parameters:
        - min_support: Minimum support threshold (frequency of itemset)
        - min_confidence: Minimum confidence threshold (reliability of rule)
        - min_lift: Minimum lift threshold (strength of association)
        """
        # Prepare transaction data
        transactions = self.prepare_transaction_data()
        
        # Convert to binary matrix format
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)
        
        # Find frequent itemsets
        self.frequent_itemsets = apriori(df_encoded, 
                                       min_support=min_support, 
                                       use_colnames=True, 
                                       verbose=1)
        
        # Generate association rules
        if len(self.frequent_itemsets) > 0:
            self.rules_df = association_rules(self.frequent_itemsets, 
                                            metric="confidence", 
                                            min_threshold=min_confidence)
            
            # Filter by lift
            self.rules_df = self.rules_df[self.rules_df['lift'] >= min_lift]
            
            # Sort by confidence and lift
            self.rules_df = self.rules_df.sort_values(['confidence', 'lift'], 
                                                    ascending=[False, False])
        
        return self.rules_df
    
    def analyze_break_patterns(self):
        """
        Focus specifically on rules that lead to breaks.
        """
        if self.rules_df is None:
            print("Please run find_association_rules first!")
            return None
        
        # Filter rules where consequent is BREAK_OCCURRED
        break_rules = self.rules_df[
            self.rules_df['consequents'].apply(
                lambda x: 'BREAK_OCCURRED' in x
            )
        ].copy()
        
        # Add interpretable columns
        break_rules['antecedent_changes'] = break_rules['antecedents'].apply(
            lambda x: [item.replace('_changed', '') for item in x if '_changed' in item]
        )
        break_rules['num_changes'] = break_rules['antecedent_changes'].apply(len)
        
        return break_rules
    
    def get_top_break_predictors(self, top_n=10):
        """
        Get the top N most reliable combinations that predict breaks.
        """
        break_rules = self.analyze_break_patterns()
        
        if break_rules is None or len(break_rules) == 0:
            print("No break prediction rules found!")
            return None
        
        # Select top rules by confidence
        top_rules = break_rules.head(top_n)[['antecedent_changes', 'support', 
                                           'confidence', 'lift', 'num_changes']]
        
        return top_rules
    
    def visualize_rules(self, top_n=15):
        """
        Create visualizations of the association rules.
        """
        break_rules = self.analyze_break_patterns()
        
        if break_rules is None or len(break_rules) == 0:
            print("No rules to visualize!")
            return
        
        # Take top N rules
        plot_data = break_rules.head(top_n).copy()
        
        # Create rule labels
        plot_data['rule_label'] = plot_data['antecedent_changes'].apply(
            lambda x: ' + '.join(x[:3]) + ('...' if len(x) > 3 else '')
        )
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Support vs Confidence scatter
        axes[0,0].scatter(plot_data['support'], plot_data['confidence'], 
                         s=plot_data['lift']*20, alpha=0.6, c=plot_data['num_changes'])
        axes[0,0].set_xlabel('Support')
        axes[0,0].set_ylabel('Confidence')
        axes[0,0].set_title('Support vs Confidence (size=lift, color=num_changes)')
        
        # Confidence bar chart
        axes[0,1].barh(range(len(plot_data)), plot_data['confidence'])
        axes[0,1].set_yticks(range(len(plot_data)))
        axes[0,1].set_yticklabels(plot_data['rule_label'], fontsize=8)
        axes[0,1].set_xlabel('Confidence')
        axes[0,1].set_title('Rule Confidence')
        
        # Lift bar chart
        axes[1,0].barh(range(len(plot_data)), plot_data['lift'])
        axes[1,0].set_yticks(range(len(plot_data)))
        axes[1,0].set_yticklabels(plot_data['rule_label'], fontsize=8)
        axes[1,0].set_xlabel('Lift')
        axes[1,0].set_title('Rule Lift')
        
        # Number of changes histogram
        axes[1,1].hist(plot_data['num_changes'], bins=range(1, max(plot_data['num_changes'])+2))
        axes[1,1].set_xlabel('Number of Changes in Rule')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of Rule Complexity')
        
        plt.tight_layout()
        plt.show()

# Example usage:
# Assuming you have your dataframe ready
# analyzer = BreakAssociationAnalyzer(df, binary_change_cols, 'target')
# rules = analyzer.find_association_rules(min_support=0.02, min_confidence=0.6, min_lift=1.5)
# top_predictors = analyzer.get_top_break_predictors(10)
# analyzer.visualize_rules()

# Additional analysis functions:

def analyze_rule_stability(df, binary_change_cols, target_col, n_splits=5):
    """
    Check if association rules are stable across different data splits.
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_rules = []
    
    for train_idx, _ in skf.split(df, df[target_col]):
        train_df = df.iloc[train_idx]
        analyzer = BreakAssociationAnalyzer(train_df, binary_change_cols, target_col)
        rules = analyzer.find_association_rules()
        
        if rules is not None and len(rules) > 0:
            break_rules = analyzer.analyze_break_patterns()
            if break_rules is not None:
                all_rules.append(break_rules)
    
    return all_rules

def compare_break_vs_no_break_patterns(df, binary_change_cols, target_col):
    """
    Compare association patterns between break and no-break cases.
    """
    break_df = df[df[target_col] == 1]
    no_break_df = df[df[target_col] == 0].sample(len(break_df))  # Balance the datasets
    
    # Analyze break patterns
    break_analyzer = BreakAssociationAnalyzer(break_df, binary_change_cols, target_col)
    break_transactions = break_analyzer.prepare_transaction_data()
    
    # Analyze no-break patterns
    no_break_analyzer = BreakAssociationAnalyzer(no_break_df, binary_change_cols, target_col)
    no_break_transactions = no_break_analyzer.prepare_transaction_data()
    
    return break_transactions, no_break_transactions


"""
Method 5: Co-occurrence Network Analysis
This approach builds a network where nodes are your change indicators and edges represent how often they occur together, especially before breaks.

Key Insights from Both Methods:
Association Rule Mining tells you:

- Specific combinations: "When fields A, B, and C all change, breaks occur 85% of the time"
- Rule strength: How reliable these patterns are (confidence) and how surprising they are (lift)
- Pattern frequency: How often these combinations actually occur in your data

Co-occurrence Network Analysis reveals:

- Field relationships: Which fields tend to change together regardless of outcome
- Central fields: Fields that are involved in many different change combinations (high centrality)
- Communities: Groups of fields that always change together (potential system dependencies)
- Break-specific patterns: Combinations that are much more common in break scenarios

Practical Interpretation Tips:
- High-confidence association rules with high lift are your best break predictors. Look for rules like:

{field_A_changed, field_B_changed} → BREAK_OCCURRED (confidence: 0.9, lift: 3.2)

- High centrality nodes in the break network are "hub" fields - when they change, many other fields tend to change too, often leading to breaks.
- Dense communities in the difference network show groups of fields that are tightly coupled in break scenarios but not in normal operation.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from scipy.stats import chi2_contingency
import seaborn as sns
from collections import defaultdict
import itertools

class BreakCooccurrenceNetwork:
    def __init__(self, df, binary_change_cols, target_col):
        self.df = df
        self.binary_change_cols = binary_change_cols
        self.target_col = target_col
        self.break_network = None
        self.no_break_network = None
        self.combined_network = None
        
    def calculate_cooccurrence_matrix(self, subset_df):
        """
        Calculate co-occurrence matrix for field changes.
        """
        change_data = subset_df[self.binary_change_cols]
        
        # Calculate co-occurrence (dot product of transpose)
        cooccurrence = change_data.T.dot(change_data)
        
        # Convert to correlation-like measure (Jaccard similarity)
        jaccard_matrix = np.zeros_like(cooccurrence, dtype=float)
        
        for i, col1 in enumerate(self.binary_change_cols):
            for j, col2 in enumerate(self.binary_change_cols):
                if i != j:
                    # Calculate Jaccard similarity
                    intersection = ((change_data[col1] == 1) & (change_data[col2] == 1)).sum()
                    union = ((change_data[col1] == 1) | (change_data[col2] == 1)).sum()
                    jaccard_matrix[i,j] = intersection / union if union > 0 else 0
                else:
                    jaccard_matrix[i,j] = 1.0
        
        return pd.DataFrame(jaccard_matrix, 
                          index=self.binary_change_cols, 
                          columns=self.binary_change_cols)
    
    def build_networks(self, min_cooccurrence=0.1):
        """
        Build separate networks for break and no-break cases.
        """
        # Separate data
        break_df = self.df[self.df[self.target_col] == 1]
        no_break_df = self.df[self.df[self.target_col] == 0]
        
        print(f"Break cases: {len(break_df)}, No-break cases: {len(no_break_df)}")
        
        # Calculate co-occurrence matrices
        break_cooccurrence = self.calculate_cooccurrence_matrix(break_df)
        no_break_cooccurrence = self.calculate_cooccurrence_matrix(no_break_df)
        
        # Build networks
        self.break_network = self._matrix_to_network(break_cooccurrence, 
                                                   min_cooccurrence, 
                                                   "break")
        self.no_break_network = self._matrix_to_network(no_break_cooccurrence, 
                                                      min_cooccurrence, 
                                                      "no_break")
        
        # Create difference network (break - no_break)
        diff_matrix = break_cooccurrence - no_break_cooccurrence
        self.combined_network = self._matrix_to_network(diff_matrix, 
                                                      0.05,  # Lower threshold for differences
                                                      "difference")
        
        return self.break_network, self.no_break_network, self.combined_network
    
    def _matrix_to_network(self, matrix, threshold, network_type):
        """
        Convert co-occurrence matrix to NetworkX graph.
        """
        G = nx.Graph()
        
        # Add nodes
        for col in self.binary_change_cols:
            G.add_node(col)
        
        # Add edges
        for i, col1 in enumerate(self.binary_change_cols):
            for j, col2 in enumerate(self.binary_change_cols):
                if i < j:  # Avoid duplicate edges
                    weight = matrix.iloc[i, j]
                    if abs(weight) > threshold:
                        G.add_edge(col1, col2, 
                                 weight=weight, 
                                 network_type=network_type)
        
        return G
    
    def analyze_network_properties(self, network, network_name):
        """
        Calculate various network properties.
        """
        if network.number_of_nodes() == 0:
            return None
        
        properties = {
            'network_name': network_name,
            'num_nodes': network.number_of_nodes(),
            'num_edges': network.number_of_edges(),
            'density': nx.density(network),
            'avg_clustering': nx.average_clustering(network) if network.number_of_edges() > 0 else 0,
            'num_components': nx.number_connected_components(network)
        }
        
        # Centrality measures
        if network.number_of_edges() > 0:
            degree_centrality = nx.degree_centrality(network)
            betweenness_centrality = nx.betweenness_centrality(network)
            closeness_centrality = nx.closeness_centrality(network)
            
            # Get top central nodes
            properties['top_degree_nodes'] = sorted(degree_centrality.items(), 
                                                   key=lambda x: x[1], 
                                                   reverse=True)[:5]
            properties['top_betweenness_nodes'] = sorted(betweenness_centrality.items(), 
                                                       key=lambda x: x[1], 
                                                       reverse=True)[:5]
            
            # Store full centrality measures
            properties['degree_centrality'] = degree_centrality
            properties['betweenness_centrality'] = betweenness_centrality
            properties['closeness_centrality'] = closeness_centrality
        
        return properties
    
    def find_critical_combinations(self, top_n=10):
        """
        Identify the most critical field change combinations for breaks.
        """
        if self.break_network is None:
            print("Please build networks first!")
            return None
        
        # Get edges with highest weights in break network
        break_edges = [(u, v, d['weight']) for u, v, d in self.break_network.edges(data=True)]
        break_edges.sort(key=lambda x: x[2], reverse=True)
        
        # Get edges with highest positive difference (more common in breaks)
        diff_edges = [(u, v, d['weight']) for u, v, d in self.combined_network.edges(data=True)]
        diff_edges.sort(key=lambda x: x[2], reverse=True)
        
        results = {
            'strongest_break_pairs': break_edges[:top_n],
            'most_break_specific_pairs': diff_edges[:top_n]
        }
        
        return results
    
    def detect_communities(self, network):
        """
        Find communities (clusters) of field changes that occur together.
        """
        if network.number_of_edges() == 0:
            return []
        
        # Use Louvain community detection
        try:
            communities = nx.community.louvain_communities(network)
            return list(communities)
        except:
            # Fallback to simple connected components
            return list(nx.connected_components(network))
    
    def visualize_networks(self, figsize=(20, 12)):
        """
        Create comprehensive network visualizations.
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        networks = [
            (self.break_network, "Break Network", 'red'),
            (self.no_break_network, "No-Break Network", 'blue'),
            (self.combined_network, "Difference Network", 'green')
        ]
        
        for idx, (network, title, color) in enumerate(networks):
            if network.number_of_nodes() == 0:
                axes[0, idx].text(0.5, 0.5, 'No edges found', 
                                ha='center', va='center', transform=axes[0, idx].transAxes)
                axes[0, idx].set_title(title)
                continue
            
            # Network layout
            pos = nx.spring_layout(network, k=1, iterations=50)
            
            # Draw nodes
            node_sizes = [network.degree(node) * 100 + 200 for node in network.nodes()]
            nx.draw_networkx_nodes(network, pos, ax=axes[0, idx],
                                 node_color=color, node_size=node_sizes, alpha=0.7)
            
            # Draw edges with thickness based on weight
            edges = network.edges(data=True)
            weights = [abs(d['weight']) for u, v, d in edges]
            if weights:
                max_weight = max(weights)
                edge_widths = [5 * abs(d['weight']) / max_weight for u, v, d in edges]
                nx.draw_networkx_edges(network, pos, ax=axes[0, idx],
                                     width=edge_widths, alpha=0.5, edge_color='gray')
            
            # Draw labels
            nx.draw_networkx_labels(network, pos, ax=axes[0, idx], 
                                  font_size=8, font_weight='bold')
            
            axes[0, idx].set_title(title)
            axes[0, idx].axis('off')
        
        # Network properties comparison
        properties_data = []
        for network, title, _ in networks:
            props = self.analyze_network_properties(network, title)
            if props:
                properties_data.append(props)
        
        if properties_data:
            # Create comparison plots
            prop_df = pd.DataFrame(properties_data)
            
            # Density comparison
            axes[1, 0].bar(prop_df['network_name'], prop_df['density'])
            axes[1, 0].set_title('Network Density')
            axes[1, 0].set_ylabel('Density')
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # Number of edges comparison
            axes[1, 1].bar(prop_df['network_name'], prop_df['num_edges'])
            axes[1, 1].set_title('Number of Edges')
            axes[1, 1].set_ylabel('Edges')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
            
            # Average clustering
            axes[1, 2].bar(prop_df['network_name'], prop_df['avg_clustering'])
            axes[1, 2].set_title('Average Clustering')
            axes[1, 2].set_ylabel('Clustering Coefficient')
            plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def get_network_insights(self):
        """
        Generate comprehensive insights from the network analysis.
        """
        insights = {}
        
        # Analyze each network
        for network_name, network in [('break', self.break_network), 
                                    ('no_break', self.no_break_network),
                                    ('difference', self.combined_network)]:
            if network and network.number_of_edges() > 0:
                props = self.analyze_network_properties(network, network_name)
                insights[network_name] = props
                
                # Find communities
                communities = self.detect_communities(network)
                insights[f'{network_name}_communities'] = communities
        
        # Critical combinations
        critical = self.find_critical_combinations()
        if critical:
            insights['critical_combinations'] = critical
        
        return insights

# Example usage functions:
def compare_field_importance_networks(df, binary_change_cols, target_col):
    """
    Compare individual field importance vs network-based importance.
    """
    # Individual field importance (correlation with target)
    individual_importance = {}
    for col in binary_change_cols:
        corr = df[col].corr(df[target_col])
        individual_importance[col] = abs(corr)
    
    # Network-based importance (centrality in break network)
    network_analyzer = BreakCooccurrenceNetwork(df, binary_change_cols, target_col)
    network_analyzer.build_networks()
    insights = network_analyzer.get_network_insights()
    
    if 'break' in insights and 'degree_centrality' in insights['break']:
        network_importance = insights['break']['degree_centrality']
        
        # Compare
        comparison_df = pd.DataFrame({
            'field': binary_change_cols,
            'individual_importance': [individual_importance.get(col, 0) for col in binary_change_cols],
            'network_centrality': [network_importance.get(col, 0) for col in binary_change_cols]
        })
        
        return comparison_df
    
    return None

# Usage example:
# network_analyzer = BreakCooccurrenceNetwork(df, binary_change_cols, 'target')
# break_net, no_break_net, diff_net = network_analyzer.build_networks(min_cooccurrence=0.1)
# insights = network_analyzer.get_network_insights()
# network_analyzer.visualize_networks()
# critical_combos = network_analyzer.find_critical_combinations()
