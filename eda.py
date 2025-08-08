"""
1. Pairwise Interaction Analysis

- Chi-square test for combinations: For your 20 binary change indicators, test all pairwise combinations against the break target:
- Interaction strength metrics: Calculate lift or odds ratios for combinations:

Key Patterns to Look For in Chi-Square Results:
1. Statistical Significance Patterns

- Red zones (p < 0.001): Extremely reliable break predictors
- Clusters of significance: Multiple related fields showing strong associations
- Significance escalation: Individual → pairwise → triplet significance increases

2. Effect Size Patterns

- High Cramér's V (>0.3): Strong practical significance
- Break rate jumps: Look for combinations where break rate jumps from 10% → 60%+
- Lift patterns: Individual lifts of 2-3x that become 5-10x in combinations

3. Interaction Patterns

- Synergistic effects: Pairs that predict breaks better than either field alone
- Threshold effects: Certain combinations that cross critical break rate thresholds
- Diminishing returns: When adding more fields doesn't improve prediction

4. Practical Patterns

- Sample size considerations: Strong effects with adequate sample sizes (>20 cases)
- False discovery rate: Multiple testing corrections for many comparisons
- Business relevance: High-impact combinations that are operationally meaningful

Interpretation Guide:
Individual fields with χ² > 10 and p < 0.001: Your most reliable single predictors
Pairs with interaction lift > 1.5: Combinations more dangerous than individual fields
Break rates > 50%: Combinations that should trigger immediate alerts
Effect sizes > 0.2: Practically significant relationships worth monitoring

The visualizations will help you spot patterns like:
- Which field combinations form "danger zones"
- Whether certain fields act as "catalysts" (appear in many high-chi² pairs)
- If there are natural groupings of fields that change together before breaks
"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class ChiSquareBreakAnalyzer:
    def __init__(self, df, binary_change_cols, target_col):
        self.df = df
        self.binary_change_cols = binary_change_cols
        self.target_col = target_col
        self.individual_results = None
        self.pairwise_results = None
        self.triplet_results = None
        
    def individual_chi_square_tests(self):
        """
        Perform chi-square tests for each individual field change vs break.
        """
        results = []
        
        for col in self.binary_change_cols:
            # Create contingency table
            contingency = pd.crosstab(self.df[col], self.df[self.target_col])
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
            
            # Calculate effect size (Cramér's V)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))
            
            # Calculate odds ratio (for 2x2 table)
            if contingency.shape == (2, 2):
                odds_ratio = (contingency.iloc[1,1] * contingency.iloc[0,0]) / \
                           (contingency.iloc[1,0] * contingency.iloc[0,1])
            else:
                odds_ratio = np.nan
            
            # Break rate when field changes vs doesn't change
            break_rate_when_changed = contingency.iloc[1,1] / contingency.iloc[1,:].sum() if contingency.iloc[1,:].sum() > 0 else 0
            break_rate_when_not_changed = contingency.iloc[0,1] / contingency.iloc[0,:].sum() if contingency.iloc[0,:].sum() > 0 else 0
            
            results.append({
                'field': col,
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'odds_ratio': odds_ratio,
                'break_rate_when_changed': break_rate_when_changed,
                'break_rate_when_not_changed': break_rate_when_not_changed,
                'lift': break_rate_when_changed / break_rate_when_not_changed if break_rate_when_not_changed > 0 else np.inf,
                'significance_level': self._get_significance_level(p_value)
            })
        
        self.individual_results = pd.DataFrame(results).sort_values('chi2_statistic', ascending=False)
        return self.individual_results
    
    def pairwise_chi_square_tests(self, top_n_fields=None):
        """
        Perform chi-square tests for pairs of field changes vs break.
        """
        # Use top N most significant individual fields if specified
        if top_n_fields and self.individual_results is not None:
            fields_to_test = self.individual_results.head(top_n_fields)['field'].tolist()
        else:
            fields_to_test = self.binary_change_cols
        
        results = []
        
        for col1, col2 in combinations(fields_to_test, 2):
            # Create combined field (both changes occur)
            combined_change = ((self.df[col1] == 1) & (self.df[col2] == 1)).astype(int)
            
            # Skip if no cases where both change
            if combined_change.sum() == 0:
                continue
            
            # Create contingency table
            contingency = pd.crosstab(combined_change, self.df[self.target_col])
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
            
            # Calculate metrics
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))
            
            # Break rates
            break_rate_both_changed = contingency.iloc[1,1] / contingency.iloc[1,:].sum() if contingency.iloc[1,:].sum() > 0 else 0
            break_rate_not_both = contingency.iloc[0,1] / contingency.iloc[0,:].sum() if contingency.iloc[0,:].sum() > 0 else 0
            
            # Compare to individual field performance
            individual_1 = self.individual_results[self.individual_results['field'] == col1]['break_rate_when_changed'].iloc[0]
            individual_2 = self.individual_results[self.individual_results['field'] == col2]['break_rate_when_changed'].iloc[0]
            
            # Interaction effect (is combination better than individual?)
            expected_combined = max(individual_1, individual_2)  # Simple expectation
            interaction_lift = break_rate_both_changed / expected_combined if expected_combined > 0 else 0
            
            results.append({
                'field_1': col1,
                'field_2': col2,
                'pair_name': f"{col1} + {col2}",
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'break_rate_both_changed': break_rate_both_changed,
                'break_rate_not_both': break_rate_not_both,
                'lift': break_rate_both_changed / break_rate_not_both if break_rate_not_both > 0 else np.inf,
                'interaction_lift': interaction_lift,
                'cases_both_changed': combined_change.sum(),
                'significance_level': self._get_significance_level(p_value)
            })
        
        self.pairwise_results = pd.DataFrame(results).sort_values('chi2_statistic', ascending=False)
        return self.pairwise_results
    
    def triplet_chi_square_tests(self, top_n_pairs=10):
        """
        Test triplets based on best-performing pairs.
        """
        if self.pairwise_results is None:
            print("Run pairwise tests first!")
            return None
        
        # Get top pairs and their constituent fields
        top_pairs = self.pairwise_results.head(top_n_pairs)
        top_fields = set()
        for _, row in top_pairs.iterrows():
            top_fields.add(row['field_1'])
            top_fields.add(row['field_2'])
        
        results = []
        
        for col1, col2, col3 in combinations(list(top_fields), 3):
            # Create combined field (all three changes occur)
            combined_change = ((self.df[col1] == 1) & (self.df[col2] == 1) & (self.df[col3] == 1)).astype(int)
            
            # Skip if too few cases
            if combined_change.sum() < 5:
                continue
            
            # Create contingency table
            contingency = pd.crosstab(combined_change, self.df[self.target_col])
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency)
            
            # Calculate metrics
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency.shape) - 1)))
            
            break_rate_all_changed = contingency.iloc[1,1] / contingency.iloc[1,:].sum() if contingency.iloc[1,:].sum() > 0 else 0
            break_rate_not_all = contingency.iloc[0,1] / contingency.iloc[0,:].sum() if contingency.iloc[0,:].sum() > 0 else 0
            
            results.append({
                'field_1': col1,
                'field_2': col2,
                'field_3': col3,
                'triplet_name': f"{col1} + {col2} + {col3}",
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'cramers_v': cramers_v,
                'break_rate_all_changed': break_rate_all_changed,
                'break_rate_not_all': break_rate_not_all,
                'lift': break_rate_all_changed / break_rate_not_all if break_rate_not_all > 0 else np.inf,
                'cases_all_changed': combined_change.sum(),
                'significance_level': self._get_significance_level(p_value)
            })
        
        self.triplet_results = pd.DataFrame(results).sort_values('chi2_statistic', ascending=False)
        return self.triplet_results
    
    def _get_significance_level(self, p_value):
        """Convert p-value to significance level category."""
        if p_value < 0.001:
            return 'Highly Significant (p<0.001)'
        elif p_value < 0.01:
            return 'Very Significant (p<0.01)'
        elif p_value < 0.05:
            return 'Significant (p<0.05)'
        elif p_value < 0.1:
            return 'Marginally Significant (p<0.1)'
        else:
            return 'Not Significant (p>=0.1)'
    
    def create_comprehensive_visualizations(self, figsize=(20, 16)):
        """
        Create comprehensive visualizations of chi-square patterns.
        """
        fig = plt.figure(figsize=figsize)
        
        # Create a complex subplot layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Individual field chi-square statistics (top plot, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if self.individual_results is not None:
            top_individual = self.individual_results.head(15)
            colors = ['red' if x < 0.001 else 'orange' if x < 0.01 else 'yellow' if x < 0.05 else 'gray' 
                     for x in top_individual['p_value']]
            
            bars = ax1.barh(range(len(top_individual)), top_individual['chi2_statistic'], color=colors)
            ax1.set_yticks(range(len(top_individual)))
            ax1.set_yticklabels(top_individual['field'], fontsize=10)
            ax1.set_xlabel('Chi-Square Statistic')
            ax1.set_title('Individual Field Changes vs Break\n(Red: p<0.001, Orange: p<0.01, Yellow: p<0.05)')
            
            # Add chi2 values on bars
            for i, (bar, chi2_val) in enumerate(zip(bars, top_individual['chi2_statistic'])):
                ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{chi2_val:.1f}', va='center', fontsize=8)
        
        # 2. Effect sizes (Cramér's V) vs Statistical significance
        ax2 = fig.add_subplot(gs[0, 2:])
        if self.individual_results is not None:
            scatter = ax2.scatter(self.individual_results['cramers_v'], 
                                -np.log10(self.individual_results['p_value']),
                                s=self.individual_results['lift']*20,
                                c=self.individual_results['break_rate_when_changed'],
                                cmap='Reds', alpha=0.7)
            
            ax2.set_xlabel('Effect Size (Cramér\'s V)')
            ax2.set_ylabel('-log10(p-value)')
            ax2.set_title('Effect Size vs Statistical Significance\n(Size=Lift, Color=Break Rate)')
            
            # Add significance threshold lines
            ax2.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='p=0.05')
            ax2.axhline(y=-np.log10(0.001), color='red', linestyle='--', alpha=0.5, label='p=0.001')
            
            plt.colorbar(scatter, ax=ax2, label='Break Rate When Changed')
        
        # 3. Pairwise interaction heatmap
        ax3 = fig.add_subplot(gs[1, :2])
        if self.pairwise_results is not None and len(self.pairwise_results) > 0:
            # Create a matrix for visualization
            top_pairs = self.pairwise_results.head(20)
            pair_matrix = np.zeros((len(self.binary_change_cols), len(self.binary_change_cols)))
            
            for _, row in top_pairs.iterrows():
                i = self.binary_change_cols.index(row['field_1'])
                j = self.binary_change_cols.index(row['field_2'])
                pair_matrix[i, j] = row['chi2_statistic']
                pair_matrix[j, i] = row['chi2_statistic']
            
            # Mask the diagonal and zero values
            mask = np.triu(np.ones_like(pair_matrix, dtype=bool))
            pair_matrix[pair_matrix == 0] = np.nan
            
            sns.heatmap(pair_matrix, mask=mask, annot=False, cmap='Reds', 
                       xticklabels=self.binary_change_cols, yticklabels=self.binary_change_cols,
                       ax=ax3, cbar_kws={'label': 'Chi-Square Statistic'})
            ax3.set_title('Pairwise Field Combinations Chi-Square Heatmap')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8)
        
        # 4. Top pairwise combinations
        ax4 = fig.add_subplot(gs[1, 2:])
        if self.pairwise_results is not None:
            top_pairs = self.pairwise_results.head(10)
            colors = ['red' if x < 0.001 else 'orange' if x < 0.01 else 'yellow' if x < 0.05 else 'gray' 
                     for x in top_pairs['p_value']]
            
            bars = ax4.barh(range(len(top_pairs)), top_pairs['chi2_statistic'], color=colors)
            ax4.set_yticks(range(len(top_pairs)))
            ax4.set_yticklabels([name[:20] + '...' if len(name) > 20 else name 
                               for name in top_pairs['pair_name']], fontsize=8)
            ax4.set_xlabel('Chi-Square Statistic')
            ax4.set_title('Top Pairwise Combinations')
        
        # 5. Break rate comparison: Individual vs Pairs vs Triplets
        ax5 = fig.add_subplot(gs[2, :2])
        break_rates = []
        categories = []
        
        if self.individual_results is not None:
            top_individual = self.individual_results.head(5)
            break_rates.extend(top_individual['break_rate_when_changed'].tolist())
            categories.extend([f"Ind: {field[:10]}" for field in top_individual['field']])
        
        if self.pairwise_results is not None:
            top_pairs = self.pairwise_results.head(5)
            break_rates.extend(top_pairs['break_rate_both_changed'].tolist())
            categories.extend([f"Pair: {name[:10]}..." for name in top_pairs['pair_name']])
        
        if self.triplet_results is not None:
            top_triplets = self.triplet_results.head(3)
            break_rates.extend(top_triplets['break_rate_all_changed'].tolist())
            categories.extend([f"Trip: {name[:10]}..." for name in top_triplets['triplet_name']])
        
        if break_rates:
            colors = ['blue']*5 + ['green']*5 + ['purple']*3
            ax5.bar(range(len(break_rates)), break_rates, color=colors[:len(break_rates)])
            ax5.set_xticks(range(len(categories)))
            ax5.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
            ax5.set_ylabel('Break Rate')
            ax5.set_title('Break Rates: Individual vs Pairs vs Triplets')
        
        # 6. Statistical power analysis
        ax6 = fig.add_subplot(gs[2, 2:])
        if self.individual_results is not None:
            # Plot relationship between sample size and statistical power
            chi2_stats = self.individual_results['chi2_statistic']
            p_values = self.individual_results['p_value']
            effect_sizes = self.individual_results['cramers_v']
            
            # Calculate approximate sample sizes needed for each effect
            sample_sizes = []
            for cramers_v in effect_sizes:
                # Rough approximation for sample size needed to detect effect
                if cramers_v > 0:
                    n_needed = (2.706 / (cramers_v ** 2))  # For alpha=0.1, power=0.8
                    sample_sizes.append(min(n_needed, 10000))  # Cap at 10k for visualization
                else:
                    sample_sizes.append(10000)
            
            scatter = ax6.scatter(effect_sizes, -np.log10(p_values), 
                                s=[min(s/50, 200) for s in sample_sizes], 
                                c=chi2_stats, cmap='viridis', alpha=0.7)
            ax6.set_xlabel('Effect Size (Cramér\'s V)')
            ax6.set_ylabel('-log10(p-value)')
            ax6.set_title('Statistical Power Analysis\n(Size ∝ Sample Size Needed)')
            plt.colorbar(scatter, ax=ax6, label='Chi-Square Statistic')
        
        # 7. Pattern discovery summary
        ax7 = fig.add_subplot(gs[3, :])
        
        # Create summary statistics
        summary_text = self._generate_pattern_summary()
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis('off')
        ax7.set_title('Pattern Discovery Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _generate_pattern_summary(self):
        """Generate a text summary of discovered patterns."""
        summary = []
        
        if self.individual_results is not None:
            n_significant = len(self.individual_results[self.individual_results['p_value'] < 0.05])
            top_field = self.individual_results.iloc[0]
            summary.append(f"INDIVIDUAL FIELDS:")
            summary.append(f"• {n_significant}/{len(self.individual_results)} fields significantly associated with breaks")
            summary.append(f"• Top predictor: {top_field['field']}")
            summary.append(f"  - Break rate when changed: {top_field['break_rate_when_changed']:.1%}")
            summary.append(f"  - Lift: {top_field['lift']:.1f}x")
            summary.append("")
        
        if self.pairwise_results is not None:
            n_significant = len(self.pairwise_results[self.pairwise_results['p_value'] < 0.05])
            if len(self.pairwise_results) > 0:
                top_pair = self.pairwise_results.iloc[0]
                summary.append(f"PAIRWISE COMBINATIONS:")
                summary.append(f"• {n_significant}/{len(self.pairwise_results)} pairs significantly associated with breaks")
                summary.append(f"• Top combination: {top_pair['pair_name']}")
                summary.append(f"  - Break rate when both change: {top_pair['break_rate_both_changed']:.1%}")
                summary.append(f"  - Interaction lift: {top_pair['interaction_lift']:.1f}x")
                summary.append("")
        
        if self.triplet_results is not None and len(self.triplet_results) > 0:
            top_triplet = self.triplet_results.iloc[0]
            summary.append(f"TRIPLET COMBINATIONS:")
            summary.append(f"• Top triplet: {top_triplet['triplet_name']}")
            summary.append(f"  - Break rate when all change: {top_triplet['break_rate_all_changed']:.1%}")
            summary.append("")
        
        # Add recommendations
        summary.append("RECOMMENDATIONS:")
        if self.individual_results is not None:
            high_lift_fields = self.individual_results[self.individual_results['lift'] > 2]
            if len(high_lift_fields) > 0:
                summary.append(f"• Monitor these high-impact fields: {', '.join(high_lift_fields['field'].head(3))}")
        
        if self.pairwise_results is not None:
            high_interaction = self.pairwise_results[self.pairwise_results['interaction_lift'] > 1.5]
            if len(high_interaction) > 0:
                summary.append(f"• Watch for these dangerous combinations:")
                for _, row in high_interaction.head(2).iterrows():
                    summary.append(f"  - {row['field_1']} + {row['field_2']}")
        
        return "\n".join(summary)
    
    def get_actionable_insights(self, min_cases=10, min_break_rate=0.3):
        """
        Extract actionable insights for break prediction.
        """
        insights = {
            'high_impact_individual_fields': [],
            'dangerous_combinations': [],
            'monitoring_recommendations': []
        }
        
        # High-impact individual fields
        if self.individual_results is not None:
            high_impact = self.individual_results[
                (self.individual_results['p_value'] < 0.05) &
                (self.individual_results['break_rate_when_changed'] > min_break_rate) &
                (self.individual_results['lift'] > 1.5)
            ]
            insights['high_impact_individual_fields'] = high_impact.to_dict('records')
        
        # Dangerous combinations
        if self.pairwise_results is not None:
            dangerous = self.pairwise_results[
                (self.pairwise_results['p_value'] < 0.05) &
                (self.pairwise_results['cases_both_changed'] >= min_cases) &
                (self.pairwise_results['break_rate_both_changed'] > min_break_rate)
            ]
            insights['dangerous_combinations'] = dangerous.to_dict('records')
        
        return insights

# Example usage:
# analyzer = ChiSquareBreakAnalyzer(df, binary_change_cols, 'target')
# individual_results = analyzer.individual_chi_square_tests()
# pairwise_results = analyzer.pairwise_chi_square_tests(top_n_fields=15)
# triplet_results = analyzer.triplet_chi_square_tests(top_n_pairs=10)
# analyzer.create_comprehensive_visualizations()
# insights = analyzer.get_actionable_insights()

"""
1. Individual Field Lift (in Chi-Square script)
pythonlift = break_rate_when_changed / break_rate_when_not_changed
What it measures: How much more likely a break is when this field changes vs when it doesn't change.
Example: If breaks happen 40% of the time when field A changes, but only 10% when field A doesn't change:

Lift = 0.40 / 0.10 = 4.0
Interpretation: "Field A changing makes breaks 4x more likely"

2. Pairwise Combination Lift (in Chi-Square script)
pythonlift = break_rate_both_changed / break_rate_not_both
What it measures: How much more likely a break is when both fields change vs when they don't both change.
Example: If breaks happen 60% when both A and B change, but 15% in all other cases:

Lift = 0.60 / 0.15 = 4.0
Interpretation: "The combination A+B makes breaks 4x more likely than baseline"

3. Interaction Lift (your function from association rules)
pythonboth_change = df[(df[col1]==1) & (df[col2]==1)][target].mean()
col1_only = df[df[col1]==1][target].mean() 
col2_only = df[df[col2]==1][target].mean()
expected = col1_only + col2_only - (col1_only * col2_only)
lift = both_change / expected
What it measures: How much better the combination performs compared to what you'd expect from the individual fields.
The math:

expected uses probability theory: P(A or B) = P(A) + P(B) - P(A and B)
If fields were independent, this would be the expected break rate
Interpretation: "The combination A+B performs X times better than statistical independence would predict"
"""

import pandas as pd
import numpy as np

def compare_lift_calculations(df, col1, col2, target):
    """
    Compare all three lift calculations with detailed explanations.
    """
    
    # Basic statistics
    total_cases = len(df)
    total_breaks = df[target].sum()
    baseline_break_rate = total_breaks / total_cases
    
    print(f"Dataset Overview:")
    print(f"Total cases: {total_cases}")
    print(f"Total breaks: {total_breaks}")
    print(f"Baseline break rate: {baseline_break_rate:.1%}")
    print("\n" + "="*50 + "\n")
    
    # Individual field statistics
    col1_changes = df[col1].sum()
    col2_changes = df[col2].sum()
    
    col1_break_rate = df[df[col1]==1][target].mean()
    col2_break_rate = df[df[col2]==1][target].mean()
    
    col1_no_change_break_rate = df[df[col1]==0][target].mean()
    col2_no_change_break_rate = df[df[col2]==0][target].mean()
    
    print(f"Individual Field Analysis:")
    print(f"{col1}: {col1_changes} changes ({col1_changes/total_cases:.1%})")
    print(f"  Break rate when {col1} changes: {col1_break_rate:.1%}")
    print(f"  Break rate when {col1} doesn't change: {col1_no_change_break_rate:.1%}")
    print(f"  Individual lift: {col1_break_rate/col1_no_change_break_rate:.2f}x")
    print()
    
    print(f"{col2}: {col2_changes} changes ({col2_changes/total_cases:.1%})")
    print(f"  Break rate when {col2} changes: {col2_break_rate:.1%}")
    print(f"  Break rate when {col2} doesn't change: {col2_no_change_break_rate:.1%}")
    print(f"  Individual lift: {col2_break_rate/col2_no_change_break_rate:.2f}x")
    print("\n" + "="*50 + "\n")
    
    # Combination statistics
    both_change = ((df[col1]==1) & (df[col2]==1))
    both_change_cases = both_change.sum()
    both_change_break_rate = df[both_change][target].mean() if both_change_cases > 0 else 0
    
    not_both_change = ~both_change
    not_both_break_rate = df[not_both_change][target].mean()
    
    print(f"Combination Analysis:")
    print(f"Cases where both {col1} and {col2} change: {both_change_cases} ({both_change_cases/total_cases:.1%})")
    print(f"Break rate when both change: {both_change_break_rate:.1%}")
    print(f"Break rate when NOT both change: {not_both_break_rate:.1%}")
    print("\n" + "="*50 + "\n")
    
    # LIFT CALCULATION 1: Individual Field Lift (from chi-square script)
    individual_lift_1 = col1_break_rate / col1_no_change_break_rate if col1_no_change_break_rate > 0 else np.inf
    individual_lift_2 = col2_break_rate / col2_no_change_break_rate if col2_no_change_break_rate > 0 else np.inf
    
    print(f"LIFT CALCULATION 1 - Individual Field Lift:")
    print(f"Formula: break_rate_when_changed / break_rate_when_not_changed")
    print(f"{col1} lift: {col1_break_rate:.3f} / {col1_no_change_break_rate:.3f} = {individual_lift_1:.2f}x")
    print(f"{col2} lift: {col2_break_rate:.3f} / {col2_no_change_break_rate:.3f} = {individual_lift_2:.2f}x")
    print(f"Interpretation: How much more likely breaks are when each field changes individually")
    print()
    
    # LIFT CALCULATION 2: Pairwise Combination Lift (from chi-square script)
    pairwise_lift = both_change_break_rate / not_both_break_rate if not_both_break_rate > 0 else np.inf
    
    print(f"LIFT CALCULATION 2 - Pairwise Combination Lift:")
    print(f"Formula: break_rate_both_changed / break_rate_not_both")
    print(f"Combination lift: {both_change_break_rate:.3f} / {not_both_break_rate:.3f} = {pairwise_lift:.2f}x")
    print(f"Interpretation: How much more likely breaks are with this combination vs baseline")
    print()
    
    # LIFT CALCULATION 3: Interaction Lift (from association rules)
    # This measures synergy - is the combination better than expected from independence?
    expected_independent = col1_break_rate + col2_break_rate - (col1_break_rate * col2_break_rate)
    interaction_lift = both_change_break_rate / expected_independent if expected_independent > 0 else 0
    
    print(f"LIFT CALCULATION 3 - Interaction/Synergy Lift:")
    print(f"Formula: P(break|both) / [P(break|col1) + P(break|col2) - P(break|col1)*P(break|col2)]")
    print(f"Expected break rate if independent: {col1_break_rate:.3f} + {col2_break_rate:.3f} - {col1_break_rate:.3f}*{col2_break_rate:.3f} = {expected_independent:.3f}")
    print(f"Actual break rate when both change: {both_change_break_rate:.3f}")
    print(f"Interaction lift: {both_change_break_rate:.3f} / {expected_independent:.3f} = {interaction_lift:.2f}x")
    print(f"Interpretation: How much the combination exceeds statistical independence expectations")
    print()
    
    # Summary and recommendations
    print("="*70)
    print("SUMMARY AND INTERPRETATION:")
    print("="*70)
    
    if interaction_lift > 1.2:
        print(f"✓ SYNERGISTIC EFFECT DETECTED!")
        print(f"  The combination performs {interaction_lift:.2f}x better than expected from independence")
        print(f"  This suggests {col1} and {col2} have a genuine interaction effect")
    elif interaction_lift < 0.8:
        print(f"⚠ INTERFERENCE EFFECT DETECTED!")
        print(f"  The combination performs worse than expected ({interaction_lift:.2f}x)")
        print(f"  {col1} and {col2} may interfere with each other")
    else:
        print(f"→ ADDITIVE EFFECT:")
        print(f"  The combination performs as expected from independence ({interaction_lift:.2f}x)")
        print(f"  No special interaction between {col1} and {col2}")
    
    print()
    
    # Practical recommendations
    if pairwise_lift > 3 and both_change_cases >= 10:
        print(f"🚨 HIGH PRIORITY ALERT RULE:")
        print(f"   When both {col1} and {col2} change → {pairwise_lift:.1f}x break risk")
        print(f"   Break probability: {both_change_break_rate:.1%}")
    
    if individual_lift_1 > 2 or individual_lift_2 > 2:
        print(f"⚡ INDIVIDUAL MONITORING:")
        if individual_lift_1 > individual_lift_2:
            print(f"   {col1} is the stronger individual predictor ({individual_lift_1:.1f}x)")
        else:
            print(f"   {col2} is the stronger individual predictor ({individual_lift_2:.1f}x)")
    
    return {
        'individual_lift_1': individual_lift_1,
        'individual_lift_2': individual_lift_2,
        'pairwise_lift': pairwise_lift,
        'interaction_lift': interaction_lift,
        'both_change_cases': both_change_cases,
        'both_change_break_rate': both_change_break_rate,
        'expected_independent': expected_independent
    }

# Example function to demonstrate with synthetic data
def create_example_scenarios():
    """
    Create example scenarios to show different lift patterns.
    """
    np.random.seed(42)
    n_samples = 1000
    
    scenarios = {}
    
    # Scenario 1: Independent effects
    print("SCENARIO 1: INDEPENDENT EFFECTS")
    print("-" * 40)
    df1 = pd.DataFrame({
        'field_a': np.random.binomial(1, 0.2, n_samples),
        'field_b': np.random.binomial(1, 0.15, n_samples),
    })
    # Break probability increases independently for each field
    break_prob = 0.1 + 0.2 * df1['field_a'] + 0.25 * df1['field_b']
    df1['target'] = np.random.binomial(1, break_prob)
    
    result1 = compare_lift_calculations(df1, 'field_a', 'field_b', 'target')
    scenarios['independent'] = result1
    
    print("\n" + "="*80 + "\n")
    
    # Scenario 2: Synergistic effects
    print("SCENARIO 2: SYNERGISTIC EFFECTS")
    print("-" * 40)
    df2 = pd.DataFrame({
        'field_a': np.random.binomial(1, 0.2, n_samples),
        'field_b': np.random.binomial(1, 0.15, n_samples),
    })
    # Break probability has interaction term
    break_prob = 0.05 + 0.15 * df2['field_a'] + 0.2 * df2['field_b'] + 0.4 * df2['field_a'] * df2['field_b']
    df2['target'] = np.random.binomial(1, break_prob)
    
    result2 = compare_lift_calculations(df2, 'field_a', 'field_b', 'target')
    scenarios['synergistic'] = result2
    
    return scenarios

# Example usage:
# scenarios = create_example_scenarios()
# 
# Or with your real data:
# result = compare_lift_calculations(df, 'field_change_1', 'field_change_2', 'target')

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
