"""
Compare Fractional-Order vs Integer-Order Models using Excel data
"""
import pandas as pd
import numpy as np

print("="*80)
print("FRACTIONAL vs INTEGER-ORDER COMPARISON (From Excel Data)")
print("="*80)

# Load all relevant sheets
all_data = pd.read_excel('all_alpha_results_comprehensive.xlsx', sheet_name='All_Data')
efficacy_pivot = pd.read_excel('all_alpha_results_comprehensive.xlsx', sheet_name='Efficacy_by_Patient_Protocol')
tumor_pivot = pd.read_excel('all_alpha_results_comprehensive.xlsx', sheet_name='Tumor_Reduction_by_Patient_Protocol')
resistance_pivot = pd.read_excel('all_alpha_results_comprehensive.xlsx', sheet_name='Resistance_by_Patient_Protocol')
summary_patient = pd.read_excel('all_alpha_results_comprehensive.xlsx', sheet_name='Summary_by_Patient')
summary_protocol = pd.read_excel('all_alpha_results_comprehensive.xlsx', sheet_name='Summary_by_Protocol')

# Separate integer-order (alpha=1.0) and fractional-order (alpha<1.0)
integer_data = all_data[all_data['Alpha'] == 1.0].copy()
fractional_data = all_data[all_data['Alpha'] < 1.0].copy()

print(f"\nData Summary:")
print(f"  Integer-order (alpha=1.0): {len(integer_data)} simulations")
print(f"  Fractional-order (alpha<1.0): {len(fractional_data)} simulations")
print(f"  Total: {len(all_data)} simulations")

# ============================================================================
# AGGREGATE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("1. AGGREGATE COMPARISON")
print("="*80)

metrics = ['Tumor_Reduction_%', 'Final_Resistance_%', 'Efficacy_Score']
comparison_results = []

for metric in metrics:
    int_values = integer_data[metric].dropna()
    frac_values = fractional_data[metric].dropna()
    
    int_mean = int_values.mean()
    int_std = int_values.std()
    int_median = int_values.median()
    
    frac_mean = frac_values.mean()
    frac_std = frac_values.std()
    frac_median = frac_values.median()
    
    diff = frac_mean - int_mean
    diff_pct = (diff / int_mean * 100) if int_mean != 0 else 0
    
    comparison_results.append({
        'Metric': metric,
        'Integer_Mean': int_mean,
        'Integer_Std': int_std,
        'Integer_Median': int_median,
        'Fractional_Mean': frac_mean,
        'Fractional_Std': frac_std,
        'Fractional_Median': frac_median,
        'Difference': diff,
        'Difference_%': diff_pct
    })
    
    print(f"\n{metric}:")
    print(f"  Integer-order:  {int_mean:.3f} ± {int_std:.3f} (median: {int_median:.3f})")
    print(f"  Fractional-order: {frac_mean:.3f} ± {frac_std:.3f} (median: {frac_median:.3f})")
    print(f"  Difference: {diff:+.3f} ({diff_pct:+.2f}%)")

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv('fractional_vs_integer_aggregate.csv', index=False)
print(f"\n✓ Saved: fractional_vs_integer_aggregate.csv")

# ============================================================================
# CONTEXT-SPECIFIC COMPARISON (Using Pivot Tables)
# ============================================================================
print("\n" + "="*80)
print("2. CONTEXT-SPECIFIC COMPARISON")
print("="*80)

# Integer-order column is labeled as "1" in pivot tables
# Fractional-order columns are 0.75, 0.8, 0.85, 0.9, 0.93, 0.95

context_comparison = []

for idx, row in efficacy_pivot.iterrows():
    patient = row['Patient_Profile']
    protocol = row['Protocol']
    int_eff = row[1.0]  # Integer-order (alpha=1.0)
    
    # Calculate mean across all fractional alphas
    frac_alphas = [0.75, 0.8, 0.85, 0.9, 0.93, 0.95]
    frac_effs = [row[alpha] for alpha in frac_alphas if pd.notna(row.get(alpha, None))]
    frac_eff = np.mean(frac_effs) if frac_effs else None
    
    if pd.notna(int_eff) and frac_eff is not None:
        diff = frac_eff - int_eff
        diff_pct = (diff / int_eff * 100) if int_eff != 0 else 0
        
        context_comparison.append({
            'Patient_Profile': patient,
            'Protocol': protocol,
            'Integer_Efficacy': int_eff,
            'Fractional_Efficacy': frac_eff,
            'Difference': diff,
            'Difference_%': diff_pct
        })

context_df = pd.DataFrame(context_comparison)
context_df = context_df.sort_values('Difference_%', ascending=False)
context_df.to_csv('fractional_vs_integer_context.csv', index=False)
print(f"✓ Saved: fractional_vs_integer_context.csv")

print("\nTop 10 contexts where fractional model shows HIGHER efficacy:")
print(context_df.head(10)[['Patient_Profile', 'Protocol', 'Integer_Efficacy', 'Fractional_Efficacy', 'Difference_%']].to_string(index=False))

print("\nTop 10 contexts where fractional model shows LOWER efficacy:")
print(context_df.tail(10)[['Patient_Profile', 'Protocol', 'Integer_Efficacy', 'Fractional_Efficacy', 'Difference_%']].to_string(index=False))

# ============================================================================
# ALPHA-SPECIFIC ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. ALPHA-SPECIFIC ANALYSIS")
print("="*80)

alpha_analysis = []
alphas = [0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 1.0]

for alpha in alphas:
    alpha_data = all_data[all_data['Alpha'] == alpha]
    if len(alpha_data) > 0:
        alpha_analysis.append({
            'Alpha': alpha,
            'Type': 'Fractional' if alpha < 1.0 else 'Integer',
            'Mean_Efficacy': alpha_data['Efficacy_Score'].mean(),
            'Mean_Tumor_Reduction': alpha_data['Tumor_Reduction_%'].mean(),
            'Mean_Resistance': alpha_data['Final_Resistance_%'].mean(),
            'Std_Efficacy': alpha_data['Efficacy_Score'].std(),
        })

alpha_df = pd.DataFrame(alpha_analysis)
int_mean_eff = alpha_df[alpha_df['Alpha'] == 1.0]['Mean_Efficacy'].values[0]
alpha_df['Efficacy_vs_Integer'] = alpha_df['Mean_Efficacy'] - int_mean_eff
alpha_df['Efficacy_vs_Integer_%'] = (alpha_df['Efficacy_vs_Integer'] / int_mean_eff * 100)

alpha_df.to_csv('alpha_specific_analysis.csv', index=False)
print(alpha_df[['Alpha', 'Type', 'Mean_Efficacy', 'Std_Efficacy', 'Efficacy_vs_Integer_%']].to_string(index=False))
print(f"\n✓ Saved: alpha_specific_analysis.csv")

# ============================================================================
# PATIENT-SPECIFIC ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. PATIENT-SPECIFIC ANALYSIS")
print("="*80)

patient_analysis = []
patients = ['average', 'young', 'elderly', 'compromised']

for patient in patients:
    int_patient = integer_data[integer_data['Patient_Profile'] == patient]
    frac_patient = fractional_data[fractional_data['Patient_Profile'] == patient]
    
    if len(int_patient) > 0 and len(frac_patient) > 0:
        int_eff = int_patient['Efficacy_Score'].mean()
        frac_eff = frac_patient['Efficacy_Score'].mean()
        diff_pct = ((frac_eff - int_eff) / int_eff * 100) if int_eff != 0 else 0
        
        patient_analysis.append({
            'Patient_Profile': patient,
            'Integer_Efficacy': int_eff,
            'Fractional_Efficacy': frac_eff,
            'Difference_%': diff_pct
        })

patient_df = pd.DataFrame(patient_analysis)
patient_df = patient_df.sort_values('Difference_%', ascending=False)
patient_df.to_csv('fractional_vs_integer_by_patient.csv', index=False)
print(patient_df.to_string(index=False))
print(f"\n✓ Saved: fractional_vs_integer_by_patient.csv")

# ============================================================================
# PROTOCOL-SPECIFIC ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. PROTOCOL-SPECIFIC ANALYSIS")
print("="*80)

protocol_analysis = []
protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']

for protocol in protocols:
    int_protocol = integer_data[integer_data['Protocol'] == protocol]
    frac_protocol = fractional_data[fractional_data['Protocol'] == protocol]
    
    if len(int_protocol) > 0 and len(frac_protocol) > 0:
        int_eff = int_protocol['Efficacy_Score'].mean()
        frac_eff = frac_protocol['Efficacy_Score'].mean()
        diff_pct = ((frac_eff - int_eff) / int_eff * 100) if int_eff != 0 else 0
        
        protocol_analysis.append({
            'Protocol': protocol,
            'Integer_Efficacy': int_eff,
            'Fractional_Efficacy': frac_eff,
            'Difference_%': diff_pct
        })

protocol_df = pd.DataFrame(protocol_analysis)
protocol_df = protocol_df.sort_values('Difference_%', ascending=False)
protocol_df.to_csv('fractional_vs_integer_by_protocol.csv', index=False)
print(protocol_df.to_string(index=False))
print(f"\n✓ Saved: fractional_vs_integer_by_protocol.csv")

# ============================================================================
# STATISTICAL SIGNIFICANCE TEST
# ============================================================================
print("\n" + "="*80)
print("6. STATISTICAL SIGNIFICANCE")
print("="*80)

from scipy import stats

for metric in metrics:
    int_values = integer_data[metric].dropna()
    frac_values = fractional_data[metric].dropna()
    
    # T-test
    t_stat, p_value = stats.ttest_ind(frac_values, int_values)
    
    print(f"\n{metric}:")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)
print("\nAll comparison files saved successfully!")

