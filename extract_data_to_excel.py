import os
import re
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

# Define alpha values and their corresponding directory names
alpha_mapping = {
    0.75: 'alpha_0_75',
    0.80: 'alpha_0_8',
    0.85: 'alpha_0_85',
    0.90: 'alpha_0_9',
    0.93: 'alpha_0_93',
    0.95: 'alpha_0_95',
    1.0: 'alpha_1_0'
}

def parse_summary_file(filepath, alpha_value):
    """Parse a summary file and extract all relevant data"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    data = {
        'alpha': alpha_value,
        'protocols': {},
        'patient_specific': {},
        'rankings': []
    }
    
    # Extract protocol performance for average patient
    protocol_section = re.search(r'Protocol Performance for Average Patient:(.*?)Patient-Specific', content, re.DOTALL)
    if protocol_section:
        protocol_text = protocol_section.group(1)
        
        # Extract each protocol's data
        protocols = ['Standard', 'Continuous', 'Adaptive', 'Immuno_Combo', 'Hyperthermia']
        for protocol in protocols:
            protocol_match = re.search(
                rf'{protocol} Protocol:(.*?)(?=\n\n|\n[A-Z]|$)',
                protocol_text,
                re.DOTALL
            )
            if protocol_match:
                protocol_data = protocol_match.group(1)
                metrics = {}
                
                # Extract metrics
                tumor_match = re.search(r'Tumor Reduction: ([\d.]+)%', protocol_data)
                if tumor_match:
                    metrics['tumor_reduction'] = float(tumor_match.group(1))
                
                resistance_match = re.search(r'Final Resistance: ([\d.]+)%', protocol_data)
                if resistance_match:
                    metrics['final_resistance'] = float(resistance_match.group(1))
                
                efficacy_match = re.search(r'Efficacy Score: ([\d.]+)', protocol_data)
                if efficacy_match:
                    metrics['efficacy_score'] = float(efficacy_match.group(1))
                
                immune_match = re.search(r'Immune Activation: ([\d.]+)x', protocol_data)
                if immune_match:
                    metrics['immune_activation'] = float(immune_match.group(1))
                
                genetic_match = re.search(r'Genetic Instability: ([\d.]+)', protocol_data)
                if genetic_match:
                    metrics['genetic_instability'] = float(genetic_match.group(1))
                
                metabolic_match = re.search(r'Metabolic Shift: ([\d.]+)', protocol_data)
                if metabolic_match:
                    metrics['metabolic_shift'] = float(metabolic_match.group(1))
                
                data['protocols'][protocol] = metrics
    
    # Extract patient-specific data
    patient_section = re.search(r'Patient-Specific Protocol Performance:(.*?)Key Observations', content, re.DOTALL)
    if patient_section:
        patient_text = patient_section.group(1)
        
        patients = ['Young', 'Elderly', 'Compromised']
        for patient in patients:
            patient_match = re.search(
                rf'{patient} Patient:(.*?)(?=\n\n|\n[A-Z]|$)',
                patient_text,
                re.DOTALL
            )
            if patient_match:
                patient_data = patient_match.group(1)
                patient_metrics = {}
                
                best_match = re.search(r'Best Protocol: (\w+)', patient_data)
                if best_match:
                    patient_metrics['best_protocol'] = best_match.group(1).lower()
                
                efficacy_match = re.search(r'Efficacy Score: ([\d.]+)', patient_data)
                if efficacy_match:
                    patient_metrics['efficacy_score'] = float(efficacy_match.group(1))
                
                tumor_match = re.search(r'Tumor Reduction: ([\d.]+)%', patient_data)
                if tumor_match:
                    patient_metrics['tumor_reduction'] = float(tumor_match.group(1))
                
                resistance_match = re.search(r'Final Resistance: ([\d.]+)%', patient_data)
                if resistance_match:
                    patient_metrics['final_resistance'] = float(resistance_match.group(1))
                
                data['patient_specific'][patient.lower()] = patient_metrics
    
    # Extract protocol rankings
    # NOTE: These averages are calculated across ALL 4 patient profiles (average, young, elderly, compromised)
    # as computed in generate_model_summary function (lines 1617-1627)
    ranking_section = re.search(r'1\. Protocol Effectiveness Ranking:(.*?)(?=\n2\.|$)', content, re.DOTALL)
    if ranking_section:
        ranking_text = ranking_section.group(1)
        ranking_lines = ranking_text.strip().split('\n')
        for line in ranking_lines:
            match = re.search(r'\d+\.\s+(\w+):\s+([\d.]+)', line)
            if match:
                protocol = match.group(1)
                score = float(match.group(2))
                # This score is the average efficacy across all 4 patient profiles for this protocol
                data['rankings'].append({'protocol': protocol, 'avg_score': score})
    
    return data

def create_excel_file():
    """Create Excel file with multiple sheets from all summary files"""
    
    all_data = []
    protocol_data = []
    patient_data = []
    ranking_data = []
    
    # Process each alpha value
    for alpha, dir_suffix in alpha_mapping.items():
        summary_path = f'cancer_model_results_{dir_suffix}/model_summary.txt'
        
        if os.path.exists(summary_path):
            print(f"Processing alpha = {alpha}...")
            data = parse_summary_file(summary_path, alpha)
            
            # Collect protocol performance data
            for protocol, metrics in data['protocols'].items():
                protocol_data.append({
                    'Alpha': alpha,
                    'Protocol': protocol,
                    'Tumor_Reduction_%': metrics.get('tumor_reduction', None),
                    'Final_Resistance_%': metrics.get('final_resistance', None),
                    'Efficacy_Score': metrics.get('efficacy_score', None),
                    'Immune_Activation': metrics.get('immune_activation', None),
                    'Genetic_Instability': metrics.get('genetic_instability', None),
                    'Metabolic_Shift': metrics.get('metabolic_shift', None)
                })
            
            # Collect patient-specific data
            for patient, metrics in data['patient_specific'].items():
                patient_data.append({
                    'Alpha': alpha,
                    'Patient_Profile': patient,
                    'Best_Protocol': metrics.get('best_protocol', None),
                    'Efficacy_Score': metrics.get('efficacy_score', None),
                    'Tumor_Reduction_%': metrics.get('tumor_reduction', None),
                    'Final_Resistance_%': metrics.get('final_resistance', None)
                })
            
            # Collect ranking data
            for rank_info in data['rankings']:
                ranking_data.append({
                    'Alpha': alpha,
                    'Protocol': rank_info['protocol'],
                    'Average_Score': rank_info['avg_score']
                })
    
    # Create DataFrames
    df_protocols = pd.DataFrame(protocol_data)
    df_patients = pd.DataFrame(patient_data)
    df_rankings = pd.DataFrame(ranking_data)
    
    # Create a comprehensive combined sheet
    # Pivot protocol data for easier analysis
    if not df_protocols.empty:
        # Create efficacy score matrix
        efficacy_pivot = df_protocols.pivot(index='Protocol', columns='Alpha', values='Efficacy_Score')
        efficacy_pivot = efficacy_pivot.reset_index()
        
        # Create tumor reduction matrix
        tumor_pivot = df_protocols.pivot(index='Protocol', columns='Alpha', values='Tumor_Reduction_%')
        tumor_pivot = tumor_pivot.reset_index()
        
        # Create resistance matrix
        resistance_pivot = df_protocols.pivot(index='Protocol', columns='Alpha', values='Final_Resistance_%')
        resistance_pivot = resistance_pivot.reset_index()
    else:
        efficacy_pivot = pd.DataFrame()
        tumor_pivot = pd.DataFrame()
        resistance_pivot = pd.DataFrame()
    
    # Write to Excel with multiple sheets
    excel_file = 'all_alpha_results.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Get workbook to add notes
        workbook = writer.book
        # Sheet 1: All Protocol Data
        df_protocols.to_excel(writer, sheet_name='Protocol_Performance', index=False)
        
        # Sheet 2: Efficacy Scores Matrix
        efficacy_pivot.to_excel(writer, sheet_name='Efficacy_Matrix', index=False)
        
        # Sheet 3: Tumor Reduction Matrix
        tumor_pivot.to_excel(writer, sheet_name='Tumor_Reduction_Matrix', index=False)
        
        # Sheet 4: Resistance Matrix
        resistance_pivot.to_excel(writer, sheet_name='Resistance_Matrix', index=False)
        
        # Sheet 5: Patient-Specific Results
        df_patients.to_excel(writer, sheet_name='Patient_Specific', index=False)
        
        # Sheet 6: Protocol Rankings
        # NOTE: Average_Score = average efficacy across ALL 4 patient profiles (average, young, elderly, compromised)
        # This is different from Protocol_Performance which only shows "average" patient data
        df_rankings.to_excel(writer, sheet_name='Protocol_Rankings', index=False)
        
        # Add note to Protocol_Rankings sheet
        ws_rankings = writer.sheets['Protocol_Rankings']
        ws_rankings.insert_rows(1)
        ws_rankings['A1'] = 'NOTE: Average_Score = average efficacy across ALL 4 patient profiles (average, young, elderly, compromised)'
        ws_rankings['A1'].font = Font(bold=True, italic=True)
        
        # Sheet 7: Summary Statistics
        if not df_protocols.empty:
            summary_stats = df_protocols.groupby('Protocol').agg({
                'Efficacy_Score': ['mean', 'std', 'min', 'max'],
                'Tumor_Reduction_%': ['mean', 'std', 'min', 'max'],
                'Final_Resistance_%': ['mean', 'std', 'min', 'max']
            }).round(2)
            summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns.values]
            summary_stats = summary_stats.reset_index()
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    print(f"\nExcel file created: {excel_file}")
    print(f"  - Protocol_Performance: {len(df_protocols)} rows")
    print(f"  - Efficacy_Matrix: {len(efficacy_pivot)} protocols × {len(alpha_mapping)} alpha values")
    print(f"  - Tumor_Reduction_Matrix: {len(tumor_pivot)} protocols × {len(alpha_mapping)} alpha values")
    print(f"  - Resistance_Matrix: {len(resistance_pivot)} protocols × {len(alpha_mapping)} alpha values")
    print(f"  - Patient_Specific: {len(df_patients)} rows")
    print(f"  - Protocol_Rankings: {len(df_rankings)} rows")
    
    return excel_file

if __name__ == "__main__":
    excel_file = create_excel_file()
    print(f"\n✓ Successfully created {excel_file} with all data!")

