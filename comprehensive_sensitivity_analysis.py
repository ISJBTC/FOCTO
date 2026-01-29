"""
COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS
============================================

This script performs a systematic, scientific sensitivity analysis of all model parameters.
It uses the existing code.py without modification, following proper scientific methodology.

Methodology:
- One-at-a-time (OAT) local sensitivity analysis
- Systematic parameter variation (±10%, ±20%, ±50% from baseline)
- Multiple outcome metrics (tumor reduction, resistance, efficacy)
- Normalized sensitivity coefficients
- Parameter ranking by influence

Author: Generated for manuscript revision
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import from existing code without modification
import sys
sys.path.insert(0, '.')
from code import (
    enhanced_model_params,
    get_enhanced_initial_conditions,
    enhanced_temperature_cancer_model,
    safe_solve_ivp,
    create_patient_profile,
    create_treatment_protocol
)

# ============================================================================
# PARAMETER DEFINITION AND CATEGORIZATION
# ============================================================================

def get_all_parameters():
    """
    Get all parameters from the model and categorize them.
    Returns dictionary with parameter info including:
    - name: parameter name
    - baseline: default value
    - category: biological, treatment, patient, etc.
    - source_type: experimental, literature, hypothetical
    - variation_range: suggested variation range
    """
    
    # Get baseline parameters
    baseline_params = enhanced_model_params()
    
    # Define parameter categories and metadata
    parameter_metadata = {
        # Model structure
        'alpha': {
            'category': 'Model Structure',
            'source_type': 'Theoretical',
            'variation_range': (0.75, 1.0),  # Already analyzed
            'exclude_from_analysis': True  # Already done separately
        },
        'K': {
            'category': 'Biological - Growth',
            'source_type': 'Literature',
            'variation_range': (500, 2000)
        },
        
        # Growth rates
        'lambda1': {
            'category': 'Biological - Growth',
            'source_type': 'Experimental',
            'variation_range': (0.0015, 0.006)
        },
        'lambda2': {
            'category': 'Biological - Growth',
            'source_type': 'Experimental',
            'variation_range': (0.001, 0.004)
        },
        'lambda_R1': {
            'category': 'Biological - Resistance',
            'source_type': 'Literature',
            'variation_range': (0.003, 0.012)
        },
        'lambda_R2': {
            'category': 'Biological - Resistance',
            'source_type': 'Literature',
            'variation_range': (0.0025, 0.01)
        },
        
        # Immune parameters
        'beta1': {
            'category': 'Biological - Immune',
            'source_type': 'Experimental',
            'variation_range': (0.0025, 0.01)
        },
        'beta2': {
            'category': 'Biological - Immune',
            'source_type': 'Literature',
            'variation_range': (0.0005, 0.002)
        },
        'phi1': {
            'category': 'Biological - Immune',
            'source_type': 'Literature',
            'variation_range': (0.05, 0.2)
        },
        'phi2': {
            'category': 'Biological - Immune',
            'source_type': 'Literature',
            'variation_range': (0.0005, 0.002)
        },
        'phi3': {
            'category': 'Biological - Immune',
            'source_type': 'Literature',
            'variation_range': (0.00015, 0.0006)
        },
        'delta_I': {
            'category': 'Biological - Immune',
            'source_type': 'Literature',
            'variation_range': (0.02, 0.08)
        },
        'immune_resist_factor1': {
            'category': 'Biological - Immune',
            'source_type': 'Hypothetical',
            'variation_range': (0.05, 0.2)
        },
        'immune_resist_factor2': {
            'category': 'Biological - Immune',
            'source_type': 'Hypothetical',
            'variation_range': (0.025, 0.1)
        },
        
        # Metastasis
        'gamma': {
            'category': 'Biological - Metastasis',
            'source_type': 'Literature',
            'variation_range': (0.00005, 0.0002)
        },
        'delta_P': {
            'category': 'Biological - Metastasis',
            'source_type': 'Literature',
            'variation_range': (0.005, 0.02)
        },
        
        # Angiogenesis
        'alpha_A': {
            'category': 'Biological - Angiogenesis',
            'source_type': 'Literature',
            'variation_range': (0.005, 0.02)
        },
        'delta_A': {
            'category': 'Biological - Angiogenesis',
            'source_type': 'Literature',
            'variation_range': (0.05, 0.2)
        },
        
        # Quiescence
        'kappa_Q': {
            'category': 'Biological - Quiescence',
            'source_type': 'Literature',
            'variation_range': (0.0005, 0.002)
        },
        'lambda_Q': {
            'category': 'Biological - Quiescence',
            'source_type': 'Literature',
            'variation_range': (0.00025, 0.001)
        },
        
        # Resistance development
        'omega_R1': {
            'category': 'Biological - Resistance',
            'source_type': 'Experimental',
            'variation_range': (0.002, 0.008)
        },
        'omega_R2': {
            'category': 'Biological - Resistance',
            'source_type': 'Experimental',
            'variation_range': (0.0015, 0.006)
        },
        'resistance_floor': {
            'category': 'Biological - Resistance',
            'source_type': 'Hypothetical',
            'variation_range': (0.005, 0.02)
        },
        
        # Senescence
        'kappa_S': {
            'category': 'Biological - Senescence',
            'source_type': 'Literature',
            'variation_range': (0.00025, 0.001)
        },
        'delta_S': {
            'category': 'Biological - Senescence',
            'source_type': 'Literature',
            'variation_range': (0.0025, 0.01)
        },
        
        # Therapy effectiveness
        'etaE': {
            'category': 'Treatment - Efficacy',
            'source_type': 'Clinical',
            'variation_range': (0.005, 0.02)
        },
        'etaH': {
            'category': 'Treatment - Efficacy',
            'source_type': 'Clinical',
            'variation_range': (0.005, 0.02)
        },
        'etaC': {
            'category': 'Treatment - Efficacy',
            'source_type': 'Clinical',
            'variation_range': (0.005, 0.02)
        },
        
        # Treatment resistance effects
        'immuno_resist_boost': {
            'category': 'Treatment - Resistance',
            'source_type': 'Hypothetical',
            'variation_range': (0.25, 1.0)
        },
        'continuous_resist_dev': {
            'category': 'Treatment - Resistance',
            'source_type': 'Hypothetical',
            'variation_range': (1.0, 4.0)
        },
        'adaptive_resist_dev': {
            'category': 'Treatment - Resistance',
            'source_type': 'Hypothetical',
            'variation_range': (0.6, 2.4)
        },
        
        # Pharmacokinetic parameters
        'absorption_rate': {
            'category': 'Pharmacokinetics',
            'source_type': 'Clinical',
            'variation_range': (0.25, 1.0)
        },
        'elimination_rate': {
            'category': 'Pharmacokinetics',
            'source_type': 'Clinical',
            'variation_range': (0.05, 0.2)
        },
        'distribution_vol': {
            'category': 'Pharmacokinetics',
            'source_type': 'Clinical',
            'variation_range': (35, 140)
        },
        'bioavailability': {
            'category': 'Pharmacokinetics',
            'source_type': 'Clinical',
            'variation_range': (0.425, 1.0)
        },
        'max_drug_effect': {
            'category': 'Pharmacokinetics',
            'source_type': 'Clinical',
            'variation_range': (0.5, 2.0)
        },
        'EC50': {
            'category': 'Pharmacokinetics',
            'source_type': 'Clinical',
            'variation_range': (0.15, 0.6)
        },
        'hill_coef': {
            'category': 'Pharmacokinetics',
            'source_type': 'Clinical',
            'variation_range': (0.75, 3.0)
        },
        
        # Circadian rhythm
        'circadian_amplitude': {
            'category': 'Circadian',
            'source_type': 'Experimental',
            'variation_range': (0.1, 0.4)
        },
        'circadian_phase': {
            'category': 'Circadian',
            'source_type': 'Experimental',
            'variation_range': (-12, 12)
        },
        'circadian_period': {
            'category': 'Circadian',
            'source_type': 'Experimental',
            'variation_range': (20, 28)
        },
        
        # Genetic/epigenetic
        'mutation_rate': {
            'category': 'Genetic',
            'source_type': 'Experimental',
            'variation_range': (0.00005, 0.0002)
        },
        'epigenetic_silencing': {
            'category': 'Genetic',
            'source_type': 'Literature',
            'variation_range': (0.001, 0.004)
        },
        'genetic_instability': {
            'category': 'Genetic',
            'source_type': 'Literature',
            'variation_range': (0.5, 2.0)
        },
        
        # Microenvironment
        'hypoxia_threshold': {
            'category': 'Microenvironment',
            'source_type': 'Experimental',
            'variation_range': (0.15, 0.6)
        },
        'acidosis_factor': {
            'category': 'Microenvironment',
            'source_type': 'Literature',
            'variation_range': (0.005, 0.02)
        },
        'metabolic_switch_rate': {
            'category': 'Microenvironment',
            'source_type': 'Experimental',
            'variation_range': (0.01, 0.04)
        },
        'microenv_stress_factor': {
            'category': 'Microenvironment',
            'source_type': 'Hypothetical',
            'variation_range': (0.5, 2.0)
        },
        
        # Treatment scheduling
        'treatment_cycle_period': {
            'category': 'Treatment - Scheduling',
            'source_type': 'Clinical',
            'variation_range': (14, 28)
        },
        'treatment_active_days': {
            'category': 'Treatment - Scheduling',
            'source_type': 'Clinical',
            'variation_range': (3, 14)
        },
        'rest_period_days': {
            'category': 'Treatment - Scheduling',
            'source_type': 'Clinical',
            'variation_range': (7, 21)
        },
        'treatment_intensity': {
            'category': 'Treatment - Scheduling',
            'source_type': 'Clinical',
            'variation_range': (0.5, 2.0)
        },
        
        # Patient-specific (baseline = 1.0, so use multiplicative factors)
        'age_factor': {
            'category': 'Patient - Demographics',
            'source_type': 'Clinical',
            'variation_range': (0.6, 1.4),
            'is_factor': True
        },
        'performance_status': {
            'category': 'Patient - Demographics',
            'source_type': 'Clinical',
            'variation_range': (0.6, 1.4),
            'is_factor': True
        },
        'bmi_factor': {
            'category': 'Patient - Demographics',
            'source_type': 'Clinical',
            'variation_range': (0.8, 1.2),
            'is_factor': True
        },
        'prior_treatment_factor': {
            'category': 'Patient - History',
            'source_type': 'Clinical',
            'variation_range': (0.7, 1.5),
            'is_factor': True
        },
        'liver_function': {
            'category': 'Patient - Organ Function',
            'source_type': 'Clinical',
            'variation_range': (0.5, 1.5),
            'is_factor': True
        },
        'kidney_function': {
            'category': 'Patient - Organ Function',
            'source_type': 'Clinical',
            'variation_range': (0.5, 1.5),
            'is_factor': True
        },
        'immune_status': {
            'category': 'Patient - Immune',
            'source_type': 'Clinical',
            'variation_range': (0.5, 1.5),
            'is_factor': True
        }
    }
    
    # Build complete parameter list
    parameters = []
    for param_name, baseline_value in baseline_params.items():
        if param_name in ['uE', 'uH', 'uC', 'uI']:  # Skip control variables
            continue
            
        if param_name in parameter_metadata:
            meta = parameter_metadata[param_name]
            if meta.get('exclude_from_analysis', False):
                continue
            parameters.append({
                'name': param_name,
                'baseline': baseline_value,
                'category': meta['category'],
                'source_type': meta['source_type'],
                'variation_range': meta['variation_range'],
                'is_factor': meta.get('is_factor', False)
            })
        else:
            # Default metadata for parameters not explicitly defined
            parameters.append({
                'name': param_name,
                'baseline': baseline_value,
                'category': 'Uncategorized',
                'source_type': 'Unknown',
                'variation_range': (baseline_value * 0.5, baseline_value * 1.5),
                'is_factor': False
            })
    
    return parameters

# ============================================================================
# SENSITIVITY ANALYSIS METHODS
# ============================================================================

def calculate_outcome_metrics(simulation_result, initial_conditions):
    """
    Calculate key outcome metrics from simulation results.
    
    Returns:
        dict: Dictionary with outcome metrics
    """
    if not simulation_result.success:
        return {
            'tumor_reduction': None,
            'final_resistance': None,
            'efficacy_score': None,
            'final_burden': None,
            'initial_burden': None,
            'success': False
        }
    
    # Extract cell populations
    N1 = simulation_result.y[0]  # Sensitive cells
    N2 = simulation_result.y[1]  # Partially resistant cells
    Q = simulation_result.y[6]   # Quiescent cells
    R1 = simulation_result.y[7]  # Type 1 resistant cells
    R2 = simulation_result.y[8]  # Type 2 resistant cells
    S = simulation_result.y[9]    # Senescent cells
    
    total_tumor = N1 + N2 + Q + R1 + R2 + S
    total_resistant = R1 + R2
    
    # Calculate metrics
    initial_burden = total_tumor[0]
    final_burden = total_tumor[-1]
    tumor_reduction = 100 * (1 - final_burden / initial_burden) if initial_burden > 0 else 0
    
    resistance_fraction = (total_resistant[-1] / total_tumor[-1] * 100) if total_tumor[-1] > 0 else 0
    
    # Efficacy score (tumor reduction penalized by resistance)
    efficacy_score = tumor_reduction / (1 + resistance_fraction/100)
    
    return {
        'tumor_reduction': tumor_reduction,
        'final_resistance': resistance_fraction,
        'efficacy_score': efficacy_score,
        'final_burden': final_burden,
        'initial_burden': initial_burden,
        'success': True
    }

def run_single_simulation(params, patient_profile_name='average', protocol_name='standard', simulation_days=500):
    """
    Run a single simulation with given parameters.
    
    Args:
        params: Parameter dictionary
        patient_profile_name: Patient profile to use
        protocol_name: Treatment protocol to use
        simulation_days: Number of days to simulate
        
    Returns:
        dict: Outcome metrics
    """
    try:
        # Get patient profile
        patient_profile = create_patient_profile(patient_profile_name)
        
        # Update params with patient profile
        if patient_profile:
            for key, value in patient_profile.items():
                if key in params:
                    params[key] = value
        
        # Get treatment protocol
        protocol = create_treatment_protocol(protocol_name)
        
        # Set up simulation
        t_span = [0, simulation_days]
        t_eval = np.linspace(0, simulation_days, simulation_days + 1)
        initial_conditions = get_enhanced_initial_conditions()
        
        # Set up drug schedules
        drug_schedules = {}
        if 'hormone' in protocol and protocol['hormone']:
            drug_schedules['hormone'] = protocol['hormone']
        if 'her2' in protocol and protocol['her2']:
            drug_schedules['her2'] = protocol['her2']
        if 'chemo' in protocol and protocol['chemo']:
            drug_schedules['chemo'] = protocol['chemo']
        if 'immuno' in protocol and protocol['immuno']:
            drug_schedules['immuno'] = protocol['immuno']
        
        # Temperature function
        temp_func = protocol.get('temperature', lambda t: 37.0)
        
        # Define model function
        def model_func(t, y):
            current_temp = temp_func(t)
            return enhanced_temperature_cancer_model(t, y, params, drug_schedules, current_temp, use_circadian=True)
        
        # Run simulation
        result = safe_solve_ivp(model_func, t_span, initial_conditions, 'RK45', t_eval)
        
        # Calculate outcomes
        outcomes = calculate_outcome_metrics(result, initial_conditions)
        return outcomes
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        return {
            'tumor_reduction': None,
            'final_resistance': None,
            'efficacy_score': None,
            'final_burden': None,
            'initial_burden': None,
            'success': False
        }

def calculate_sensitivity_coefficient(baseline_outcome, perturbed_outcome, baseline_param, perturbed_param):
    """
    Calculate normalized sensitivity coefficient.
    
    S = (ΔOutcome/Outcome) / (ΔParameter/Parameter)
    
    Returns:
        float: Sensitivity coefficient
    """
    if baseline_outcome is None or perturbed_outcome is None:
        return None
    
    if baseline_outcome == 0 or baseline_param == 0:
        return None
    
    delta_outcome = perturbed_outcome - baseline_outcome
    delta_param = perturbed_param - baseline_param
    
    if delta_param == 0:
        return 0.0
    
    # Normalized sensitivity
    sensitivity = (delta_outcome / baseline_outcome) / (delta_param / baseline_param)
    
    return sensitivity

def perform_parameter_sensitivity_analysis(
    parameter_name,
    baseline_value,
    variation_levels=[-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5],
    patient_profile='average',
    protocol='standard',
    simulation_days=500
):
    """
    Perform sensitivity analysis for a single parameter.
    
    Args:
        parameter_name: Name of parameter to vary
        baseline_value: Baseline value of parameter
        variation_levels: List of fractional variations (e.g., -0.2 = 20% decrease)
        patient_profile: Patient profile to use
        protocol: Treatment protocol to use
        
    Returns:
        dict: Sensitivity analysis results
    """
    print(f"  Analyzing parameter: {parameter_name}")
    
    # Get baseline parameters
    baseline_params = enhanced_model_params()
    
    # Run baseline simulation
    baseline_outcomes = run_single_simulation(
        baseline_params.copy(),
        patient_profile,
        protocol,
        simulation_days
    )
    
    if not baseline_outcomes['success']:
        print(f"    Warning: Baseline simulation failed for {parameter_name}")
        return None
    
    # Store results
    results = {
        'parameter_name': parameter_name,
        'baseline_value': baseline_value,
        'baseline_outcomes': baseline_outcomes,
        'variations': []
    }
    
    # Test each variation level
    for variation in variation_levels:
        if variation == 0.0:
            # Skip baseline (already done)
            continue
        
        # Calculate perturbed parameter value
        if isinstance(baseline_value, (int, float)) and baseline_value != 0:
            perturbed_value = baseline_value * (1 + variation)
        else:
            # For parameters that might be zero or non-numeric
            perturbed_value = baseline_value + variation * abs(baseline_value) if baseline_value != 0 else variation
        
        # Create perturbed parameter set
        perturbed_params = baseline_params.copy()
        perturbed_params[parameter_name] = perturbed_value
        
        # Run simulation with perturbed parameter
        perturbed_outcomes = run_single_simulation(
            perturbed_params,
            patient_profile,
            protocol,
            simulation_days
        )
        
        if perturbed_outcomes['success']:
            # Calculate sensitivity coefficients for each outcome
            sensitivities = {}
            for outcome_name in ['tumor_reduction', 'final_resistance', 'efficacy_score']:
                baseline_outcome = baseline_outcomes[outcome_name]
                perturbed_outcome = perturbed_outcomes[outcome_name]
                
                sensitivity = calculate_sensitivity_coefficient(
                    baseline_outcome,
                    perturbed_outcome,
                    baseline_value,
                    perturbed_value
                )
                sensitivities[outcome_name] = sensitivity
            
            results['variations'].append({
                'variation_level': variation,
                'parameter_value': perturbed_value,
                'outcomes': perturbed_outcomes,
                'sensitivities': sensitivities
            })
    
    # Calculate summary statistics
    if results['variations']:
        # Average absolute sensitivity across all variations
        avg_sensitivities = {}
        for outcome_name in ['tumor_reduction', 'final_resistance', 'efficacy_score']:
            sens_values = [
                abs(v['sensitivities'][outcome_name])
                for v in results['variations']
                if v['sensitivities'][outcome_name] is not None
            ]
            if sens_values:
                avg_sensitivities[outcome_name] = np.mean(sens_values)
            else:
                avg_sensitivities[outcome_name] = 0.0
        
        results['average_sensitivity'] = avg_sensitivities
        results['max_sensitivity'] = max(avg_sensitivities.values()) if avg_sensitivities else 0.0
    else:
        results['average_sensitivity'] = {}
        results['max_sensitivity'] = 0.0
    
    return results

# ============================================================================
# MAIN SENSITIVITY ANALYSIS EXECUTION
# ============================================================================

def run_comprehensive_sensitivity_analysis(
    output_dir='sensitivity_analysis_results',
    patient_profiles=['average'],
    protocols=['standard'],
    variation_levels=[-0.2, -0.1, 0.1, 0.2],  # ±10% and ±20%
    simulation_days=500
):
    """
    Run comprehensive sensitivity analysis for all parameters.
    
    Args:
        output_dir: Directory to save results
        patient_profiles: List of patient profiles to analyze
        protocols: List of protocols to analyze
        variation_levels: List of fractional variations to test
        simulation_days: Number of days to simulate
        
    Returns:
        dict: Complete sensitivity analysis results
    """
    print("="*80)
    print("COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all parameters
    parameters = get_all_parameters()
    print(f"Total parameters to analyze: {len(parameters)}")
    print(f"Patient profiles: {patient_profiles}")
    print(f"Protocols: {protocols}")
    print(f"Variation levels: {variation_levels}")
    print()
    
    # Store all results
    all_results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'total_parameters': len(parameters),
            'patient_profiles': patient_profiles,
            'protocols': protocols,
            'variation_levels': variation_levels,
            'simulation_days': simulation_days
        },
        'parameter_results': []
    }
    
    # Run sensitivity analysis for each parameter
    total_simulations = len(parameters) * len(patient_profiles) * len(protocols) * len(variation_levels)
    current_simulation = 0
    
    print("Starting parameter sensitivity analysis...")
    print("-"*80)
    
    for param_idx, param_info in enumerate(parameters, 1):
        param_name = param_info['name']
        baseline_value = param_info['baseline']
        
        print(f"\n[{param_idx}/{len(parameters)}] Parameter: {param_name}")
        print(f"  Category: {param_info['category']}")
        print(f"  Source: {param_info['source_type']}")
        print(f"  Baseline: {baseline_value}")
        
        param_results = {
            'parameter_info': param_info,
            'sensitivity_results': {}
        }
        
        # Analyze for each patient profile and protocol combination
        for patient in patient_profiles:
            for protocol in protocols:
                print(f"  Testing: {patient} patient, {protocol} protocol")
                
                # Perform sensitivity analysis
                sensitivity_result = perform_parameter_sensitivity_analysis(
                    param_name,
                    baseline_value,
                    variation_levels,
                    patient,
                    protocol,
                    simulation_days
                )
                
                if sensitivity_result:
                    key = f"{patient}_{protocol}"
                    param_results['sensitivity_results'][key] = sensitivity_result
                    current_simulation += len(variation_levels)
                    print(f"    Completed: {current_simulation}/{total_simulations} simulations")
                else:
                    print(f"    Failed: Could not complete sensitivity analysis")
        
        all_results['parameter_results'].append(param_results)
        
        # Save intermediate results periodically
        if param_idx % 10 == 0:
            intermediate_file = os.path.join(output_dir, f'intermediate_results_{param_idx}.json')
            with open(intermediate_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"  Intermediate results saved to {intermediate_file}")
    
    # Save final results
    final_file = os.path.join(output_dir, 'complete_sensitivity_results.json')
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {final_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results

# ============================================================================
# RESULTS PROCESSING AND VISUALIZATION
# ============================================================================

def process_sensitivity_results(results, output_dir='sensitivity_analysis_results'):
    """
    Process sensitivity analysis results and generate summary tables and visualizations.
    Generates both aggregated and context-specific results.
    """
    print("\nProcessing sensitivity analysis results...")
    
    # ========================================================================
    # 1. AGGREGATED SUMMARY (across all contexts)
    # ========================================================================
    print("  Creating aggregated summary...")
    summary_data = []
    
    for param_result in results['parameter_results']:
        param_info = param_result['parameter_info']
        param_name = param_info['name']
        
        # Get maximum sensitivity across all patient/protocol combinations
        max_sensitivities = []
        for key, sens_result in param_result['sensitivity_results'].items():
            if sens_result and 'max_sensitivity' in sens_result:
                max_sensitivities.append(sens_result['max_sensitivity'])
        
        if max_sensitivities:
            avg_max_sensitivity = np.mean(max_sensitivities)
            overall_max_sensitivity = max(max_sensitivities)
        else:
            avg_max_sensitivity = 0.0
            overall_max_sensitivity = 0.0
        
        # Calculate context-specific statistics
        context_sensitivities = {}
        for key, sens_result in param_result['sensitivity_results'].items():
            if sens_result and 'max_sensitivity' in sens_result:
                context_sensitivities[key] = sens_result['max_sensitivity']
        
        # Calculate standard deviation across contexts (shows context-dependency)
        context_std = np.std(list(context_sensitivities.values())) if context_sensitivities else 0.0
        
        summary_data.append({
            'Parameter': param_name,
            'Category': param_info['category'],
            'Source_Type': param_info['source_type'],
            'Baseline_Value': param_info['baseline'],
            'Avg_Max_Sensitivity': avg_max_sensitivity,
            'Overall_Max_Sensitivity': overall_max_sensitivity,
            'Context_Std': context_std,  # Higher = more context-dependent
            'Num_Contexts': len(context_sensitivities)
        })
    
    # Create aggregated DataFrame
    df_aggregated = pd.DataFrame(summary_data)
    df_aggregated = df_aggregated.sort_values('Overall_Max_Sensitivity', ascending=False)
    
    # Save aggregated summary table
    summary_file = os.path.join(output_dir, 'sensitivity_summary_table.csv')
    df_aggregated.to_csv(summary_file, index=False)
    print(f"    ✓ Saved: {summary_file}")
    
    # ========================================================================
    # 2. CONTEXT-SPECIFIC DATA (all patient × protocol combinations)
    # ========================================================================
    print("  Creating context-specific data...")
    context_data = []
    
    for param_result in results['parameter_results']:
        param_name = param_result['parameter_info']['name']
        param_category = param_result['parameter_info']['category']
        param_source = param_result['parameter_info']['source_type']
        
        for context_key, sens_result in param_result['sensitivity_results'].items():
            if sens_result and 'max_sensitivity' in sens_result:
                patient, protocol = context_key.split('_', 1)
                
                # Get average sensitivity across all outcome metrics
                avg_sens = sens_result.get('average_sensitivity', {})
                if isinstance(avg_sens, dict):
                    efficacy_sens = avg_sens.get('efficacy_score', 0)
                else:
                    efficacy_sens = 0
                
                context_data.append({
                    'Parameter': param_name,
                    'Category': param_category,
                    'Source_Type': param_source,
                    'Patient_Profile': patient,
                    'Protocol': protocol,
                    'Max_Sensitivity': sens_result['max_sensitivity'],
                    'Efficacy_Sensitivity': efficacy_sens
                })
    
    # Create context-specific DataFrame
    df_contexts = pd.DataFrame(context_data)
    
    # Save context-specific table
    context_file = os.path.join(output_dir, 'sensitivity_by_context.csv')
    df_contexts.to_csv(context_file, index=False)
    print(f"    ✓ Saved: {context_file} ({len(df_contexts)} rows)")
    
    # ========================================================================
    # 3. PIVOT TABLES
    # ========================================================================
    print("  Creating pivot tables...")
    
    # Pivot: Parameter × Context (Patient_Protocol)
    pivot_table = df_contexts.pivot_table(
        index='Parameter',
        columns=['Patient_Profile', 'Protocol'],
        values='Max_Sensitivity',
        aggfunc='mean'
    )
    pivot_file = os.path.join(output_dir, 'sensitivity_pivot_table.csv')
    pivot_table.to_csv(pivot_file)
    print(f"    ✓ Saved: {pivot_file}")
    
    # Pivot: Parameter × Patient Profile
    patient_summary = df_contexts.groupby(['Patient_Profile', 'Parameter'])['Max_Sensitivity'].mean().reset_index()
    patient_pivot = patient_summary.pivot(index='Parameter', columns='Patient_Profile', values='Max_Sensitivity')
    patient_file = os.path.join(output_dir, 'sensitivity_by_patient_profile.csv')
    patient_pivot.to_csv(patient_file)
    print(f"    ✓ Saved: {patient_file}")
    
    # Pivot: Parameter × Protocol
    protocol_summary = df_contexts.groupby(['Protocol', 'Parameter'])['Max_Sensitivity'].mean().reset_index()
    protocol_pivot = protocol_summary.pivot(index='Parameter', columns='Protocol', values='Max_Sensitivity')
    protocol_file = os.path.join(output_dir, 'sensitivity_by_protocol.csv')
    protocol_pivot.to_csv(protocol_file)
    print(f"    ✓ Saved: {protocol_file}")
    
    # ========================================================================
    # 4. VISUALIZATIONS (aggregated and context-specific)
    # ========================================================================
    print("  Creating visualizations...")
    create_sensitivity_visualizations(df_aggregated, df_contexts, output_dir)
    
    return df_aggregated, df_contexts

def create_sensitivity_visualizations(df_aggregated, df_contexts, output_dir):
    """
    Create comprehensive visualizations of sensitivity analysis results.
    Includes both aggregated and context-specific visualizations.
    """
    # Set plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # ========================================================================
    # AGGREGATED VISUALIZATIONS
    # ========================================================================
    
    # Figure 1: Parameter Sensitivity Ranking (Tornado Plot) - AGGREGATED
    fig, ax = plt.subplots(figsize=(14, max(10, len(df_aggregated) * 0.3)))
    
    top_n = min(30, len(df_aggregated))  # Top 30 most sensitive parameters
    df_top = df_aggregated.head(top_n)
    
    y_pos = np.arange(len(df_top))
    colors_list = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_top)))
    
    bars = ax.barh(y_pos, df_top['Overall_Max_Sensitivity'], color=colors_list, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top['Parameter'], fontsize=10)
    ax.set_xlabel('Maximum Sensitivity Coefficient', fontsize=14, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Sensitive Parameters (Aggregated Across All Contexts)\n(Higher = More Influential)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_ranking_tornado.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sensitivity_ranking_tornado.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Created: sensitivity_ranking_tornado.png/pdf")
    
    # Figure 2: Sensitivity by Category - AGGREGATED
    fig, ax = plt.subplots(figsize=(12, 8))
    
    category_sensitivity = df_aggregated.groupby('Category')['Overall_Max_Sensitivity'].mean().sort_values(ascending=False)
    
    bars = ax.bar(range(len(category_sensitivity)), category_sensitivity.values, alpha=0.8)
    ax.set_xticks(range(len(category_sensitivity)))
    ax.set_xticklabels(category_sensitivity.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Average Maximum Sensitivity', fontsize=12, fontweight='bold')
    ax.set_title('Average Parameter Sensitivity by Category (Aggregated)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_by_category.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sensitivity_by_category.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Created: sensitivity_by_category.png/pdf")
    
    # Figure 3: Sensitivity by Source Type - AGGREGATED
    fig, ax = plt.subplots(figsize=(10, 6))
    
    source_sensitivity = df_aggregated.groupby('Source_Type')['Overall_Max_Sensitivity'].mean().sort_values(ascending=False)
    
    bars = ax.bar(range(len(source_sensitivity)), source_sensitivity.values, alpha=0.8, color='steelblue')
    ax.set_xticks(range(len(source_sensitivity)))
    ax.set_xticklabels(source_sensitivity.index, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Average Maximum Sensitivity', fontsize=12, fontweight='bold')
    ax.set_title('Average Parameter Sensitivity by Source Type (Aggregated)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_by_source_type.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sensitivity_by_source_type.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Created: sensitivity_by_source_type.png/pdf")
    
    # ========================================================================
    # CONTEXT-SPECIFIC VISUALIZATIONS
    # ========================================================================
    
    # Figure 4: Heatmap - Sensitivity by Patient Profile
    fig, ax = plt.subplots(figsize=(14, max(10, len(df_aggregated) * 0.3)))
    
    # Get top 20 parameters for readability
    top_params = df_aggregated.head(20)['Parameter'].values
    df_heatmap_patient = df_contexts[df_contexts['Parameter'].isin(top_params)]
    heatmap_data_patient = df_heatmap_patient.pivot_table(
        index='Parameter',
        columns='Patient_Profile',
        values='Max_Sensitivity',
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data_patient, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Sensitivity Coefficient'}, ax=ax, linewidths=0.5)
    ax.set_title('Top 20 Parameter Sensitivity by Patient Profile', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Patient Profile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_heatmap_by_patient.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sensitivity_heatmap_by_patient.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Created: sensitivity_heatmap_by_patient.png/pdf")
    
    # Figure 5: Heatmap - Sensitivity by Protocol
    fig, ax = plt.subplots(figsize=(14, max(10, len(df_aggregated) * 0.3)))
    
    heatmap_data_protocol = df_heatmap_patient.pivot_table(
        index='Parameter',
        columns='Protocol',
        values='Max_Sensitivity',
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data_protocol, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Sensitivity Coefficient'}, ax=ax, linewidths=0.5)
    ax.set_title('Top 20 Parameter Sensitivity by Treatment Protocol', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Treatment Protocol', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_heatmap_by_protocol.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sensitivity_heatmap_by_protocol.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Created: sensitivity_heatmap_by_protocol.png/pdf")
    
    # Figure 6: Context Dependency (Box Plot)
    fig, ax = plt.subplots(figsize=(14, max(8, len(df_aggregated.head(20)) * 0.3)))
    
    top_params = df_aggregated.head(20)['Parameter'].values
    df_box = df_contexts[df_contexts['Parameter'].isin(top_params)]
    
    sns.boxplot(data=df_box, x='Max_Sensitivity', y='Parameter', ax=ax, orient='h')
    ax.set_xlabel('Sensitivity Coefficient (Distribution Across Contexts)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Parameters: Sensitivity Distribution Across All Contexts\n(Box shows variability = context-dependency)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensitivity_context_dependency.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'sensitivity_context_dependency.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Created: sensitivity_context_dependency.png/pdf")
    
    print("    All visualizations created successfully!")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Starting Comprehensive Sensitivity Analysis...")
    print("This will analyze ALL model parameters across ALL patient profiles and protocols.")
    print("This provides scientifically rigorous, context-independent sensitivity analysis.")
    print("Estimated time: 1-3 hours depending on system speed.")
    print()
    
    # Run comprehensive analysis
    # Test ALL patient profiles and ALL protocols for robust, scientifically rigorous analysis
    results = run_comprehensive_sensitivity_analysis(
        output_dir='sensitivity_analysis_results',
        patient_profiles=['average', 'young', 'elderly', 'compromised'],  # All patient profiles
        protocols=['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia'],  # All protocols
        variation_levels=[-0.2, -0.1, 0.1, 0.2],  # ±10% and ±20%
        simulation_days=500
    )
    
    # Process and visualize results (generates all CSVs and figures)
    df_summary, df_contexts = process_sensitivity_results(results, 'sensitivity_analysis_results')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nTop 10 Most Sensitive Parameters (across all contexts):")
    print(df_summary.head(10)[['Parameter', 'Category', 'Source_Type', 'Overall_Max_Sensitivity', 'Context_Std']].to_string(index=False))
    print(f"\nNote: Context_Std shows context-dependency (higher = more variable across patient/protocol combinations)")
    print(f"\nTotal contexts analyzed: {len(df_contexts['Patient_Profile'].unique())} patients × {len(df_contexts['Protocol'].unique())} protocols = {len(df_contexts)} parameter-context combinations")
    print("\n" + "="*80)
    print("ALL RESULTS GENERATED IN 'sensitivity_analysis_results' FOLDER:")
    print("="*80)
    print("\nCSV FILES:")
    print("  ✓ sensitivity_summary_table.csv (aggregated summary)")
    print("  ✓ sensitivity_by_context.csv (all 20 contexts)")
    print("  ✓ sensitivity_pivot_table.csv (parameter × context matrix)")
    print("  ✓ sensitivity_by_patient_profile.csv (parameter × patient)")
    print("  ✓ sensitivity_by_protocol.csv (parameter × protocol)")
    print("\nFIGURES (Aggregated):")
    print("  ✓ sensitivity_ranking_tornado.png/pdf")
    print("  ✓ sensitivity_by_category.png/pdf")
    print("  ✓ sensitivity_by_source_type.png/pdf")
    print("\nFIGURES (Context-Specific):")
    print("  ✓ sensitivity_heatmap_by_patient.png/pdf")
    print("  ✓ sensitivity_heatmap_by_protocol.png/pdf")
    print("  ✓ sensitivity_context_dependency.png/pdf")
    print("\nDATA:")
    print("  ✓ complete_sensitivity_results.json (complete raw data)")
    print("="*80)

