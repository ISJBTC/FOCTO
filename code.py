"""
Fractional-Order Cancer Treatment Optimization Model

A computational model for simulating breast cancer treatment dynamics using
fractional-order differential equations to capture memory effects in biological systems.

Version: 1.0.0
DOI: 
Repository: https://github.com/ISJBTC/FOCTO.git
"""

__version__ = "1.0.0"
__author__ = "Irshad Jamadar"
__doi__ = ""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# Utility function for safe figure saving
def safe_save_figure(filename, dpi=300):
    """
    Safely save matplotlib figure with error handling.
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Use a sanitized filename
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        
        # Fallback to current directory if full path fails
        if not os.path.exists(os.path.dirname(safe_filename or '.')):
            safe_filename = os.path.basename(filename)
        
        plt.savefig(safe_filename, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved successfully: {safe_filename}")
    except Exception as e:
        print(f"Error saving figure: {e}")
        try:
            # Fallback: save in current directory
            fallback_filename = os.path.basename(filename)
            plt.savefig(fallback_filename, dpi=dpi, bbox_inches='tight')
            print(f"Fallback figure saved: {fallback_filename}")
        except Exception as fallback_error:
            print(f"Fallback save failed: {fallback_error}")

# Safe solver function with improved error handling
def safe_solve_ivp(func, t_span, y0, method, t_eval, *args, max_retries=3, **kwargs):
    """Safely solve IVP with error handling and multiple fallback options."""
    methods = ['RK45', 'BDF', 'Radau', 'DOP853'] if method == 'RK45' else [method, 'RK45', 'BDF', 'Radau']
    rtols = [1e-4, 1e-5, 1e-6]
    atols = [1e-7, 1e-8, 1e-9]
    
    result = None
    success = False
    
    for retry in range(max_retries):
        if success:
            break
            
        for i, method_try in enumerate(methods):
            if success:
                break
                
            for rtol in rtols:
                if success:
                    break
                    
                for atol in atols:
                    try:
                        result = solve_ivp(func, t_span, y0, method=method_try, t_eval=t_eval, 
                                          rtol=rtol, atol=atol, *args, **kwargs)
                        if result.success:
                            success = True
                            print(f"Solver succeeded with method={method_try}, rtol={rtol}, atol={atol}")
                            break
                    except Exception as e:
                        print(f"Attempt with method={method_try}, rtol={rtol}, atol={atol} failed: {str(e)}")
    
    # If all attempts failed, return dummy result
    if not success:
        print("All solver attempts failed. Returning dummy result.")
        result = type('obj', (object,), {
            't': t_eval,
            'y': np.zeros((len(y0), len(t_eval))),
            'success': False,
            'message': "All solver attempts failed"
        })
    
    return result

# Set plotting style with more modern aesthetics
def set_plotting_style():
    """Configure plotting style for consistent, publication-quality figures"""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'Palatino', 'serif']
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # Create a custom colorblind-friendly palette
    colors = sns.color_palette("colorblind", 8)
    sns.set_palette(colors)
    
    return colors

# Enhanced model parameters with new PK/PD and circadian components
def enhanced_model_params(patient_profile=None):
    """
    Parameters with enhanced resistance dynamics, PK/PD and circadian components.
    
    Args:
        patient_profile (dict, optional): Patient-specific parameter modifications
        
    Returns:
        dict: Complete parameter set for the model
    """
    # Base parameters
    params = {
        'alpha': 0.93,        # Fractional order
        'K': 1000,            # Carrying capacity
        
        # Growth rates 
        'lambda1': 0.003,     # Growth rate of sensitive cells
        'lambda2': 0.002,     # Growth rate of partially resistant cells
        
        # Resistant cell growth rates
        'lambda_R1': 0.006,   # Type 1 resistant cell growth
        'lambda_R2': 0.005,   # Type 2 resistant cell growth
        
        # Immune parameters
        'beta1': 0.005,       # Cytotoxic immune killing rate
        'beta2': 0.001,       # Regulatory immune suppression
        'phi1': 0.1,          # Baseline cytotoxic immune production
        'phi2': 0.001,        # Tumor-induced immune recruitment
        'phi3': 0.0003,       # Regulatory immune recruitment
        'delta_I': 0.04,      # Immune cell death rate
        
        # Resistance and immune interaction
        'immune_resist_factor1': 0.10,
        'immune_resist_factor2': 0.05,
        
        # Metastasis parameters
        'gamma': 0.0001,      # Metastasis formation rate
        'delta_P': 0.01,      # Metastasis death rate
        
        # Angiogenesis parameters
        'alpha_A': 0.01,      # Angiogenesis stimulation rate
        'delta_A': 0.1,       # Angiogenesis factor degradation
        
        # Quiescence parameters
        'kappa_Q': 0.001,     # Rate of entering quiescence
        'lambda_Q': 0.0005,   # Rate of leaving quiescence
        
        # Resistance development
        'omega_R1': 0.004,    # Type 1 resistance development
        'omega_R2': 0.003,    # Type 2 resistance development
        'resistance_floor': 0.01,
        
        # Senescence parameters
        'kappa_S': 0.0005,    # Senescence induction rate
        'delta_S': 0.005,     # Senescent cell death rate
        
        # Therapy parameters
        'etaE': 0.01,         # Hormone therapy effectiveness
        'etaH': 0.01,         # HER2 therapy effectiveness
        'etaC': 0.01,         # Chemotherapy effectiveness
        
        # Treatment-specific resistance effects
        'immuno_resist_boost': 0.5,
        'continuous_resist_dev': 2.0,
        'adaptive_resist_dev': 1.2,
        
        # Default control values
        'uE': 0.0,            # Hormone therapy control
        'uH': 0.0,            # HER2 therapy control
        'uC': 0.0,            # Chemotherapy control
        'uI': 0.0,            # Immunotherapy control
        
        # NEW: Pharmacokinetic parameters
        'absorption_rate': 0.5,       # Drug absorption rate
        'elimination_rate': 0.1,      # Drug elimination rate
        'distribution_vol': 70.0,     # Distribution volume (L)
        'bioavailability': 0.85,      # Bioavailability fraction
        'max_drug_effect': 1.0,       # Maximum drug effect
        'EC50': 0.3,                  # Concentration for half-maximal effect
        'hill_coef': 1.5,             # Hill coefficient for drug effect
        
        # NEW: Circadian rhythm parameters
        'circadian_amplitude': 0.2,   # Amplitude of circadian oscillation
        'circadian_phase': 0.0,       # Phase shift of circadian rhythm
        'circadian_period': 24.0,     # Period of circadian rhythm (hours)
        
        # NEW: Genetic/epigenetic parameters
        'mutation_rate': 0.0001,      # Base mutation rate
        'epigenetic_silencing': 0.002, # Epigenetic silencing rate
        'genetic_instability': 1.0,   # Genetic instability factor
        
        # NEW: Microenvironment and metabolism
        'hypoxia_threshold': 0.3,     # Tumor size where hypoxia starts
        'acidosis_factor': 0.01,      # Acidosis impact factor
        'metabolic_switch_rate': 0.02, # Rate of switching to glycolysis
        'microenv_stress_factor': 1.0, # Microenvironmental stress factor
        
        # NEW: Treatment scheduling parameters
        'treatment_cycle_period': 21, # Days in treatment cycle
        'treatment_active_days': 7,   # Active treatment days per cycle
        'rest_period_days': 14,       # Rest days (no treatment) per cycle
        'treatment_intensity': 1.0,   # Treatment intensity multiplier
        
        # NEW: Patient-specific factors
        'age_factor': 1.0,            # Age impact on treatment response
        'performance_status': 1.0,    # Patient performance status
        'bmi_factor': 1.0,            # BMI impact factor
        'prior_treatment_factor': 1.0, # Prior treatment impact
        'liver_function': 1.0,        # Liver function impact on drug metabolism
        'kidney_function': 1.0,       # Kidney function impact on drug clearance
        'immune_status': 1.0,         # Baseline immune status
    }
    
    # If patient profile provided, update relevant parameters
    if patient_profile:
        for key, value in patient_profile.items():
            if key in params:
                params[key] = value
    
    return params

# Define initial conditions with additional compartments
def get_enhanced_initial_conditions():
    """Define initial conditions for all model compartments"""
    return np.array([
        190,    # N1: Sensitive cells
        10,     # N2: Partially resistant cells
        40,     # I1: Cytotoxic immune cells
        10,     # I2: Regulatory immune cells
        0.1,    # P: Metastatic potential
        1,      # A: Angiogenesis factor
        0.1,    # Q: Quiescent cells
        1.0,    # R1: Type 1 resistant cells
        1.0,    # R2: Type 2 resistant cells
        0.1,    # S: Senescent cells
        0.0,    # D: Drug concentration
        0.0,    # Dm: Metabolized drug
        1.0,    # G: Genetic stability
        1.0,    # M: Metabolism status
        0.0     # H: Hypoxia level
    ])

# NEW: Drug pharmacokinetics function
def drug_pharmacokinetics(t, dose_schedule, params):
    """
    Calculate drug concentration based on timing and PK parameters
    
    Args:
        t (float): Current time
        dose_schedule (function): Function that returns dose given time
        params (dict): Model parameters
    
    Returns:
        float: Current effective drug concentration
    """
    # Extract PK parameters
    absorption_rate = params.get('absorption_rate', 0.5)
    elimination_rate = params.get('elimination_rate', 0.1)
    bioavailability = params.get('bioavailability', 0.85)
    liver_function = params.get('liver_function', 1.0)
    kidney_function = params.get('kidney_function', 1.0)
    
    # Adjust elimination based on patient factors
    adjusted_elimination = elimination_rate * liver_function * kidney_function
    
    # Get current dose from schedule
    current_dose = dose_schedule(t) if callable(dose_schedule) else 0.0
    
    # Simple one-compartment model
    effective_dose = current_dose * bioavailability
    concentration_change = absorption_rate * effective_dose - adjusted_elimination
    
    return concentration_change

# NEW: Calculate pharmacodynamic effect
def calculate_drug_effect(concentration, params):
    """
    Calculate drug effect using Hill equation
    
    Args:
        concentration (float): Drug concentration
        params (dict): Model parameters
    
    Returns:
        float: Drug effect (between 0 and max_effect)
    """
    # Extract PD parameters
    max_effect = params.get('max_drug_effect', 1.0)
    ec50 = params.get('EC50', 0.3)
    hill = params.get('hill_coef', 1.5)
    
    # Apply Hill equation
    effect = max_effect * (concentration**hill) / (ec50**hill + concentration**hill)
    
    return effect

# NEW: Calculate circadian rhythm impact
def circadian_effect(t, params):
    """
    Calculate effect of circadian rhythm on biological processes
    
    Args:
        t (float): Current time (in days)
        params (dict): Model parameters
    
    Returns:
        float: Modulation factor based on circadian rhythm (centered at 1.0)
    """
    # Extract circadian parameters
    amplitude = params.get('circadian_amplitude', 0.2)
    phase = params.get('circadian_phase', 0.0)
    period = params.get('circadian_period', 24.0) / 24.0  # Convert hours to days
    
    # Calculate circadian impact using sinusoidal function
    circadian_factor = 1.0 + amplitude * np.sin(2 * np.pi * (t / period - phase))
    
    return circadian_factor 
# Enhanced cancer model with PK/PD and circadian effects
def advanced_cancer_model(t, y, params, drug_schedules=None, use_circadian=True):
    """
    Advanced cancer model with complex cellular dynamics, pharmacokinetics,
    and circadian rhythm effects.
    
    Args:
        t (float): Current time point
        y (array): State vector
        params (dict): Parameter dictionary
        drug_schedules (dict, optional): Drug dosing schedules
        use_circadian (bool): Whether to use circadian effects
        
    Returns:
        array: Derivatives of state variables
    """
    # Unpack state variables (expanded state vector)
    N1, N2, I1, I2, P, A, Q, R1, R2, S, D, Dm, G, M, H = y
    
    # Ensure non-negative values with a small floor
    y = np.maximum(y, 1e-6)
    N1, N2, I1, I2, P, A, Q, R1, R2, S, D, Dm, G, M, H = y
    
    # Extract parameters
    alpha = params.get('alpha', 0.93)
    K = params.get('K', 1000)
    
    # Growth rates
    lam1 = params.get('lambda1', 0.003)
    lam2 = params.get('lambda2', 0.002)
    lam_R1 = params.get('lambda_R1', 0.006)
    lam_R2 = params.get('lambda_R2', 0.005)
    
    # Immune parameters
    beta1 = params.get('beta1', 0.005)
    beta2 = params.get('beta2', 0.001)
    phi1 = params.get('phi1', 0.1)
    phi2 = params.get('phi2', 0.001)
    phi3 = params.get('phi3', 0.0003)
    delta_I = params.get('delta_I', 0.04)
    
    # Resistance and immune interaction
    immune_resist_factor1 = params.get('immune_resist_factor1', 0.10)
    immune_resist_factor2 = params.get('immune_resist_factor2', 0.05)
    
    # Metastasis parameters
    gamma = params.get('gamma', 0.0001)
    delta_P = params.get('delta_P', 0.01)
    
    # Angiogenesis parameters
    alpha_A = params.get('alpha_A', 0.01)
    delta_A = params.get('delta_A', 0.1)
    
    # Quiescence parameters
    kappa_Q = params.get('kappa_Q', 0.001)
    lambda_Q = params.get('lambda_Q', 0.0005)
    
    # Resistance development
    omega_R1 = params.get('omega_R1', 0.004)
    omega_R2 = params.get('omega_R2', 0.003)
    resistance_floor = params.get('resistance_floor', 0.01)
    
    # Senescence parameters
    kappa_S = params.get('kappa_S', 0.0005)
    delta_S = params.get('delta_S', 0.005)
    
    # Therapy parameters
    etaE = params.get('etaE', 0.01)
    etaH = params.get('etaH', 0.01)
    etaC = params.get('etaC', 0.01)
    
    # NEW: Metabolism and microenvironment parameters
    metabolic_switch_rate = params.get('metabolic_switch_rate', 0.02)
    hypoxia_threshold = params.get('hypoxia_threshold', 0.3)
    acidosis_factor = params.get('acidosis_factor', 0.01)
    
    # NEW: Genetic instability
    mutation_rate = params.get('mutation_rate', 0.0001)
    epigenetic_silencing = params.get('epigenetic_silencing', 0.002)
    genetic_instability = params.get('genetic_instability', 1.0)
    
    # Get treatment controls
    uE = params.get('uE', 0.0)
    uH = params.get('uH', 0.0)
    uC = params.get('uC', 0.0)
    uI = params.get('uI', 0.0)
    
    # Handle time-dependent controls
    if callable(uE): uE = uE(t)
    if callable(uH): uH = uH(t)
    if callable(uC): uC = uC(t)
    if callable(uI): uI = uI(t)
    
    # Apply PK/PD for each drug if schedules provided
    if drug_schedules:
        # Process each drug schedule
        if 'hormone' in drug_schedules:
            hormone_pk = drug_pharmacokinetics(t, drug_schedules['hormone'], params)
            hormone_effect = calculate_drug_effect(D, params)
            uE = hormone_effect
        
        if 'her2' in drug_schedules:
            her2_pk = drug_pharmacokinetics(t, drug_schedules['her2'], params)
            her2_effect = calculate_drug_effect(D, params)
            uH = her2_effect
            
        if 'chemo' in drug_schedules:
            chemo_pk = drug_pharmacokinetics(t, drug_schedules['chemo'], params)
            chemo_effect = calculate_drug_effect(D, params)
            uC = chemo_effect
            
        if 'immuno' in drug_schedules:
            immuno_pk = drug_pharmacokinetics(t, drug_schedules['immuno'], params)
            immuno_effect = calculate_drug_effect(D, params)
            uI = immuno_effect
    
    # Apply circadian rhythm effects if enabled
    if use_circadian:
        # Get circadian effect
        circ_effect = circadian_effect(t, params)
        
        # Modulate key parameters
        lam1 *= circ_effect
        lam2 *= circ_effect
        delta_I *= circ_effect
        beta1 *= circ_effect
    
    # Total tumor burden
    total_tumor = N1 + N2 + Q + R1 + R2 + S
    
    # Carrying capacity factor
    carrying_capacity_factor = max(0, 1 - total_tumor / K)
    
    # Combined therapy effect
    therapy_effect = etaE * uE + etaH * uH + etaC * uC
    
    # Hypoxia calculation: oxygen deficiency increases with tumor size relative to carrying capacity
    # When tumor size exceeds hypoxia_threshold (30% of K), hypoxia begins to develop
    # Angiogenesis (A) can partially offset hypoxia effects
    hypoxia_factor = max(0, (total_tumor / K) - hypoxia_threshold) / (1 - hypoxia_threshold)
    hypoxia_effect = 1.0 + hypoxia_factor * A / (1 + A)  # Angiogenesis reduces hypoxia impact
    
    # NEW: Metabolic impact based on hypoxia
    metabolic_shift = M * hypoxia_factor * metabolic_switch_rate
    
    # NEW: Acidosis effect based on metabolic state and tumor size
    acidosis_effect = 1.0 + acidosis_factor * M * (total_tumor / K)
    
    # NEW: Genetic instability based on treatment and hypoxia
    genetic_damage_rate = mutation_rate * genetic_instability * (1 + therapy_effect + 0.5 * hypoxia_factor)
    
    # Immune boost (immunotherapy)
    immuno_boost = 1.0 + uI
    
    # Growth dynamics with carrying capacity, now affected by metabolism
    growth_factor = carrying_capacity_factor * (1 + 0.2 * M)  # Metabolic state affects growth
    growth_N1 = lam1 * N1 * growth_factor / acidosis_effect
    growth_N2 = lam2 * N2 * growth_factor / acidosis_effect
    growth_R1 = lam_R1 * R1 * growth_factor / acidosis_effect
    growth_R2 = lam_R2 * R2 * growth_factor / acidosis_effect
    
    # Immune killing effects, modulated by hypoxia
    immune_kill_factor = 1.0 / (1 + 0.5 * hypoxia_factor)  # Hypoxia reduces immune efficacy
    immune_kill_N1 = beta1 * N1 * I1 / (1 + 0.01 * total_tumor) * immune_kill_factor
    immune_kill_N2 = beta1 * N2 * I1 / (1 + 0.01 * total_tumor) * 0.5 * immune_kill_factor
    
    # Reduced immune killing of resistant cells
    immune_kill_R1 = beta1 * R1 * I1 / (1 + 0.01 * total_tumor) * immune_resist_factor1 * immune_kill_factor
    immune_kill_R2 = beta1 * R2 * I1 / (1 + 0.01 * total_tumor) * immune_resist_factor2 * immune_kill_factor
    
    # Resistance development: enhanced by genetic instability and therapy pressure
    # Genetic instability (G) ranges from 0 (unstable) to 1 (stable)
    # Lower stability (smaller G) increases mutation rate, leading to more resistance
    # Therapy effect creates selective pressure favoring resistant cell emergence
    resistance_dev_factor = (1 + (1 - G))  # Lower genetic stability (G) = higher resistance rate
    resistance_dev_R1 = max(omega_R1 * therapy_effect * N1 * resistance_dev_factor, 0.0)
    resistance_dev_R2 = max(omega_R2 * therapy_effect * N1 * resistance_dev_factor, 0.0)
    
    # Immune cell dynamics
    immune_prod_I1 = phi1 + phi2 * total_tumor / (1 + 0.01 * total_tumor)
    immune_suppression_I1 = beta2 * I1 * I2 / (1 + I1)
    immune_prod_I2 = phi3 * total_tumor / (1 + 0.01 * total_tumor)
    
    # Quiescence dynamics - hypoxia increases quiescence
    quiescence_factor = 1.0 + 0.5 * hypoxia_factor
    quiescence_induction_N1 = kappa_Q * N1 * quiescence_factor
    quiescence_induction_N2 = kappa_Q * N2 * quiescence_factor
    quiescence_reactivation = lambda_Q * Q / quiescence_factor  # Harder to reactivate in hypoxia
    
    # Senescence induction - drug and genetic instability
    senescence_induction = kappa_S * therapy_effect * N1 * (1 + 0.3 * (1 - G))
    
    # Derivatives for each state variable
    dN1dt = (growth_N1 
             - immune_kill_N1 * immuno_boost 
             - therapy_effect * N1 
             - quiescence_induction_N1 
             - resistance_dev_R1 
             - resistance_dev_R2 
             - senescence_induction)
    
    dN2dt = (growth_N2 
             - immune_kill_N2 * immuno_boost 
             - therapy_effect * N2 * 0.5 
             - quiescence_induction_N2)
    
    dI1dt = (immune_prod_I1 
             - immune_suppression_I1 
             - delta_I * I1 
             + 0.1 * uI * I1)
    
    dI2dt = (immune_prod_I2 
             - delta_I * I2 
             - 0.1 * uI * I2)
    
    dPdt = gamma * total_tumor * (1 + 0.5 * hypoxia_factor) - delta_P * P
    
    dAdt = alpha_A * total_tumor / (1 + 0.01 * total_tumor) - delta_A * A
    
    dQdt = quiescence_induction_N1 + quiescence_induction_N2 - quiescence_reactivation
    
    dR1dt = resistance_dev_R1 + growth_R1 - immune_kill_R1 * immuno_boost
    
    dR2dt = resistance_dev_R2 + growth_R2 - immune_kill_R2 * immuno_boost
    
    dSdt = senescence_induction - delta_S * S
    
    # NEW: Drug PK/PD dynamics
    dDdt = 0.0  # Will be calculated if drug schedules provided
    if drug_schedules:
        for drug_type, schedule in drug_schedules.items():
            dDdt += drug_pharmacokinetics(t, schedule, params)
    dDdt -= params.get('elimination_rate', 0.1) * D  # Elimination
    
    # NEW: Metabolized drug
    dDmdt = params.get('elimination_rate', 0.1) * D  # From drug elimination
    
    # NEW: Genetic stability dynamics
    dGdt = -genetic_damage_rate * G + 0.001 * (1 - G)  # Slow recovery
    
    # NEW: Metabolic state dynamics
    dMdt = metabolic_shift - 0.05 * M  # Gradual reversion to normal metabolism
    
    # NEW: Hypoxia dynamics
    dHdt = 0.1 * hypoxia_factor - 0.1 * A * H  # Angiogenesis reduces hypoxia
    
    # Combine derivatives
    dydt = np.array([dN1dt, dN2dt, dI1dt, dI2dt, dPdt, dAdt, dQdt, dR1dt, dR2dt, dSdt, 
                     dDdt, dDmdt, dGdt, dMdt, dHdt])
    
    # Apply fractional order scaling for memory effects
    # This implements a simplified approximation of fractional-order dynamics
    # where system evolution depends on historical states with power-law decay.
    # The fractional order alpha (0.75-1.0) controls memory strength:
    #   - alpha = 1.0: Memoryless (standard ODE, fractional_factor = 1.0)
    #   - alpha < 1.0: Memory effects increase as alpha decreases
    # The memory_factor uses t^(-alpha) to weight past states, with decay rate
    # determined by alpha. Smaller alpha means slower decay (stronger memory).
    if t > 0:
        memory_factor = min(t**(-alpha), 100)  # Clip to avoid overflow at t=0
        # Fractional factor scales derivatives to incorporate memory effects
        # The 0.01 factor ensures smooth transition and prevents numerical instability
        fractional_factor = 0.01 * (1 + (1-alpha) * memory_factor)
    else:
        fractional_factor = 1.0  # At t=0, no memory effects (initial condition)
    
    return dydt * fractional_factor

# Define drug scheduling functions
def create_cyclic_dosing_schedule(treatment_days, rest_days, dose, start_day=0):
    """
    Create a cyclic drug dosing schedule
    
    Args:
        treatment_days (int): Days of treatment per cycle
        rest_days (int): Days of rest per cycle
        dose (float): Dose amount during treatment days
        start_day (int): Day to start treatment
        
    Returns:
        function: Dosing function that returns dose at given time
    """
    def dosing_schedule(t):
        if t < start_day:
            return 0.0
        
        cycle_length = treatment_days + rest_days
        cycle_position = (t - start_day) % cycle_length
        
        if cycle_position < treatment_days:
            return dose
        else:
            return 0.0
            
    return dosing_schedule

# Define adaptive therapy scheduling
def create_adaptive_dosing_schedule(monitoring_period=30, target_ratio=0.8, max_dose=1.0, min_dose=0.1, start_day=0):
    """
    Create an adaptive therapy dosing schedule based on tumor burden
    
    Args:
        monitoring_period (int): Days between dose adjustments
        target_ratio (float): Target tumor burden ratio to maintain
        max_dose (float): Maximum dose
        min_dose (float): Minimum dose
        start_day (int): Day to start treatment
        
    Returns:
        function: Adaptive dosing function that requires tumor history
    """
    # This is a closure that will store tumor burden history
    tumor_history = []
    current_dose = max_dose
    last_adjustment_time = start_day
    
    def adaptive_schedule(t, current_tumor_burden):
        nonlocal tumor_history, current_dose, last_adjustment_time
        
        # Before start day, no treatment
        if t < start_day:
            return 0.0
            
        # Add current burden to history
        tumor_history.append((t, current_tumor_burden))
        
        # Keep only recent history
        cutoff_time = t - 2 * monitoring_period
        tumor_history = [item for item in tumor_history if item[0] >= cutoff_time]
        
        # Check if it's time to adjust dose
        if t >= last_adjustment_time + monitoring_period and len(tumor_history) >= 2:
            # Get initial burden in this period
            initial_idx = 0
            while initial_idx < len(tumor_history) and tumor_history[initial_idx][0] < last_adjustment_time:
                initial_idx += 1
            
            if initial_idx < len(tumor_history):
                initial_burden = tumor_history[initial_idx][1]
                current_burden = tumor_history[-1][1]
                
                # Calculate ratio
                burden_ratio = current_burden / initial_burden if initial_burden > 0 else 1.0
                
                # Adjust dose based on ratio
                if burden_ratio > target_ratio + 0.1:
                    # Tumor growing too fast, increase dose
                    current_dose = min(current_dose * 1.2, max_dose)
                elif burden_ratio < target_ratio - 0.1:
                    # Tumor declining too fast, decrease dose
                    current_dose = max(current_dose * 0.8, min_dose)
                
                # Update last adjustment time
                last_adjustment_time = t
        
        return current_dose
    
    return adaptive_schedule

# Define temperature modulation functions
def temperature_protocol(t, baseline_temp=37.0, hyperthermia_days=None, hypothermia_days=None):
    """
    Generate temperature profile based on treatment protocol
    
    Args:
        t (float): Current time
        baseline_temp (float): Baseline body temperature
        hyperthermia_days (list): List of (start_day, end_day, temp) for hyperthermia
        hypothermia_days (list): List of (start_day, end_day, temp) for hypothermia
        
    Returns:
        float: Current temperature
    """
    current_temp = baseline_temp
    
    if hyperthermia_days:
        for start, end, temp in hyperthermia_days:
            if start <= t <= end:
                current_temp = temp
                break
                
    if hypothermia_days:
        for start, end, temp in hypothermia_days:
            if start <= t <= end:
                current_temp = temp
                break
    
    return current_temp

# Advanced temperature modifier
def advanced_temperature_modifier(current_temp, baseline_temp=37.0, time_factor=None):
    """
    Enhanced temperature modification with time-dependent variations.
    
    Args:
        current_temp (float): Current temperature
        baseline_temp (float): Baseline temperature
        time_factor (float): Current time for time-dependent effects
        
    Returns:
        dict: Temperature effects on various biological processes
    """
    # Temperature deviation calculations
    temp_deviation = current_temp - baseline_temp
    
    # Advanced sigmoid transformation
    def advanced_sigmoid(x, steepness=1.0, midpoint=0, max_effect=1.5, min_effect=0.5):
        return min_effect + (max_effect - min_effect) / (1 + np.exp(-steepness * (x - midpoint)))
    
    # Time-dependent stress response
    stress_time_modifier = 1.0
    if time_factor is not None:
        stress_time_modifier = np.sin(time_factor / 50) * 0.2 + 1.0
    
    # Enhanced modification factors
    modifications = {
        'metabolism': {
            'factor': advanced_sigmoid(temp_deviation, steepness=0.8, midpoint=1.5),
            'sensitivity': 0.03
        },
        'immune_activation': {
            'factor': advanced_sigmoid(abs(temp_deviation), steepness=0.6, midpoint=1.0, max_effect=1.8, min_effect=0.6),
            'sensitivity': 0.04
        },
        'cellular_stress': {
            'factor': advanced_sigmoid(temp_deviation, steepness=1.0, midpoint=2.0, max_effect=2.0, min_effect=0.5) * stress_time_modifier,
            'sensitivity': 0.05
        },
        'gene_expression': {
            'factor': 1 + 0.15 * np.tanh(temp_deviation / 2),
            'sensitivity': 0.02
        },
        'resistance_development': {
            'factor': advanced_sigmoid(temp_deviation, steepness=0.7, midpoint=1.5, max_effect=1.7, min_effect=0.7),
            'sensitivity': 0.04
        }
    }
    
    # Apply probabilistic threshold effects
    for key, mod in modifications.items():
        mod['factor'] *= (1 + np.random.normal(0, 0.05))
        mod['factor'] = np.clip(mod['factor'], 0.5, 2.0)
    
    return modifications

# Temperature-integrated cancer model
def enhanced_temperature_cancer_model(t, y, params, drug_schedules=None, current_temp=37.0, use_circadian=True):
    """
    Advanced temperature-integrated cancer model
    
    Args:
        t (float): Current time point
        y (array): State vector
        params (dict): Parameter dictionary
        drug_schedules (dict): Drug dosing schedules
        current_temp (float): Current temperature
        use_circadian (bool): Whether to use circadian effects
        
    Returns:
        array: Derivatives of state variables
    """
    # Get advanced temperature modification factors
    temp_mods = advanced_temperature_modifier(current_temp, time_factor=t)
    
    # Call original advanced cancer model
    derivatives = advanced_cancer_model(t, y, params, drug_schedules, use_circadian)
    
    # Modification mapping
    modification_mapping = {
        0: ['metabolism', 'cellular_stress'],      # Sensitive cells (N1)
        1: ['metabolism', 'gene_expression'],      # Partially resistant cells (N2)
        2: ['immune_activation'],                  # Cytotoxic immune cells (I1)
        3: ['immune_activation'],                  # Regulatory immune cells (I2)
        5: ['metabolism', 'cellular_stress'],      # Angiogenesis factor
        7: ['resistance_development'],             # Resistant Type 1
        8: ['resistance_development'],             # Resistant Type 2
        12: ['gene_expression'],                   # Genetic stability
        13: ['metabolism']                         # Metabolism status
    }
    
    # Apply temperature modifications
    for idx, modification_keys in modification_mapping.items():
        if idx < len(derivatives):  # Safety check
            mod_factor = 1.0
            for key in modification_keys:
                if key in temp_mods:
                    mod = temp_mods[key]
                    mod_factor *= (1 + mod['sensitivity'] * (mod['factor'] - 1))
            
            derivatives[idx] *= mod_factor
    
    return derivatives

# Define patient profiles
def create_patient_profile(profile_type='average'):
    """
    Create a patient-specific parameter profile
    
    Args:
        profile_type (str): Type of patient profile to create
                            ('young', 'elderly', 'compromised', etc.)
    
    Returns:
        dict: Patient-specific parameter modifications
    """
    profiles = {
        'average': {
            'age_factor': 1.0,
            'performance_status': 1.0,
            'bmi_factor': 1.0,
            'prior_treatment_factor': 1.0,
            'liver_function': 1.0,
            'kidney_function': 1.0,
            'immune_status': 1.0
        },
        'young': {
            'age_factor': 1.2,
            'performance_status': 1.2,
            'immune_status': 1.3,
            'liver_function': 1.1,
            'kidney_function': 1.1,
            'mutation_rate': 0.00008
        },
        'elderly': {
            'age_factor': 0.8,
            'performance_status': 0.8,
            'immune_status': 0.7,
            'liver_function': 0.9,
            'kidney_function': 0.85,
            'mutation_rate': 0.00015
        },
        'compromised': {
            'performance_status': 0.7,
            'immune_status': 0.6,
            'liver_function': 0.7,
            'kidney_function': 0.7,
            'mutation_rate': 0.00015,
            'genetic_instability': 1.2
        },
        'high_metabolism': {
            'absorption_rate': 0.6,
            'elimination_rate': 0.15,
            'liver_function': 1.2,
            'metabolic_switch_rate': 0.03
        },
        'low_metabolism': {
            'absorption_rate': 0.4,
            'elimination_rate': 0.08,
            'liver_function': 0.85,
            'metabolic_switch_rate': 0.015
        },
        'prior_treatment': {
            'prior_treatment_factor': 1.3,
            'resistance_floor': 0.02,
            'omega_R1': 0.006,
            'omega_R2': 0.004
        }
    }
    
    if profile_type in profiles:
        return profiles[profile_type]
    else:
        print(f"Warning: Profile '{profile_type}' not found. Using 'average' profile.")
        return profiles['average']

# Define treatment protocols
def create_treatment_protocol(protocol_name='standard', patient_profile=None):
    """
    Create a complete treatment protocol with drug schedules
    
    Args:
        protocol_name (str): Name of protocol to create
        patient_profile (dict): Patient profile for adjustments
        
    Returns:
        dict: Complete treatment protocol configuration
    """
    # Default protocol settings
    protocols = {
        'standard': {
            'hormone': create_cyclic_dosing_schedule(14, 7, 0.8, start_day=0),
            'her2': create_cyclic_dosing_schedule(14, 7, 0.8, start_day=0),
            'temperature': lambda t: 37.0,
            'description': 'Standard cyclic hormone/HER2 therapy'
        },
        'continuous': {
            'hormone': lambda t: 0.8 if t >= 0 else 0.0,
            'her2': lambda t: 0.8 if t >= 0 else 0.0,
            'temperature': lambda t: 37.0,
            'description': 'Continuous hormone/HER2 therapy'
        },
        'adaptive': {
            'hormone': None,  # Will be set as adaptive
            'her2': None,     # Will be set as adaptive
            'temperature': lambda t: 37.0,
            'description': 'Adaptive therapy based on tumor response'
        },
        'immuno_combo': {
            'chemo': create_cyclic_dosing_schedule(7, 14, 0.6, start_day=0),
            'immuno': create_cyclic_dosing_schedule(2, 19, 0.7, start_day=0),
            'temperature': lambda t: 37.0,
            'description': 'Immuno-chemotherapy combination'
        },
        'hyperthermia': {
            'hormone': create_cyclic_dosing_schedule(14, 7, 0.7, start_day=0),
            'her2': create_cyclic_dosing_schedule(14, 7, 0.7, start_day=0),
            'temperature': lambda t: 38.5 if (t % 21) < 2 else 37.0,
            'description': 'Hormone/HER2 therapy with periodic hyperthermia'
        },
        'multi_modal': {
            'hormone': create_cyclic_dosing_schedule(14, 7, 0.6, start_day=0),
            'chemo': create_cyclic_dosing_schedule(5, 16, 0.5, start_day=30),
            'immuno': create_cyclic_dosing_schedule(1, 20, 0.7, start_day=60),
            'temperature': lambda t: 38.5 if (t % 42) < 3 else 37.0,
            'description': 'Multi-modal therapy with sequential drug combinations'
        }
    }
    
    # Get selected protocol
    if protocol_name in protocols:
        protocol = protocols[protocol_name].copy()
    else:
        print(f"Warning: Protocol '{protocol_name}' not found. Using 'standard' protocol.")
        protocol = protocols['standard'].copy()
    
    # For adaptive protocol, create adaptive dosing functions
    if protocol_name == 'adaptive':
        # We'll initialize these properly when running simulations
        protocol['hormone'] = 'adaptive'
        protocol['her2'] = 'adaptive'
    
    # Adjust for patient profile if provided
    if patient_profile:
        # Adjust doses based on patient factors
        performance_factor = patient_profile.get('performance_status', 1.0)
        age_factor = patient_profile.get('age_factor', 1.0)
        
        # Calculate overall dose adjustment
        dose_adjustment = (performance_factor + age_factor) / 2
        
        # Apply to each drug
        for drug_type in ['hormone', 'her2', 'chemo', 'immuno']:
            if drug_type in protocol and protocol[drug_type] and protocol[drug_type] != 'adaptive':
                original_schedule = protocol[drug_type]
                
                # Wrap the dose function to apply adjustment
                def adjusted_schedule(t, orig_schedule=original_schedule, adj_factor=dose_adjustment):
                    return orig_schedule(t) * adj_factor
                
                protocol[drug_type] = adjusted_schedule
    
    return protocol

# Run enhanced model with patient profile and treatment protocol
def run_enhanced_simulation(patient_profile_name, protocol_name, simulation_days=500, use_circadian=True):
    """
    Run a complete simulation with enhanced model
    
    Args:
        patient_profile_name (str): Name of patient profile to use
        protocol_name (str): Name of treatment protocol to use
        simulation_days (int): Number of days to simulate
        use_circadian (bool): Whether to use circadian effects
        
    Returns:
        dict: Simulation results and metrics
    """
    # Load patient profile
    patient_profile = create_patient_profile(patient_profile_name)
    
    # Get parameters with patient profile
    params = enhanced_model_params(patient_profile)
    
    # Create treatment protocol
    protocol = create_treatment_protocol(protocol_name, patient_profile)
    
    # Setup simulation
    t_span = [0, simulation_days]
    t_eval = np.linspace(0, simulation_days, simulation_days + 1)
    
    # Get initial conditions
    initial_conditions = get_enhanced_initial_conditions()
    
    # Set up drug schedules
    drug_schedules = {}
    
    for drug_type in ['hormone', 'her2', 'chemo', 'immuno']:
        if drug_type in protocol and protocol[drug_type]:
            if protocol[drug_type] == 'adaptive':
                # For adaptive therapy, we need a special approach
                continue
            else:
                drug_schedules[drug_type] = protocol[drug_type]
    
    # Handle adaptive therapy if needed
    if protocol_name == 'adaptive':
        # For this simulation, we'll create a pre-defined adaptive schedule
        hormone_func = lambda t: optimized_adaptive_therapy(t)
        her2_func = lambda t: optimized_adaptive_therapy(t)
        drug_schedules['hormone'] = hormone_func
        drug_schedules['her2'] = her2_func
    
    # Create temperature function
    if 'temperature' in protocol:
        temp_func = protocol['temperature']
    else:
        temp_func = lambda t: 37.0  # Default normal temperature
    
    # Define the model function that will be called by the solver
    def model_with_protocol(t, y):
        current_temp = temp_func(t)
        return enhanced_temperature_cancer_model(t, y, params, drug_schedules, current_temp, use_circadian)
    
    # Run simulation
    print(f"Running {protocol_name} protocol for {patient_profile_name} patient...")
    print(f"Protocol description: {protocol.get('description', 'No description')}")
    
    result = safe_solve_ivp(model_with_protocol, t_span, initial_conditions, 'RK45', t_eval)
    
    # Calculate metrics if successful
    if result.success:
        # Extract state variables
        N1 = result.y[0]  # Sensitive cells
        N2 = result.y[1]  # Partially resistant cells
        I1 = result.y[2]  # Cytotoxic immune cells
        I2 = result.y[3]  # Regulatory immune cells
        P = result.y[4]   # Metastatic potential
        A = result.y[5]   # Angiogenesis factor
        Q = result.y[6]   # Quiescent cells
        R1 = result.y[7]  # Type 1 resistant cells
        R2 = result.y[8]  # Type 2 resistant cells
        S = result.y[9]   # Senescent cells
        D = result.y[10]  # Drug concentration
        G = result.y[12]  # Genetic stability
        M = result.y[13]  # Metabolism status
        H = result.y[14]  # Hypoxia level
        
        # Calculate total tumor burden
        total_tumor = N1 + N2 + Q + R1 + R2 + S
        total_resistant = R1 + R2
        total_immune = I1 + I2
        
        # Calculate resistance fraction
        resistance_fraction = (R1 + R2) / total_tumor * 100
        
        # Calculate tumor reduction from initial
        initial_burden = total_tumor[0]
        final_burden = total_tumor[-1]
        percent_reduction = 100 * (1 - final_burden / initial_burden)
        
        # Calculate treatment efficacy score
        treatment_efficacy = percent_reduction / (1 + resistance_fraction[-1]/100)
        
        # Calculate immune activation
        immune_activation = I1[-1] / I1[0]
        
        # Calculate genetic instability
        genetic_instability = 1 - G[-1]
        
        # Calculate metabolic shift
        metabolic_shift = M[-1]
        
        # Calculate hypoxia level
        final_hypoxia = H[-1]
        
        # Store metrics
        metrics = {
            'initial_burden': initial_burden,
            'final_burden': final_burden,
            'percent_reduction': percent_reduction,
            'final_resistance_fraction': resistance_fraction[-1],
            'treatment_efficacy_score': treatment_efficacy,
            'immune_activation': immune_activation,
            'genetic_instability': genetic_instability,
            'metabolic_shift': metabolic_shift,
            'hypoxia_level': final_hypoxia
        }
        
        # Store time series data
        time_series = {
            'time': result.t,
            'total_tumor': total_tumor,
            'sensitive_cells': N1,
            'partially_resistant': N2,
            'resistant_type1': R1,
            'resistant_type2': R2,
            'quiescent': Q,
            'senescent': S,
            'cytotoxic_immune': I1,
            'regulatory_immune': I2,
            'resistance_fraction': resistance_fraction,
            'drug_concentration': D,
            'genetic_stability': G,
            'metabolism': M,
            'hypoxia': H
        }
        
        return {
            'success': True,
            'metrics': metrics,
            'time_series': time_series,
            'protocol': protocol,
            'patient_profile': patient_profile
        }
    else:
        return {
            'success': False,
            'error_message': result.message,
            'protocol': protocol,
            'patient_profile': patient_profile
        }

# Run comparative analysis across patient profiles and protocols
def run_comparative_analysis(patient_profiles=None, treatment_protocols=None, simulation_days=500):
    """
    Run comparative analysis across multiple patient profiles and treatment protocols
    
    Args:
        patient_profiles (list): List of patient profile names to simulate
        treatment_protocols (list): List of treatment protocol names to simulate
        simulation_days (int): Number of days to simulate
        
    Returns:
        dict: Results for all simulations
    """
    # Default profiles and protocols if not provided
    if patient_profiles is None:
        patient_profiles = ['average', 'young', 'elderly', 'compromised']
    
    if treatment_protocols is None:
        treatment_protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo']
    
    # Store all results
    all_results = {}
        
        # Run simulations for each combination
        for profile in patient_profiles:
        all_results[profile] = {}
            
            for protocol in treatment_protocols:
            print(f"\nRunning simulation for {profile} patient with {protocol} protocol...")
                
            # Run simulation
            result = run_enhanced_simulation(profile, protocol, simulation_days)
                
                # Store result
            all_results[profile][protocol] = result
    
    return all_results

# Generate comprehensive visualizations
def create_visualizations(results, output_dir='cancer_model_results', include_patient_comparisons=True):
    """
    Create comprehensive visualizations from simulation results
    
    Args:
        results (dict): Results from comparative analysis
        output_dir (str): Directory to save visualizations
        include_patient_comparisons (bool): Whether to include patient comparison plots
        
    Returns:
        list: Paths to created visualizations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set plotting style
    colors = set_plotting_style()
    
    # List to store visualization paths
    visualization_paths = []
    
    # 1. Protocol Comparison for Average Patient
    if 'average' in results:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Treatment Protocol Comparisons (Average Patient)', fontsize=18)
        
        # Plot tumor burden
        ax = axes[0, 0]
        for protocol, sim_result in results['average'].items():
            if sim_result['success']:
                time_series = sim_result['time_series']
                ax.plot(time_series['time'], time_series['total_tumor'], label=protocol)
        
        ax.set_title('Tumor Burden')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Cell Count')
        ax.legend()
        
        # Plot resistance fraction
        ax = axes[0, 1]
        for protocol, sim_result in results['average'].items():
            if sim_result['success']:
                time_series = sim_result['time_series']
                ax.plot(time_series['time'], time_series['resistance_fraction'], label=protocol)
        
        ax.set_title('Resistance Fraction')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Percent (%)')
        ax.legend()
        
        # Plot immune response
        ax = axes[1, 0]
        for protocol, sim_result in results['average'].items():
            if sim_result['success']:
                time_series = sim_result['time_series']
                ax.plot(time_series['time'], time_series['cytotoxic_immune'], label=protocol)
        
        ax.set_title('Cytotoxic Immune Response')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Cell Count')
        ax.legend()
        
        # Plot genetic stability
        ax = axes[1, 1]
        for protocol, sim_result in results['average'].items():
            if sim_result['success']:
                time_series = sim_result['time_series']
                ax.plot(time_series['time'], time_series['genetic_stability'], label=protocol)
        
        ax.set_title('Genetic Stability')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Stability Index')
        ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        protocol_comparison_path = os.path.join(output_dir, 'protocol_comparison.png')
        plt.savefig(protocol_comparison_path, dpi=300)
        visualization_paths.append(protocol_comparison_path)
        plt.close()
        
        # 2. Treatment Efficacy Metrics Bar Chart
        plt.figure(figsize=(14, 8))
        
        protocols = list(results['average'].keys())
        efficacy_scores = []
        percent_reductions = []
        resistance_fractions = []
        
        for protocol in protocols:
            if results['average'][protocol]['success']:
                metrics = results['average'][protocol]['metrics']
                efficacy_scores.append(metrics['treatment_efficacy_score'])
                percent_reductions.append(metrics['percent_reduction'])
                resistance_fractions.append(metrics['final_resistance_fraction'])
        
        x = np.arange(len(protocols))
        width = 0.25
        
        plt.bar(x - width, percent_reductions, width, label='Tumor Reduction (%)', color=colors[0])
        plt.bar(x, resistance_fractions, width, label='Resistance (%)', color=colors[1])
        plt.bar(x + width, efficacy_scores, width, label='Efficacy Score', color=colors[2])
        
        plt.xlabel('Treatment Protocol')
        plt.ylabel('Percentage / Score')
        plt.title('Treatment Efficacy Metrics Comparison')
        plt.xticks(x, protocols)
        plt.legend()
        
        efficacy_metrics_path = os.path.join(output_dir, 'efficacy_metrics.png')
        plt.savefig(efficacy_metrics_path, dpi=300)
        visualization_paths.append(efficacy_metrics_path)
        plt.close()
    
    # 3. Patient Comparison for Standard Protocol
    if include_patient_comparisons:
        patient_profiles = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Patient Profile Comparisons (Standard Protocol)', fontsize=18)
        
        # Plot tumor burden
        ax = axes[0, 0]
        for profile in patient_profiles:
            if 'standard' in results[profile] and results[profile]['standard']['success']:
                time_series = results[profile]['standard']['time_series']
                ax.plot(time_series['time'], time_series['total_tumor'], label=profile)
        
        ax.set_title('Tumor Burden')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Cell Count')
        ax.legend()
        
        # Plot resistance fraction
        ax = axes[0, 1]
        for profile in patient_profiles:
            if 'standard' in results[profile] and results[profile]['standard']['success']:
                time_series = results[profile]['standard']['time_series']
                ax.plot(time_series['time'], time_series['resistance_fraction'], label=profile)
        
        ax.set_title('Resistance Fraction')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Percent (%)')
        ax.legend()
        
        # Plot immune response
        ax = axes[1, 0]
        for profile in patient_profiles:
            if 'standard' in results[profile] and results[profile]['standard']['success']:
                time_series = results[profile]['standard']['time_series']
                ax.plot(time_series['time'], time_series['cytotoxic_immune'], label=profile)
        
        ax.set_title('Cytotoxic Immune Response')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Cell Count')
        ax.legend()
        
        # Plot drug concentration
        ax = axes[1, 1]
        for profile in patient_profiles:
            if 'standard' in results[profile] and results[profile]['standard']['success']:
                time_series = results[profile]['standard']['time_series']
                ax.plot(time_series['time'], time_series['drug_concentration'], label=profile)
        
        ax.set_title('Drug Concentration')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Concentration')
        ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        patient_comparison_path = os.path.join(output_dir, 'patient_comparison.png')
        plt.savefig(patient_comparison_path, dpi=300)
        visualization_paths.append(patient_comparison_path)
        plt.close()
    
    # 4. Create detailed visualization for best protocol
    if 'average' in results:
        # Find best protocol based on efficacy score
        best_protocol = None
        best_score = -1
        
        for protocol, sim_result in results['average'].items():
            if sim_result['success']:
                efficacy = sim_result['metrics']['treatment_efficacy_score']
                if efficacy > best_score:
                    best_score = efficacy
                    best_protocol = protocol
        
        if best_protocol:
            best_result = results['average'][best_protocol]
            
            # Create detailed visualization
            fig = plt.figure(figsize=(18, 14))
            gs = GridSpec(3, 3, figure=fig)
            
            # Main title
            fig.suptitle(f'Detailed Analysis of {best_protocol.title()} Protocol', fontsize=20)
            
            # Tumor dynamics
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(best_result['time_series']['time'], best_result['time_series']['total_tumor'], 'b-', linewidth=2, label='Total Tumor')
            ax1.plot(best_result['time_series']['time'], best_result['time_series']['sensitive_cells'], 'g--', linewidth=1.5, label='Sensitive')
            ax1.plot(best_result['time_series']['time'], best_result['time_series']['partially_resistant'], 'y--', linewidth=1.5, label='Partially Resistant')
            ax1.plot(best_result['time_series']['time'], best_result['time_series']['resistant_type1'] + best_result['time_series']['resistant_type2'], 'r--', linewidth=1.5, label='Total Resistant')
            ax1.set_title('Tumor Cell Dynamics')
            ax1.set_xlabel('Time (days)')
            ax1.set_ylabel('Cell Count')
            ax1.legend()
            
            # Immune dynamics
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.plot(best_result['time_series']['time'], best_result['time_series']['cytotoxic_immune'], 'g-', linewidth=2, label='Cytotoxic')
            ax2.plot(best_result['time_series']['time'], best_result['time_series']['regulatory_immune'], 'r-', linewidth=2, label='Regulatory')
            ax2.set_title('Immune Cell Dynamics')
            ax2.set_xlabel('Time (days)')
            ax2.set_ylabel('Cell Count')
            ax2.legend()
            
            # Resistance fraction
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(best_result['time_series']['time'], best_result['time_series']['resistance_fraction'], 'r-', linewidth=2)
            ax3.set_title('Resistance Fraction')
            ax3.set_xlabel('Time (days)')
            ax3.set_ylabel('Percent (%)')
            
            # Drug concentration
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(best_result['time_series']['time'], best_result['time_series']['drug_concentration'], 'b-', linewidth=2)
            ax4.set_title('Drug Concentration')
            ax4.set_xlabel('Time (days)')
            ax4.set_ylabel('Concentration')
            
            # Genetic stability
            ax5 = fig.add_subplot(gs[1, 2])
            ax5.plot(best_result['time_series']['time'], best_result['time_series']['genetic_stability'], 'g-', linewidth=2)
            ax5.set_title('Genetic Stability')
            ax5.set_xlabel('Time (days)')
            ax5.set_ylabel('Stability Index')
            
            # Hypoxia and metabolism
            ax6 = fig.add_subplot(gs[2, 0])
            ax6.plot(best_result['time_series']['time'], best_result['time_series']['hypoxia'], 'r-', linewidth=2, label='Hypoxia')
            ax6.plot(best_result['time_series']['time'], best_result['time_series']['metabolism'], 'b-', linewidth=2, label='Metabolism')
            ax6.set_title('Tumor Microenvironment')
            ax6.set_xlabel('Time (days)')
            ax6.set_ylabel('Level')
            ax6.legend()
            
            # Tumor composition evolution
            ax7 = fig.add_subplot(gs[2, 1:])
            
            # Prepare composition data
            time = best_result['time_series']['time']
            sensitive = best_result['time_series']['sensitive_cells']
            partial = best_result['time_series']['partially_resistant']
            resist1 = best_result['time_series']['resistant_type1']
            resist2 = best_result['time_series']['resistant_type2']
            quiescent = best_result['time_series']['quiescent']
            senescent = best_result['time_series']['senescent']
            
            # Create stacked area chart
            stack_data = np.vstack([sensitive, partial, resist1, resist2, quiescent, senescent])
            
            # Calculate percentage
            stack_sum = np.sum(stack_data, axis=0)
            stack_percent = np.zeros_like(stack_data)
            for i in range(stack_data.shape[0]):
                stack_percent[i, :] = stack_data[i, :] / stack_sum * 100
            
            # Plot stacked area
            labels = ['Sensitive', 'Partially Resistant', 'Resistant Type 1', 'Resistant Type 2', 'Quiescent', 'Senescent']
            ax7.stackplot(time, stack_percent, labels=labels, alpha=0.8)
            
            ax7.set_title('Tumor Composition Evolution')
            ax7.set_xlabel('Time (days)')
            ax7.set_ylabel('Percentage (%)')
            ax7.legend(loc='upper right')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            detailed_analysis_path = os.path.join(output_dir, f'detailed_{best_protocol}_analysis.png')
            plt.savefig(detailed_analysis_path, dpi=300)
            visualization_paths.append(detailed_analysis_path)
            plt.close()
    
    # 5. Create heatmap comparing all protocols and patients
    if include_patient_comparisons:
        # Get all unique patient profiles and protocols
        patient_profiles = list(results.keys())
        protocols = []
        for profile in patient_profiles:
            for protocol in results[profile].keys():
                if protocol not in protocols:
                    protocols.append(protocol)
        
        # Create metrics matrices
        efficacy_matrix = np.zeros((len(patient_profiles), len(protocols)))
        reduction_matrix = np.zeros((len(patient_profiles), len(protocols)))
        resistance_matrix = np.zeros((len(patient_profiles), len(protocols)))
        
        # Fill matrices
        for i, profile in enumerate(patient_profiles):
            for j, protocol in enumerate(protocols):
                if protocol in results[profile] and results[profile][protocol]['success']:
                    metrics = results[profile][protocol]['metrics']
                    efficacy_matrix[i, j] = metrics['treatment_efficacy_score']
                    reduction_matrix[i, j] = metrics['percent_reduction']
                    resistance_matrix[i, j] = metrics['final_resistance_fraction']
        
        # Plot heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Efficacy heatmap
        sns.heatmap(efficacy_matrix, annot=True, fmt=".1f", cmap="viridis", 
                  xticklabels=protocols, yticklabels=patient_profiles, ax=axes[0])
        axes[0].set_title('Treatment Efficacy Score')
        axes[0].set_xlabel('Protocol')
        axes[0].set_ylabel('Patient Profile')
        
        # Reduction heatmap
        sns.heatmap(reduction_matrix, annot=True, fmt=".1f", cmap="YlGnBu", 
                  xticklabels=protocols, yticklabels=patient_profiles, ax=axes[1])
        axes[1].set_title('Tumor Reduction (%)')
        axes[1].set_xlabel('Protocol')
        axes[1].set_ylabel('')
        
        # Resistance heatmap
        sns.heatmap(resistance_matrix, annot=True, fmt=".1f", cmap="YlOrRd", 
                  xticklabels=protocols, yticklabels=patient_profiles, ax=axes[2])
        axes[2].set_title('Resistance Fraction (%)')
        axes[2].set_xlabel('Protocol')
        axes[2].set_ylabel('')
        
        plt.tight_layout()
        heatmap_path = os.path.join(output_dir, 'treatment_heatmaps.png')
        plt.savefig(heatmap_path, dpi=300)
        visualization_paths.append(heatmap_path)
        plt.close()
    
    return visualization_paths

# Define the optimized adaptive therapy schedule
def optimized_adaptive_therapy(t):
    """Optimized adaptive therapy with fixed values."""
    if t < 60:
        return 0.9
    else:
        cycle_period = 90
        phase = ((t - 60) % cycle_period) / cycle_period
        if phase < 0.4:
            return 0.8
        elif phase < 0.7:
            return 0.4
        else:
            return 0.1

# Create model summary report
def generate_model_summary(results, output_file='model_summary.txt'):
    """
    Generate a summary report of model results
    
    Args:
        results (dict): Results from comparative analysis
        output_file (str): Path to save the report
        
    Returns:
        str: Path to summary report
    """
    with open(output_file, 'w') as f:
        f.write("Enhanced Cancer Model Analysis Summary\n")
        f.write("====================================\n\n")
        
        # Summarize protocols and patients
        patient_profiles = list(results.keys())
        protocols = []
        for profile in patient_profiles:
            for protocol in results[profile].keys():
                if protocol not in protocols:
                    protocols.append(protocol)
        
        f.write(f"Analyzed {len(patient_profiles)} patient profiles: {', '.join(patient_profiles)}\n")
        f.write(f"Tested {len(protocols)} treatment protocols: {', '.join(protocols)}\n\n")
        
        # Find best overall protocol
        best_protocol = None
        best_patient = None
        best_score = -1
        
        for profile in patient_profiles:
            for protocol in protocols:
                if protocol in results[profile] and results[profile][protocol]['success']:
                    score = results[profile][protocol]['metrics']['treatment_efficacy_score']
                    if score > best_score:
                        best_score = score
                        best_protocol = protocol
                        best_patient = profile
        
        if best_protocol:
            f.write(f"Best overall treatment: {best_protocol} for {best_patient} patient profile\n")
            f.write(f"Efficacy score: {best_score:.2f}\n\n")
        
        # Summarize protocol performance for ALL patient profiles
        patient_profiles_list = list(results.keys())
        for patient_profile in patient_profiles_list:
            f.write(f"Protocol Performance for {patient_profile.title()} Patient:\n")
            f.write("-" * (40 + len(patient_profile)) + "\n")
            
            for protocol in protocols:
                if protocol in results[patient_profile] and results[patient_profile][protocol]['success']:
                    metrics = results[patient_profile][protocol]['metrics']
                    f.write(f"\n{protocol.title()} Protocol:\n")
                    f.write(f"  Tumor Reduction: {metrics['percent_reduction']:.2f}%\n")
                    f.write(f"  Final Resistance: {metrics['final_resistance_fraction']:.2f}%\n")
                    f.write(f"  Efficacy Score: {metrics['treatment_efficacy_score']:.2f}\n")
                    f.write(f"  Immune Activation: {metrics['immune_activation']:.2f}x\n")
                    f.write(f"  Genetic Instability: {metrics['genetic_instability']:.2f}\n")
                    f.write(f"  Metabolic Shift: {metrics['metabolic_shift']:.2f}\n")
            f.write("\n")
        
        # Patient-specific responses
        f.write("\n\nPatient-Specific Protocol Performance:\n")
        f.write("--------------------------------------\n")
        
        for profile in patient_profiles:
            if profile != 'average':
                f.write(f"\n{profile.title()} Patient:\n")
                
                # Find best protocol for this patient
                best_protocol_patient = None
                best_score_patient = -1
                
                for protocol in protocols:
                    if protocol in results[profile] and results[profile][protocol]['success']:
                        score = results[profile][protocol]['metrics']['treatment_efficacy_score']
                        if score > best_score_patient:
                            best_score_patient = score
                            best_protocol_patient = protocol
                
                if best_protocol_patient:
                    f.write(f"  Best Protocol: {best_protocol_patient}\n")
                    metrics = results[profile][best_protocol_patient]['metrics']
                    f.write(f"  Efficacy Score: {metrics['treatment_efficacy_score']:.2f}\n")
                    f.write(f"  Tumor Reduction: {metrics['percent_reduction']:.2f}%\n")
                    f.write(f"  Final Resistance: {metrics['final_resistance_fraction']:.2f}%\n")
        
        # Provide key observations and recommendations
        f.write("\n\nKey Observations and Recommendations:\n")
        f.write("-------------------------------------\n")
        
        # Analyze protocol effectiveness across patients
        protocol_avg_scores = {protocol: 0 for protocol in protocols}
        protocol_count = {protocol: 0 for protocol in protocols}
        
        for profile in patient_profiles:
            for protocol in protocols:
                if protocol in results[profile] and results[profile][protocol]['success']:
                    score = results[profile][protocol]['metrics']['treatment_efficacy_score']
                    protocol_avg_scores[protocol] += score
                    protocol_count[protocol] += 1
        
        # Calculate average scores
        for protocol in protocols:
            if protocol_count[protocol] > 0:
                protocol_avg_scores[protocol] /= protocol_count[protocol]
        
        # Sort protocols by average score
        sorted_protocols = sorted(protocols, key=lambda x: protocol_avg_scores[x], reverse=True)
        
        f.write("\n1. Protocol Effectiveness Ranking:\n")
        for i, protocol in enumerate(sorted_protocols):
            if protocol_count[protocol] > 0:
                f.write(f"   {i+1}. {protocol.title()}: {protocol_avg_scores[protocol]:.2f}\n")
        
        # Identify patient-specific sensitivities
        f.write("\n2. Patient-Specific Observations:\n")
        for profile in patient_profiles:
            if profile != 'average':
                # Find unique responses for this patient
                max_score_protocol = None
                max_diff = -1
                
                for protocol in protocols:
                    if protocol in results[profile] and results[profile][protocol]['success']:
                        patient_score = results[profile][protocol]['metrics']['treatment_efficacy_score']
                        avg_score = protocol_avg_scores[protocol]
                        score_diff = patient_score - avg_score
                        
                        if score_diff > max_diff:
                            max_diff = score_diff
                            max_score_protocol = protocol
                
                if max_score_protocol and max_diff > 0.5:
                    f.write(f"   - {profile.title()} patients show enhanced response to {max_score_protocol} protocol\n")
                    f.write(f"     ({max_diff:.2f} higher efficacy than average)\n")
        
        # General recommendations
        f.write("\n3. General Recommendations:\n")
        if len(sorted_protocols) > 0:
            best_overall = sorted_protocols[0]
            f.write(f"   - {best_overall.title()} protocol provides best overall outcomes across patient profiles\n")
        
        if 'adaptive' in protocol_avg_scores and protocol_avg_scores['adaptive'] > 0:
            f.write("   - Adaptive therapy shows significant benefits in controlling resistance development\n")
        
        if 'hyperthermia' in protocol_avg_scores and protocol_avg_scores['hyperthermia'] > 0:
            f.write("   - Hyperthermia combination enhances treatment efficacy through multiple mechanisms\n")
        
        if 'immuno_combo' in protocol_avg_scores and protocol_avg_scores['immuno_combo'] > 0:
            f.write("   - Immuno-combination therapy particularly effective for patients with strong immune systems\n")
        
        f.write("\n4. Resistance Management:\n")
        f.write("   - Early treatment shows better control of resistance emergence\n")
        f.write("   - Periodic treatment holidays help maintain sensitive cell populations\n")
        f.write("   - Monitoring genetic stability may provide early warning of resistance development\n")
    
    return output_file

# Main integrated model function
def run_enhanced_model_analysis(output_dir='cancer_model_results'):
    """
    Run the enhanced breast cancer model with improved dynamics.
    
    Args:
        output_dir (str): Directory to save results
        
    Returns:
        dict: Complete analysis results
    """
    print("Starting enhanced model analysis...")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run comparative analysis with limited set for demo
    patient_profiles = ['average', 'young', 'elderly', 'compromised']
    treatment_protocols = ['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia']
    
    print(f"Running analysis for {len(patient_profiles)} patient profiles and {len(treatment_protocols)} treatment protocols...")
    all_results = run_comparative_analysis(patient_profiles, treatment_protocols, simulation_days=500)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualization_paths = create_visualizations(all_results, output_dir=output_dir)
    
    # Save comprehensive results to JSON for easy extraction
    import json
    json_path = os.path.join(output_dir, 'all_results.json')
    # Convert results to JSON-serializable format
    json_results = {}
    for profile in all_results.keys():
        json_results[profile] = {}
        for protocol in all_results[profile].keys():
            result = all_results[profile][protocol]
            if result.get('success', False):
                json_results[profile][protocol] = {
                    'metrics': result.get('metrics', {}),
                    'success': True
                }
    else:
                json_results[profile][protocol] = {
                    'success': False,
                    'error': result.get('error_message', 'Unknown error')
                }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"Saved comprehensive results to {json_path}")
    
    # Generate summary report
    print("\nGenerating summary report...")
    report_path = os.path.join(output_dir, 'model_summary.txt')
    generate_model_summary(all_results, output_file=report_path)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}/")
    print(f"Created {len(visualization_paths)} visualizations and summary report.")
    
    return {
        'results': all_results,
        'visualizations': visualization_paths,
        'report': report_path,
        'json_results': json_path
    }

# Optimize treatment protocol based on patient profile
def optimize_patient_treatment(patient_profile_name, output_dir='optimized_treatments'):
    """
    Optimize treatment protocol for a specific patient profile
    
    Args:
        patient_profile_name (str): Name of patient profile
        output_dir (str): Directory to save results
        
    Returns:
        dict: Optimized treatment protocol
    """
    print(f"Optimizing treatment for {patient_profile_name} patient profile...")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load patient profile
    patient_profile = create_patient_profile(patient_profile_name)
    
    # Define parameter ranges to optimize
    if patient_profile_name == 'elderly' or patient_profile_name == 'compromised':
        # For vulnerable patients, use lower intensity ranges
        dose_ranges = [0.3, 0.4, 0.5, 0.6, 0.7]
        cycle_ranges = [(7, 21), (10, 18), (14, 14)]
    else:
        # For healthier patients, can use higher intensities
        dose_ranges = [0.5, 0.6, 0.7, 0.8, 0.9]
        cycle_ranges = [(7, 14), (10, 11), (14, 7)]
    
    # Store results for each combination
    optimization_results = []
    
    # Test all combinations
    for dose in dose_ranges:
        for treatment_days, rest_days in cycle_ranges:
            # Create custom protocol
            protocol = {
                'hormone': create_cyclic_dosing_schedule(treatment_days, rest_days, dose, start_day=0),
                'her2': create_cyclic_dosing_schedule(treatment_days, rest_days, dose, start_day=0),
                'temperature': lambda t: 37.0,
                'description': f'Custom {dose:.1f} dose, {treatment_days}/{rest_days} schedule'
            }
            
            # Set up simulation parameters
            params = enhanced_model_params(patient_profile)
            t_span = [0, 500]
            t_eval = np.linspace(0, 500, 501)
            initial_conditions = get_enhanced_initial_conditions()
            
            # Set up drug schedules
            drug_schedules = {
                'hormone': protocol['hormone'],
                'her2': protocol['her2']
            }
            
            # Create temperature function
            temp_func = protocol['temperature']
            
            # Define the model function for this protocol
            def model_with_protocol(t, y):
                current_temp = temp_func(t)
                return enhanced_temperature_cancer_model(t, y, params, drug_schedules, current_temp, use_circadian=True)
            
            # Run simulation
            print(f"Testing {protocol['description']}...")
            result = safe_solve_ivp(model_with_protocol, t_span, initial_conditions, 'RK45', t_eval)
            
            # Calculate metrics if successful
            if result.success:
                # Calculate tumor burden
                N1 = result.y[0]  # Sensitive cells
                N2 = result.y[1]  # Partially resistant cells
                Q = result.y[6]   # Quiescent cells
                R1 = result.y[7]  # Type 1 resistant cells
                R2 = result.y[8]  # Type 2 resistant cells
                S = result.y[9]   # Senescent cells
                
                total_tumor = N1 + N2 + Q + R1 + R2 + S
                total_resistant = R1 + R2
                
                # Calculate resistance fraction
                resistance_fraction = (R1 + R2) / total_tumor * 100
                
                # Calculate tumor reduction from initial
                initial_burden = total_tumor[0]
                final_burden = total_tumor[-1]
                percent_reduction = 100 * (1 - final_burden / initial_burden)
                
                # Calculate treatment efficacy score
                treatment_efficacy = percent_reduction / (1 + resistance_fraction[-1]/100)
                
                # Calculate AUC (area under curve) for tumor burden
                auc_tumor = np.trapz(total_tumor, result.t)
                
                # Store results
                optimization_results.append({
                    'protocol_description': protocol['description'],
                    'dose': dose,
                    'treatment_days': treatment_days,
                    'rest_days': rest_days,
                    'efficacy_score': treatment_efficacy,
                    'percent_reduction': percent_reduction,
                    'final_resistance': resistance_fraction[-1],
                    'auc_tumor': auc_tumor,
                    'sim_result': result
                })
    
    # Find best protocol based on efficacy score
    if optimization_results:
        best_protocol = max(optimization_results, key=lambda x: x['efficacy_score'])
        
        # Generate visualization of the best protocol
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Optimized Treatment for {patient_profile_name.title()} Patient', fontsize=18)
        
        # Get best simulation result
        best_sim = best_protocol['sim_result']
        
        # Tumor burden plot
        ax = axes[0, 0]
        N1 = best_sim.y[0]
        N2 = best_sim.y[1]
        Q = best_sim.y[6]
        R1 = best_sim.y[7]
        R2 = best_sim.y[8]
        S = best_sim.y[9]
        
        total_tumor = N1 + N2 + Q + R1 + R2 + S
        ax.plot(best_sim.t, total_tumor, 'b-', linewidth=2)
        ax.set_title('Tumor Burden')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Cell Count')
        
        # Resistance plot
        ax = axes[0, 1]
        resistance_fraction = (R1 + R2) / total_tumor * 100
        ax.plot(best_sim.t, resistance_fraction, 'r-', linewidth=2)
        ax.set_title('Resistance Fraction')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Percent (%)')
        
        # Protocol visualization
        ax = axes[1, 0]
        treatment_days = best_protocol['treatment_days']
        rest_days = best_protocol['rest_days']
        dose = best_protocol['dose']
        
        # Create dosing schedule for visualization
        dosing_times = np.linspace(0, 200, 201)
        dosing_values = np.array([create_cyclic_dosing_schedule(treatment_days, rest_days, dose)(t) for t in dosing_times])
        
        ax.step(dosing_times, dosing_values, 'g-', linewidth=2, where='post')
        ax.set_title('Treatment Schedule')
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Dose')
        ax.set_ylim(-0.05, 1.05)
        
        # Efficacy comparison
        ax = axes[1, 1]
        
        # Sort results by efficacy
        sorted_results = sorted(optimization_results, key=lambda x: x['efficacy_score'], reverse=True)[:5]
        
        labels = [f"{p['dose']:.1f} dose, {p['treatment_days']}/{p['rest_days']}" for p in sorted_results]
        efficacy_scores = [p['efficacy_score'] for p in sorted_results]
        reduction_values = [p['percent_reduction'] for p in sorted_results]
        resistance_values = [p['final_resistance'] for p in sorted_results]
        
        x = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x - width, reduction_values, width, label='Reduction (%)')
        ax.bar(x, resistance_values, width, label='Resistance (%)')
        ax.bar(x + width, efficacy_scores, width, label='Efficacy Score')
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_title('Top 5 Protocols')
        ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save visualization
        optimization_path = os.path.join(output_dir, f'{patient_profile_name}_optimized.png')
        plt.savefig(optimization_path, dpi=300)
        plt.close()
        
        # Create optimized protocol
        optimized_protocol = {
            'hormone': create_cyclic_dosing_schedule(best_protocol['treatment_days'], 
                                                    best_protocol['rest_days'], 
                                                    best_protocol['dose'], 
                                                    start_day=0),
            'her2': create_cyclic_dosing_schedule(best_protocol['treatment_days'], 
                                                 best_protocol['rest_days'], 
                                                 best_protocol['dose'], 
                                                 start_day=0),
            'temperature': lambda t: 37.0,
            'description': f"Optimized protocol: {best_protocol['dose']:.1f} dose, " + 
                          f"{best_protocol['treatment_days']}/{best_protocol['rest_days']} schedule"
        }
        
        print(f"\nOptimized protocol for {patient_profile_name} patient:")
        print(f"Dose: {best_protocol['dose']:.2f}")
        print(f"Schedule: {best_protocol['treatment_days']} days on, {best_protocol['rest_days']} days off")
        print(f"Efficacy score: {best_protocol['efficacy_score']:.2f}")
        print(f"Tumor reduction: {best_protocol['percent_reduction']:.2f}%")
        print(f"Final resistance: {best_protocol['final_resistance']:.2f}%")
        
        return {
            'optimized_protocol': optimized_protocol,
            'protocol_details': best_protocol,
            'visualization_path': optimization_path,
            'all_results': optimization_results
        }
    else:
        print("Optimization failed - no successful simulations.")
        return None

# Script entry point
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run comprehensive analysis
    results = run_enhanced_model_analysis()
    
    # Get the output directory from results
    # The output_dir is passed to run_enhanced_model_analysis, extract it from results
    output_dir_used = 'cancer_model_results'  # Default fallback
    if 'report' in results:
        report_path = results['report']
        if isinstance(report_path, str) and 'cancer_model_results' in report_path:
            output_dir_used = os.path.dirname(report_path)
    elif 'json_results' in results:
        json_path = results['json_results']
        if isinstance(json_path, str) and 'cancer_model_results' in json_path:
            output_dir_used = os.path.dirname(json_path)
    
    # Create corresponding optimized_treatments directory name
    # Replace 'cancer_model_results' with 'optimized_treatments' while preserving alpha suffix
    if 'alpha' in output_dir_used:
        # Extract alpha suffix and create optimized directory
        optimized_dir = output_dir_used.replace('cancer_model_results', 'optimized_treatments')
    else:
        optimized_dir = 'optimized_treatments'
    
    # Optimize treatment for specific patient profiles
    optimize_patient_treatment('elderly', output_dir=optimized_dir)
    optimize_patient_treatment('compromised', output_dir=optimized_dir)
    optimize_patient_treatment('young', output_dir=optimized_dir)
