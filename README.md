# FOCTO

This repository contains the computational code for a fractional-order differential equation model of breast cancer treatment dynamics, incorporating tumor growth, immune response, drug pharmacokinetics, and treatment resistance.

## Overview

The model simulates cancer treatment dynamics using fractional-order differential equations (FODEs) to capture memory effects in biological systems. It includes:

- **Tumor compartments**: Sensitive cells, partially resistant cells, quiescent cells, senescent cells, and two types of fully resistant cells
- **Immune system**: Cytotoxic and regulatory immune cells with exhaustion dynamics
- **Pharmacokinetics/Pharmacodynamics**: Drug absorption, elimination, distribution, and effect modeling
- **Treatment protocols**: Standard, continuous, adaptive, immuno-combo, and hyperthermia protocols
- **Patient profiles**: Average, young, elderly, and compromised patient characteristics
- **Fractional-order dynamics**: Memory effects through fractional-order parameter α (0.75-1.0)

## Installation

### Requirements

- Python 3.7 or higher
- Required packages (see `requirements.txt`)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/ISJBTC/FOCTM.git
cd FOCTM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Test Installation (Recommended First Step)

Run a simple example to verify your installation:

```bash
python example_quickstart.py
```

This runs a quick simulation and demonstrates basic usage.

### Running the Main Model

To run a complete analysis with all patient profiles and treatment protocols:

```bash
python code.py
```

This will:
- Run simulations for 4 patient profiles (average, young, elderly, compromised)
- Test 5 treatment protocols (standard, continuous, adaptive, immuno_combo, hyperthermia)
- Generate visualizations and save results to `cancer_model_results/`
- Create optimized treatment recommendations for specific patients

### Running Sensitivity Analysis

To perform comprehensive parameter sensitivity analysis:

```bash
python comprehensive_sensitivity_analysis.py
```

This will:
- Analyze all 58 model parameters
- Test across all patient-protocol combinations (20 contexts)
- Generate sensitivity rankings and visualizations
- Save results to `sensitivity_analysis_results/`

### Running Analysis Scripts (Post-Processing)

**Note:** These scripts are for organizing and analyzing simulation results. They require that you have already run simulations and generated result files.

1. **Extract Results to Excel** (`extract_data_to_excel.py`):
   - Organizes simulation results from JSON files into Excel format
   - Creates pivot tables for easy analysis
   - Requires: JSON result files in `cancer_model_results_alpha_*/` directories
   - Output: `all_alpha_results_comprehensive.xlsx`

2. **Compare Fractional vs Integer-Order Models** (`compare_fractional_vs_integer_from_excel.py`):
   - Generates comparison statistics between fractional-order (α < 1.0) and integer-order (α = 1.0) models
   - Creates aggregate and context-specific comparison tables
   - Requires: `all_alpha_results_comprehensive.xlsx` (from step 1)
   - Output: CSV files with comparison statistics

**Workflow:**
```bash
# Step 1: Run simulations for different alpha values
python code.py  # (modify alpha parameter for each run, or run multiple times)

# Step 2: Extract results to Excel
python extract_data_to_excel.py

# Step 3: Compare fractional vs integer-order
python compare_fractional_vs_integer_from_excel.py
```

## Code Structure

### Core Simulation Code
- **`code.py`**: Main model implementation and simulation functions
- **`comprehensive_sensitivity_analysis.py`**: Parameter sensitivity analysis

### Analysis/Post-Processing Scripts
- **`extract_data_to_excel.py`**: Organizes simulation results into Excel format
- **`compare_fractional_vs_integer_from_excel.py`**: Compares fractional vs integer-order models

**Note:** The analysis scripts are optional post-processing tools. The core model (`code.py`) and sensitivity analysis (`comprehensive_sensitivity_analysis.py`) are sufficient to run all simulations and generate results.

## Main Functions

### Core Model Functions

#### `run_enhanced_model_analysis(output_dir='cancer_model_results')`
Runs the complete model analysis for all patient profiles and protocols.

**Returns:**
- Dictionary containing all simulation results, metrics, and file paths

**Example:**
```python
from code import run_enhanced_model_analysis
results = run_enhanced_model_analysis()
```

#### `run_comparative_analysis(patient_profiles, treatment_protocols, simulation_days=500)`
Runs comparative analysis across specified patient profiles and protocols.

**Parameters:**
- `patient_profiles`: List of profile names (e.g., `['average', 'young', 'elderly', 'compromised']`)
- `treatment_protocols`: List of protocol names (e.g., `['standard', 'continuous', 'adaptive']`)
- `simulation_days`: Number of days to simulate (default: 500)

**Returns:**
- Nested dictionary: `results[patient_profile][protocol]` containing simulation results

**Example:**
```python
from code import run_comparative_analysis
results = run_comparative_analysis(
    patient_profiles=['average', 'young'],
    treatment_protocols=['standard', 'continuous'],
    simulation_days=500
)
```

#### `run_enhanced_simulation(patient_profile_name, protocol_name, simulation_days=500, use_circadian=True)`
Runs a single simulation for a specific patient-protocol combination.

**Parameters:**
- `patient_profile_name`: One of 'average', 'young', 'elderly', 'compromised'
- `protocol_name`: One of 'standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia'
- `simulation_days`: Number of days to simulate
- `use_circadian`: Whether to include circadian rhythm effects

**Returns:**
- Dictionary with time series data, metrics, and success status

**Example:**
```python
from code import run_enhanced_simulation
result = run_enhanced_simulation('elderly', 'continuous', simulation_days=500)
```

### Patient and Protocol Creation

#### `create_patient_profile(profile_type='average')`
Creates patient-specific parameter modifications.

**Parameters:**
- `profile_type`: 'average', 'young', 'elderly', or 'compromised'

**Returns:**
- Dictionary of parameter modifications for the patient profile

#### `create_treatment_protocol(protocol_name='standard', patient_profile=None)`
Creates treatment protocol with drug schedules and temperature modulation.

**Parameters:**
- `protocol_name`: 'standard', 'continuous', 'adaptive', 'immuno_combo', or 'hyperthermia'
- `patient_profile`: Optional patient profile dictionary for adaptive protocols

**Returns:**
- Dictionary containing drug schedules, temperature protocol, and protocol-specific parameters

### Sensitivity Analysis

#### `run_comprehensive_sensitivity_analysis(output_dir, patient_profiles, protocols, variation_levels, simulation_days)`
Performs one-at-a-time (OAT) local sensitivity analysis.

**Parameters:**
- `output_dir`: Directory to save results (default: 'sensitivity_analysis_results')
- `patient_profiles`: List of patient profiles to analyze
- `protocols`: List of treatment protocols to analyze
- `variation_levels`: Parameter variation levels (e.g., `[-0.2, -0.1, 0.1, 0.2]`)
- `simulation_days`: Number of days to simulate

**Example:**
```python
from comprehensive_sensitivity_analysis import run_comprehensive_sensitivity_analysis
run_comprehensive_sensitivity_analysis(
    output_dir='sensitivity_results',
    patient_profiles=['average', 'young', 'elderly', 'compromised'],
    protocols=['standard', 'continuous', 'adaptive', 'immuno_combo', 'hyperthermia'],
    variation_levels=[-0.2, -0.1, 0.1, 0.2],
    simulation_days=500
)
```

## Model Parameters

The model includes 58 parameters categorized as:

- **Experimental** (11): Derived from laboratory measurements (e.g., immune cytotoxicity assays, tumor doubling time)
- **Clinical** (15): From clinical trials and patient data (e.g., treatment schedules, pharmacokinetics)
- **Literature-based** (20): From published studies (e.g., growth rates, resistance mechanisms)
- **Hypothetical** (7): Estimated based on biological plausibility (e.g., protocol-specific resistance modifiers)

Key parameters include:
- `alpha`: Fractional order (0.75-1.0), controls memory effects
- `beta1`: Immune cytotoxic killing rate (0.005 d⁻¹)
- `lambda1`, `lambda2`: Tumor growth rates
- `omega_R1`, `omega_R2`: Resistance development rates
- `K`: Carrying capacity (1000 cells)

See `code.py` function `enhanced_model_params()` for complete parameter list with default values.

## File Structure

```
FOCTM/
├── code.py                                    # Main model and simulation code
├── comprehensive_sensitivity_analysis.py     # Parameter sensitivity analysis
├── extract_data_to_excel.py                  # Post-processing: organize results to Excel
├── compare_fractional_vs_integer_from_excel.py  # Post-processing: fractional vs integer comparison
├── example_quickstart.py                     # Quick start example script
├── requirements.txt                           # Python dependencies
├── README.md                                  # This file
├── LICENSE                                    # MIT License
├── CITATION.cff                               # Citation metadata
└── .gitignore                                 # Git ignore rules
```

## Output Files

### Main Model Outputs (`cancer_model_results/`)

- `all_results.json`: Complete simulation results in JSON format
- `model_summary.txt`: Text summary of key metrics
- `efficacy_metrics.png`: Treatment efficacy comparison
- `protocol_comparison.png`: Protocol performance visualization
- `patient_comparison.png`: Patient-specific response comparison
- `treatment_heatmaps.png`: Heat maps of treatment outcomes
- `detailed_adaptive_analysis.png`: Adaptive protocol analysis

### Sensitivity Analysis Outputs (`sensitivity_analysis_results/`)

- `sensitivity_summary_table.csv`: Ranked parameter sensitivity
- `complete_sensitivity_results.json`: Full sensitivity data
- `sensitivity_heatmap.png`: Visual sensitivity ranking
- `context_specific_sensitivity.csv`: Context-dependent sensitivity
- `parameter_category_analysis.csv`: Sensitivity by parameter category

## Model Structure

The model uses a system of fractional-order differential equations with 15 state variables:

1. **N₁**: Sensitive tumor cells
2. **N₂**: Partially resistant tumor cells
3. **I₁**: Cytotoxic immune cells
4. **I₂**: Regulatory immune cells
5. **P**: Metastatic potential
6. **A**: Angiogenesis factor
7. **Q**: Quiescent cells
8. **R₁**: Type 1 resistant cells
9. **R₂**: Type 2 resistant cells
10. **S**: Senescent cells
11. **D**: Drug concentration
12. **Dₘ**: Metabolized drug
13. **G**: Genetic stability
14. **M**: Metabolism status
15. **H**: Hypoxia level

Fractional-order effects are implemented through a time-dependent scaling factor that captures memory-dependent dynamics, where past states influence current evolution with power-law decay.

## Numerical Methods

- **Solver**: `scipy.integrate.solve_ivp` with adaptive step size
- **Methods**: RK45 (default), BDF, Radau, DOP853 (fallback options)
- **Tolerance**: Adaptive (rtol=1e-4 to 1e-6, atol=1e-7 to 1e-9)
- **Error Handling**: Automatic fallback to alternative solvers and tolerances

## Reproducibility

To reproduce the results from the manuscript:

1. **Main Simulations**: Run `python code.py` (uses default random seed: 42)
2. **Sensitivity Analysis**: Run `python comprehensive_sensitivity_analysis.py`
3. **Fractional-Order Comparison**: To compare different fractional orders, modify the `alpha` parameter in `enhanced_model_params()`:
   ```python
   params = enhanced_model_params()
   params['alpha'] = 0.75  # or 0.80, 0.85, 0.90, 0.93, 0.95, 1.0
   ```
   Run simulations for each α value (0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 1.0) to generate the comparison data.

All simulations use fixed random seeds for reproducibility. The model parameters are defined in `enhanced_model_params()` and can be modified for different scenarios.

**Note on Fractional Order (α)**: The default α value in `code.py` is 0.93. To run simulations with different α values, modify the `alpha` parameter in the `enhanced_model_params()` function or pass it as part of a patient profile. The model supports α values from 0.75 to 1.0, where α=1.0 represents integer-order (memoryless) dynamics.

## Computational Requirements

- **Memory**: ~2-4 GB RAM for full analysis
- **Time**: 
  - Single simulation: ~1-5 seconds
  - Full comparative analysis: ~5-10 minutes
  - Complete sensitivity analysis: ~2-4 hours (4,640 simulations)

## Citation

If you use this code, please cite:

**Software Citation:**
```
To be added
```

**Repository:**
- https://github.com/ISJBTC/FOCTO.git

**Code Availability:**

The source code supporting this study is publicly available via the Open Science Framework at:

https://github.com/ISJBTC/FOCTO.git

If you use this code in your research, please also cite the associated manuscript (citation will be added upon publication).

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please contact [jamadarirshad@gmail.com]

## Acknowledgments

This model incorporates parameters and mechanisms from multiple published studies on cancer dynamics, immune response, and treatment resistance. See the manuscript for complete references.

