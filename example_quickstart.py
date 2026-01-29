"""
Quick Start Example: Fractional-Order Cancer Treatment Model

This script demonstrates a simple example of running the model for a single
patient-protocol combination. This is useful for testing the installation
and understanding basic usage.
"""

from code import run_enhanced_simulation, enhanced_model_params

def main():
    print("="*80)
    print("Quick Start Example: Fractional-Order Cancer Treatment Model")
    print("="*80)
    
    # Example 1: Run a single simulation
    print("\n1. Running simulation for 'average' patient with 'standard' protocol...")
    result = run_enhanced_simulation(
        patient_profile_name='average',
        protocol_name='standard',
        simulation_days=100,  # Shorter simulation for quick test
        use_circadian=True
    )
    
    if result.get('success', False):
        metrics = result.get('metrics', {})
        print(f"\n✓ Simulation completed successfully!")
        print(f"\nResults:")
        print(f"  - Tumor Reduction: {metrics.get('tumor_reduction', 0):.2f}%")
        print(f"  - Final Resistance: {metrics.get('final_resistance', 0):.2f}%")
        print(f"  - Efficacy Score: {metrics.get('efficacy_score', 0):.2f}")
        print(f"  - Final Tumor Size: {metrics.get('final_tumor_size', 0):.2f}")
    else:
        print(f"\n✗ Simulation failed: {result.get('error_message', 'Unknown error')}")
    
    # Example 2: Modify fractional order parameter
    print("\n" + "="*80)
    print("2. Testing different fractional order (alpha) values...")
    print("="*80)
    
    for alpha in [0.85, 0.93, 1.0]:
        # Get default parameters and modify alpha
        params = enhanced_model_params()
        params['alpha'] = alpha
        
        print(f"\n  Testing alpha = {alpha}...")
        # Note: In a full implementation, you would pass params to the simulation
        # This is just a demonstration of parameter modification
        print(f"    Alpha parameter set to {alpha}")
        if alpha == 1.0:
            print(f"    (Integer-order model - no memory effects)")
        else:
            print(f"    (Fractional-order model - memory effects present)")
    
    print("\n" + "="*80)
    print("Quick start example completed!")
    print("="*80)
    print("\nFor full analysis, run: python code.py")
    print("For sensitivity analysis, run: python comprehensive_sensitivity_analysis.py")

if __name__ == "__main__":
    main()

