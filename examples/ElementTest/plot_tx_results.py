#!/usr/bin/env python
"""
Triaxial Test Results Visualization Script

This script creates 4 subplots to visualize simulation results and compare with reference data:
1. p-q evolution in p-q space
2. Axial strain vs Deviatoric stress
3. void_ratio-log(p)
4. Volumetric strain vs axial strain

Usage:
------
Programmatic:
    from plot_tx_results import plot_triaxial_results
    plot_triaxial_results("/path/to/simulation", end_num=200)

    Or with the main create_plots function:
    from plot_tx_results import create_plots
    create_plots(base_path="/path/to/sim", data_path="/path/to/sim/100pa/particles")

Direct execution (modify parameters in main() function):
    python3 plot_tx_results.py
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os

# ================= UTILITY FUNCTIONS =================
def meanstress(stress):
    """Calculate mean stress (compression positive)"""
    return -(stress[0] + stress[1] + stress[2]) / 3.
    
def equistress(stress):
    """Calculate equivalent deviatoric stress (q)"""
    return math.sqrt(3.0 * (((stress[0] - stress[1]) * (stress[0] - stress[1]) +
                            (stress[1] - stress[2]) * (stress[1] - stress[2]) +
                            (stress[0] - stress[2]) * (stress[0] - stress[2])) / 6.0 +
                            stress[3] * stress[3] + stress[4] * stress[4] + stress[5] * stress[5]))

def volumetric_strain(strain):
    """Calculate volumetric strain"""
    return strain[0] + strain[1] + strain[2]

def axial_strain(strain):
    """Calculate axial strain (assuming z-direction is axial)"""
    return strain[2]

# ================= DATA LOADING =================
def load_simulation_data(data_path, particle_id=0, start_num=0, end_num=150):
    """Load simulation data from NPZ files"""
    p_data = []
    q_data = []
    e_data = []
    p_i_data = []
    strain_vol_data = []
    strain_axial_data = []
    
    for print_num in range(start_num, end_num):
        try:
            filename = f'MPMParticle{print_num:06d}.npz'
            filepath = os.path.join(data_path, filename)
            data = np.load(filepath, allow_pickle=True)
            
            stress = data['stress'][particle_id]
            strain = data['strain'][particle_id]
            state_vars = data['state_vars'].item()
            
            p = meanstress(stress)
            q = equistress(stress)
            void_ratio = state_vars['void_ratio'][particle_id]
            p_i = state_vars['p_i'][particle_id]
            strain_vol = volumetric_strain(strain)
            strain_ax = axial_strain(strain)
            
            p_data.append(p)
            q_data.append(q)
            e_data.append(void_ratio)
            p_i_data.append(p_i)
            strain_vol_data.append(strain_vol)
            strain_axial_data.append(strain_ax)
            
        except FileNotFoundError:
            print(f"Warning: File {filename} not found")
            continue
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    return np.array(p_data), np.array(q_data), np.array(e_data), np.array(p_i_data), \
           np.array(strain_vol_data), np.array(strain_axial_data)

def load_reference_data(ref_data_path):
    """Load reference data from CSV file if it exists"""
    ref_data = {
        'p': np.array([]),
        'q': np.array([]),
        'eps_vol': np.array([]),
        'eps1': np.array([]),
        'e': np.array([])
    }
    
    if os.path.exists(ref_data_path):
        try:
            # Read the CSV file
            df = pd.read_csv(ref_data_path)
            
            # Extract columns (based on the CSV structure we saw)
            ref_data['p'] = df['p'].values if 'p' in df.columns else np.array([])
            ref_data['q'] = df['q'].values if 'q' in df.columns else np.array([])
            ref_data['eps_vol'] = df['eps_vol'].values * 100 if 'eps_vol' in df.columns else np.array([])  # Convert to %
            ref_data['eps1'] = df['eps1'].values * 100 if 'eps1' in df.columns else np.array([])  # Convert to %
            ref_data['e'] = df['e'].values if 'e' in df.columns else np.array([])
            
            print(f"Reference data loaded successfully from {ref_data_path}")
            print(f"Number of reference data points: {len(ref_data['p'])}")
            
        except Exception as e:
            print(f"Error loading reference data: {e}")
    else:
        print(f"Reference data file not found: {ref_data_path}")
    
    return ref_data

# ================= PLOTTING =================
def create_plots(base_path, data_path, ref_data_path=None, particle_id=0, start_num=0, end_num=150, 
                output_filename="simulation_results.png", figure_size=(15, 12), save_dpi=150):
    """Create the four subplot figure"""
    # Load data
    p_data, q_data, e_data, p_i_data, strain_vol_data, strain_axial_data = load_simulation_data(
        data_path, particle_id, start_num, end_num)
    
    # Load reference data if path is provided
    ref_data = {}
    if ref_data_path and os.path.exists(ref_data_path):
        ref_data = load_reference_data(ref_data_path)
    else:
        ref_data = {
            'p': np.array([]), 'q': np.array([]), 'eps_vol': np.array([]), 
            'eps1': np.array([]), 'e': np.array([])
        }
    
    if len(p_data) == 0:
        print("No simulation data loaded. Check file paths and particle files.")
        return
    
    print(f"Simulation data loaded: {len(p_data)} data points")
    
    # Convert units for better display
    p_data_kPa = p_data   # Convert Pa to kPa
    q_data_kPa = q_data   # Convert Pa to kPa
    strain_axial_percent = np.abs(strain_axial_data) * 100  # Convert to percentage
    strain_vol_percent = np.abs(strain_vol_data) * 100     # Convert to percentage
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figure_size)
    
    # ================= Plot 1: p-q evolution =================
    ax1.plot(p_data_kPa, q_data_kPa, 'b-', linewidth=2, alpha=0.7, label='Simulation')
    ax1.scatter(p_data_kPa[0], q_data_kPa[0], c='green', s=80, marker='o', label='Initial state', zorder=5)
    ax1.scatter(p_data_kPa[-1], q_data_kPa[-1], c='red', s=80, marker='o', label='Final state', zorder=5)
    
    # Add reference data if available
    if len(ref_data['p']) > 0 and len(ref_data['q']) > 0:
        ax1.plot(ref_data['p'], ref_data['q'], 'r--', linewidth=2, alpha=0.8, label='Reference')
    
    ax1.set_xlabel('Mean stress p (kPa)')
    ax1.set_ylabel('Deviatoric stress q (kPa)')
    ax1.set_title('Stress Path in p-q Space')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ================= Plot 2: Axial strain vs Deviatoric stress =================
    ax2.plot(strain_axial_percent, q_data_kPa, 'b-', linewidth=2, alpha=0.7, label='Simulation')
    
    # Add reference data if available
    if len(ref_data['eps1']) > 0 and len(ref_data['q']) > 0:
        ax2.plot(ref_data['eps1'], ref_data['q'], 'r--', linewidth=2, alpha=0.8, label='Reference')
    
    ax2.set_xlabel('Axial strain (%)')
    ax2.set_ylabel('Deviatoric stress q (kPa)')
    ax2.set_title('Deviatoric Stress vs Axial Strain')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ================= Plot 3: void_ratio-log(p) =================
    log_p_data = np.log(p_data_kPa)  # log of p in kPa
    ax3.plot(log_p_data, e_data, 'b-', linewidth=2, alpha=0.7, label='Simulation path')
    ax3.scatter(log_p_data[0], e_data[0], c='green', s=80, marker='o', label='Initial state', zorder=5)
    ax3.scatter(log_p_data[-1], e_data[-1], c='red', s=80, marker='o', label='Final state', zorder=5)
    
    # Add reference data if available
    if len(ref_data['p']) > 0 and len(ref_data['e']) > 0:
        log_p_ref = np.log(ref_data['p'])
        ax3.plot(log_p_ref, ref_data['e'], 'r--', linewidth=2, alpha=0.8, label='Reference')
    
    ax3.set_xlabel('ln(p/1kPa)')
    ax3.set_ylabel('Void ratio e')
    ax3.set_title('Void Ratio vs Mean Stress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ================= Plot 4: Volumetric strain vs axial strain =================
    ax4.plot(strain_axial_percent, strain_vol_percent, 'b-', linewidth=2, alpha=0.7, label='Simulation')
    
    # Add reference data if available
    if len(ref_data['eps1']) > 0 and len(ref_data['eps_vol']) > 0:
        ax4.plot(ref_data['eps1'], ref_data['eps_vol'], 'r--', linewidth=2, alpha=0.8, label='Reference')
    
    ax4.set_xlabel('Axial strain (%)')
    ax4.set_ylabel('Volumetric strain (%)')
    ax4.set_title('Volumetric Strain vs Axial Strain')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(base_path, output_filename)
    plt.savefig(output_path, dpi=save_dpi, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    
    # Display data summary
    print("\n=== DATA SUMMARY ===")
    print(f"Simulation data points: {len(p_data)}")
    print(f"Mean stress range: {p_data_kPa.min():.2f} - {p_data_kPa.max():.2f} kPa")
    print(f"Deviatoric stress range: {q_data_kPa.min():.2f} - {q_data_kPa.max():.2f} kPa")
    print(f"Axial strain range: {strain_axial_percent.min():.3f} - {strain_axial_percent.max():.3f} %")
    print(f"Volumetric strain range: {strain_vol_percent.min():.3f} - {strain_vol_percent.max():.3f} %")
    print(f"Void ratio range: {e_data.min():.4f} - {e_data.max():.4f}")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass

def plot_triaxial_results(base_path, data_subfolder="100kpa/particles", ref_data_file="ref_results.csv", 
                         particle_id=0, start_num=0, end_num=150, output_filename="simulation_results.png",
                         figure_size=(15, 12), save_dpi=150, include_reference=True):
    """
    Convenience function to plot triaxial results with direct arguments.
    
    Parameters:
    -----------
    base_path : str
        Base path to the simulation directory
    data_subfolder : str
        Subfolder containing particle data (relative to base_path)
    ref_data_file : str
        Reference data CSV file (relative to base_path)
    particle_id : int
        Particle ID to analyze
    start_num : int
        Starting file number
    end_num : int
        Ending file number
    output_filename : str
        Output figure filename
    figure_size : tuple
        Figure size (width, height) in inches
    save_dpi : int
        Figure DPI for saving
    include_reference : bool
        Whether to include reference data
    """
    # Construct paths
    data_path = os.path.join(base_path, data_subfolder)
    ref_data_path = os.path.join(base_path, ref_data_file) if include_reference else None
    
    print("=== Triaxial Test Results Visualization ===")
    print(f"Base path: {base_path}")
    print(f"Data path: {data_path}")
    print(f"Reference data: {ref_data_path if ref_data_path else 'None'}")
    print(f"Particle ID: {particle_id}")
    print(f"File range: {start_num} to {end_num}")
    print(f"Output file: {output_filename}")
    print()
    
    # Create plots
    create_plots(
        base_path=base_path,
        data_path=data_path,
        ref_data_path=ref_data_path,
        particle_id=particle_id,
        start_num=start_num,
        end_num=end_num,
        output_filename=output_filename,
        figure_size=figure_size,
        save_dpi=save_dpi
    )

def main():
    """
    Main function with hardcoded parameters that can be easily modified.
    Edit the parameters below to customize the analysis.
    """
    
    # ================= MODIFY THESE PARAMETERS =================
    # File paths and settings
    base_path = "/home/yj/works/GeoTaichi_Yihao_v1/examples/ElementTest/UndrainedNorSand"
    data_subfolder = "100kpa/particles"
    ref_data_file = "ref_results.csv"
    
    # Analysis parameters
    particle_id = 0        # which particle to analyze
    start_num = 0          # starting file number
    end_num = 150          # ending file number
    
    # Output settings
    output_filename = "simulation_results.png"
    figure_width = 15      # figure width in inches
    figure_height = 12     # figure height in inches
    save_dpi = 150         # DPI for saving
    include_reference = True  # whether to include reference data
    
    # ================= END PARAMETERS =================
    
    # Use the convenience function
    plot_triaxial_results(
        base_path=base_path,
        data_subfolder=data_subfolder,
        ref_data_file=ref_data_file,
        particle_id=particle_id,
        start_num=start_num,
        end_num=end_num,
        output_filename=output_filename,
        figure_size=(figure_width, figure_height),
        save_dpi=save_dpi,
        include_reference=include_reference
    )

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    main()
