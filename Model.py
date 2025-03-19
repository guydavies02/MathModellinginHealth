import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec

# Parameters for the Hasty 2002 model
def hasty_model(t, state, params):
    
    x, y = state
    gamma_x = params['gamma_x']
    gamma_y = params['gamma_y']
    alpha = params['alpha']
    sigma = params['sigma']
    tau_y = params['tau_y']
    
    # Common term for both equations
    common_term = (1 + x**2 + alpha*sigma*x**4) / ((1 + x**2 + sigma*x**4) * (1 + y**4))
    
    # Derivatives
    dx_dt = common_term - gamma_x * x
    dy_dt = (1/tau_y)*(common_term - gamma_y * y)
    
    return [dx_dt, dy_dt]

def simulate_hasty_model(params, t_span, initial_conditions, n_points=1000):

    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        hasty_model,
        t_span,
        initial_conditions,
        args=(params,),
        method='RK45',
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9
    )
    
    return sol.t, sol.y

def plot_time_series(t, states, params, title="Hasty 2002 Synthetic Gene Oscillator"):

    # Create a figure with 2 subplots
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    
    # Time series plot
    ax1 = plt.subplot(gs[0])
    ax1.plot(t, states[0], 'b-', linewidth=2, label='CI (x)')
    ax1.plot(t, states[1], 'r-', linewidth=2, label='Lac (y)')
    ax1.set_xlabel('Dimensionless Time', fontsize=12)
    ax1.set_ylabel('Concentration (Dimensionless)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Parameter information
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    param_text = f"""
    Model Parameters:
    γx = {params['gamma_x']:.3f}
    γy = {params['gamma_y']:.3f}
    alpha = {params['alpha']:.1f}
    sigma = {params['sigma']:.1f}
    τy = {params['tau_y']:.1f}
    
    Corresponding to:
    kdx = {params['gamma_x']*20:.2f} min⁻¹
    kdy = {params['gamma_y']*21:.2f} min⁻¹
    
    Initial conditions: x(0) = {states[0][0]:.2f}, y(0) = {states[1][0]:.2f}
    
    Notes:
    - CI activation parameter (alpha) represents the degree of transcription increase when CI dimer binds to OR2
    - Relative binding affinity (sigma) is the affinity for CI dimer binding to OR2 relative to binding at OR1
    - The degradation rates (γx, γy) can be manipulated experimentally
    """
    ax2.text(0.05, 0.95, param_text, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def parameter_variation_study(base_params, vary_param, values, t_span, initial_conditions, n_points=1000):

    results = {}
    
    for val in values:
        params = base_params.copy()
        params[vary_param] = val
        t, states = simulate_hasty_model(params, t_span, initial_conditions, n_points)
        results[val] = (t, states)
    
    return results

def plot_parameter_variation(results, vary_param, title=None):

    fig = plt.figure(figsize=(12, 8))
    
    for val, (t, states) in results.items():
        plt.plot(t, states[0], label=f'{vary_param} = {val:.3f}')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('CI Concentration (x)', fontsize=12)
    plt.title(title or f'Effect of {vary_param} on CI Concentration', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    # Default parameters based on the paper (Table 1 and text)
    default_params = {
        'gamma_x': 0.105,  # Degradation rate for CI
        'gamma_y': 0.036,  # Degradation rate for Lac
        'alpha': 11,           # Activation parameter
        'sigma': 2,            # Relative binding affinity
        'tau_y': 5         # Time constant for Lac
    }
    
    t_span = (0, 1000)
    initial_conditions = [0.1, 0.1]
    
    t, states = simulate_hasty_model(default_params, t_span, initial_conditions)
    fig1 = plot_time_series(t, states, default_params)
    
    gamma_y_values = [0.03, 0.035, 0.04, 0.045, 0.05]
    gamma_y_results = parameter_variation_study(
        default_params, 'gamma_y', gamma_y_values, t_span, initial_conditions
    )
    fig2 = plot_parameter_variation(gamma_y_results, 'gamma_y', 'Effect of γy on CI Concentration')
    
    alpha_values = [8, 10, 12, 14, 16]
    alpha_results = parameter_variation_study(
        default_params, 'alpha', alpha_values, t_span, initial_conditions
    )
    fig3 = plot_parameter_variation(alpha_results, 'alpha', 'Effect of alpha on CI Concentration')

    sigma_values = [1, 1.5, 2, 2.5, 3]
    sigma_results = parameter_variation_study(
        default_params, 'sigma', sigma_values, t_span, initial_conditions
    )
    fig4 = plot_parameter_variation(sigma_results, 'sigma', 'Effect of sigma on CI Concentration')

    gamma_x_values = [0.09, 0.1, 0.11, 0.12, 0.13]
    gamma_x_results = parameter_variation_study(
        default_params, 'gamma_x', gamma_x_values, t_span, initial_conditions
    )
    fig5 = plot_parameter_variation(gamma_x_results, 'gamma_x', 'Effect of γx on CI Concentration')

    fig1.savefig('plots/time_series.png', dpi=300, bbox_inches='tight')
    fig2.savefig('plots/gamma_y_variation.png', dpi=300, bbox_inches='tight')
    fig3.savefig('plots/alpha_variation.png', dpi=300, bbox_inches='tight')
    fig4.savefig('plots/sigma_variation.png', dpi=300, bbox_inches='tight')
    fig5.savefig('plots/gamma_x_variation.png', dpi=300, bbox_inches='tight')

    # Phase-plane plot
    def plot_phase_plane(t, states, params, title="Phase Plane of Hasty 2002 Model"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(states[0], states[1], 'b-', linewidth=2, label='Trajectory')
        ax.set_xlabel('CI Concentration (x)', fontsize=12)
        ax.set_ylabel('Lac Concentration (y)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.show()
        return fig

    fig6 = plot_phase_plane(t, states, default_params)
    fig6.savefig('phase_plane.png', dpi=300, bbox_inches='tight')