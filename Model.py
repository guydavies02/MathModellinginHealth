import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec

# Parameters for the Hasty 2002 model
def hasty_model(t, state, params):
    """
    Dimensionless model equations for the Hasty 2002 synthetic gene oscillator.
    
    Equations:
    dx/dt = [(1 + x^2 + a*s*x^4)/(1 + x^2 + s*x^4)*(1 + y^4)] - gamma_x * x
    dy/dt = [(1 + x^2 + a*s*x^4)/(1 + x^2 + s*x^4)*(1 + y^4)] - gamma_y * y
    
    Parameters:
    ----------
    t : float
        Time
    state : array
        Current state [x, y]
    params : dict
        Dictionary of parameters
    
    Returns:
    --------
    array
        Derivatives [dx/dt, dy/dt]
    """
    x, y = state
    gamma_x = params['gamma_x']
    gamma_y = params['gamma_y']
    a = params['a']
    s = params['s']
    
    # Common term for both equations
    common_term = (1 + x**2 + a*s*x**4) / ((1 + x**2 + s*x**4) * (1 + y**4))
    
    # Derivatives
    dx_dt = common_term - gamma_x * x
    dy_dt = common_term - gamma_y * y
    
    return [dx_dt, dy_dt]

def simulate_hasty_model(params, t_span, initial_conditions, n_points=1000):
    """
    Simulate the Hasty 2002 model with given parameters.
    
    Parameters:
    ----------
    params : dict
        Dictionary of parameters
    t_span : tuple
        (t_start, t_end)
    initial_conditions : array
        [x0, y0]
    n_points : int
        Number of time points to output
    
    Returns:
    --------
    tuple
        (t, states)
    """
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
    """
    Plot time series for the Hasty 2002 model.
    
    Parameters:
    ----------
    t : array
        Time points
    states : array
        States [x(t), y(t)]
    params : dict
        Dictionary of parameters
    title : str
        Plot title
    """
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
    a = {params['a']:.1f}
    s = {params['s']:.1f}
    
    Corresponding to:
    kdx = {params['gamma_x']*20:.2f} min⁻¹
    kdy = {params['gamma_y']*21:.2f} min⁻¹
    
    Initial conditions: x(0) = {states[0][0]:.2f}, y(0) = {states[1][0]:.2f}
    
    Notes:
    - CI activation parameter (a) represents the degree of transcription increase when CI dimer binds to OR2
    - Relative binding affinity (s) is the affinity for CI dimer binding to OR2 relative to binding at OR1
    - The degradation rates (γx, γy) can be manipulated experimentally
    """
    ax2.text(0.05, 0.95, param_text, fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def parameter_variation_study(base_params, vary_param, values, t_span, initial_conditions, n_points=1000):
    """
    Perform parameter variation study.
    
    Parameters:
    ----------
    base_params : dict
        Base parameters
    vary_param : str
        Parameter to vary
    values : list
        List of values for the parameter
    t_span, initial_conditions, n_points:
        Same as in simulate_hasty_model
    
    Returns:
    --------
    dict
        Results for each parameter value
    """
    results = {}
    
    for val in values:
        params = base_params.copy()
        params[vary_param] = val
        t, states = simulate_hasty_model(params, t_span, initial_conditions, n_points)
        results[val] = (t, states)
    
    return results

def plot_parameter_variation(results, vary_param, title=None):
    """
    Plot results from parameter variation study.
    
    Parameters:
    ----------
    results : dict
        Results from parameter_variation_study
    vary_param : str
        Parameter that was varied
    title : str
        Plot title
    """
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
        'a': 11,           # Activation parameter
        's': 2             # Relative binding affinity
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
    
    a_values = [8, 10, 12, 14, 16]
    a_results = parameter_variation_study(
        default_params, 'a', a_values, t_span, initial_conditions
    )
    fig3 = plot_parameter_variation(a_results, 'a', 'Effect of a on CI Concentration')

    s_values = [1, 1.5, 2, 2.5, 3]
    s_results = parameter_variation_study(
        default_params, 's', s_values, t_span, initial_conditions
    )
    fig4 = plot_parameter_variation(s_results, 's', 'Effect of s on CI Concentration')

    gamma_x_values = [0.09, 0.1, 0.11, 0.12, 0.13]
    gamma_x_results = parameter_variation_study(
        default_params, 'gamma_x', gamma_x_values, t_span, initial_conditions
    )
    fig5 = plot_parameter_variation(gamma_x_results, 'gamma_x', 'Effect of γx on CI Concentration')

    fig1.savefig('time_series.png', dpi=300, bbox_inches='tight')
    fig2.savefig('gamma_y_variation.png', dpi=300, bbox_inches='tight')
    fig3.savefig('a_variation.png', dpi=300, bbox_inches='tight')
    fig4.savefig('s_variation.png', dpi=300, bbox_inches='tight')
    fig5.savefig('gamma_x_variation.png', dpi=300, bbox_inches='tight')
