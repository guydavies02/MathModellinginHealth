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

    fig = plt.figure(figsize=(8, 5))   
    # Time series plot
    plt.plot(t, states[0], 'b-', linewidth=2, label='CI (x)')
    plt.plot(t, states[1], 'r-', linewidth=2, label='Lac (y)')
    plt.xlabel('Dimensionless Time', fontsize=14)
    plt.ylabel('Concentration (Dimensionless)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.title(title, fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
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

    fig = plt.figure(figsize=(8, 5))
    
    for val, (t, states) in results.items():
        plt.plot(t, states[0], label=f'$\{vary_param}$ = {val:.3f}')
    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('CI Concentration (x)', fontsize=14)
    plt.title(title or f'Effect of $\{vary_param}$ on CI Concentration', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.legend(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_phase_portrait_with_nullclines(params, t_span, initial_conditions_list, title="Phase Portrait with Nullclines"):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Generate nullclines
    x = np.linspace(0, 4, 1000)
    y = np.linspace(0, 4, 1000)
    X, Y = np.meshgrid(x, y)
    
    gamma_x = params['gamma_x']
    gamma_y = params['gamma_y']
    alpha = params['alpha']
    sigma = params['sigma']
    tau_y = params['tau_y']
    
    common_term = (1 + X**2 + alpha * sigma * X**4) / ((1 + X**2 + sigma * X**4) * (1 + Y**4))
    dx_dt = common_term - gamma_x * X
    dy_dt = (1 / tau_y) * (common_term - gamma_y * Y)
    
    
    # Plot trajectories for different initial conditions
    for initial_conditions in initial_conditions_list:
        t, states = simulate_hasty_model(params, t_span, initial_conditions)
        ax.plot(states[0], states[1], 'b-', linewidth=1)
        ax.quiver(states[0][:-1], states[1][:-1], 
                    np.diff(states[0]), np.diff(states[1]), 
                    scale_units='xy', angles='xy', scale=1, color='b', alpha=0.6)
        
    # Nullclines
    ax.contour(X, Y, dx_dt, levels=[0], colors='r', linestyles='--', linewidths=1.5, label='dx/dt = 0')
    ax.contour(X, Y, dy_dt, levels=[0], colors='g', linestyles='--', linewidths=1.5, label='dy/dt = 0')
    
    # Labels and legend
    ax.set_xlabel('CI Concentration (x)', fontsize=12)
    ax.set_ylabel('Lac Concentration (y)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(['dx/dt = 0', 'dy/dt = 0'], fontsize=10)
    plt.savefig('hasty_phase_portrait.png', dpi=300)
    plt.tight_layout()
    plt.show()
    return fig


if __name__ == "__main__":

    default_params = {
        'gamma_x': 0.105,  # Degradation rate for CI
        'gamma_y': 0.033,  # Degradation rate for Lac
        'alpha': 11,           # Activation parameter
        'sigma': 2,            # Relative binding affinity
        'tau_y': 5         # Time constant for Lac
    }
    t_span=(0, 1000)

    initial_conditions_list = [
        [0.5, 1], [1.0, 1.0], [1.5, 1.5], [2.0, 2.0], [2.5, 2.5], [3.0, 3.0],
        [0.8, 3.5], [1.2, 2.8], [2.0, 3.5], [0.3, 1.2], [0, 0]
    ]
    plot_phase_portrait_with_nullclines(default_params, (0, 500), initial_conditions_list)

    
