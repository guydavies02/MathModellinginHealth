import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec

def hasty_model(t, state, params):
    
    x, y = state
    gamma_x = params['gamma_x']
    gamma_y = params['gamma_y']
    alpha = params['alpha']
    sigma = params['sigma']
    tau_y = params['tau_y']
    
    common_term = (1 + x**2 + alpha*sigma*x**4) / ((1 + x**2 + sigma*x**4) * (1 + y**4))
    
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

def plot_time_series(t, states, params, title="Hasty Synthetic Gene Oscillator"):

    fig = plt.figure(figsize=(8, 5))   
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
