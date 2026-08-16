import numpy as np
from greeks import numerical_delta, black_scholes_delta, numerical_gamma, black_scholes_gamma
from black_scholes import black_scholes

def test_analytical_vs_numerical_delta():

    rng = np.random.default_rng(42)
    
    N = 100
    h = 1e-4
    S_values = rng.uniform(20, 500, size=N)
    K_values = rng.uniform(20, 500, size=N)
    T_values = rng.uniform(0.05, 5, size=N)
    r_values = rng.uniform(-0.02, 0.15, size=N)
    sigma_values = rng.uniform(0.05, 1.0, size=N)

    for i in range(N):
        assert np.isclose(black_scholes_delta(S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i], "call"), 
                          numerical_delta(black_scholes, S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i], "call", h))
        
        assert np.isclose(black_scholes_delta(S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i], "put"), 
                                  numerical_delta(black_scholes, S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i], "put", h))
                
def test_analytical_vs_numerical_gamma():

    rng = np.random.default_rng(42)
    
    N = 100
    h = 1e-2
    S_values = rng.uniform(20, 500, size=N)
    K_values = rng.uniform(20, 500, size=N)
    T_values = rng.uniform(0.05, 5, size=N)
    r_values = rng.uniform(-0.02, 0.15, size=N)
    sigma_values = rng.uniform(0.05, 1.0, size=N)

    for i in range(N):
        assert np.isclose(black_scholes_gamma(S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i]), 
                          numerical_gamma(black_scholes, S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i], "call", h))
        
        assert np.isclose(black_scholes_gamma(S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i]), 
                                  numerical_gamma(black_scholes, S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i], "put", h))
