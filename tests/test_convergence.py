from black_scholes import black_scholes
from binomial_tree import binomial_tree
import numpy as np

def test_crr_convergence() -> None:

    rng = np.random.default_rng(42)
    n = 100
    S_values = rng.uniform(20, 500, size=n)
    K_values = rng.uniform(20, 500, size=n)
    T_values = rng.uniform(0.05, 5, size=n)
    r_values = rng.uniform(-0.02, 0.15, size=n)
    sigma_values = rng.uniform(0.05, 1.0, size=n)
    N = 1000

    for i in range(n):
        assert np.isclose(  black_scholes(S_values[i],
                                          K_values[i],
                                          T_values[i],
                                          r_values[i],
                                          sigma_values[i],
                                          "call"),
                            binomial_tree(S_values[i],
                                          K_values[i],
                                          T_values[i],
                                          r_values[i],
                                          sigma_values[i],
                                          "call",
                                          N),
                            rtol=5e-3,
                            atol=5e-3
                        )

        assert np.isclose(  black_scholes(S_values[i],
                                          K_values[i],
                                          T_values[i],
                                          r_values[i],
                                          sigma_values[i],
                                          "put"),
                            binomial_tree(S_values[i],
                                          K_values[i],
                                          T_values[i],
                                          r_values[i],
                                          sigma_values[i],
                                          "put",
                                          N),
                            rtol=5e-3,
                            atol=5e-3
                                )

        

