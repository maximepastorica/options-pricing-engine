import numpy as np
from binomial_tree import binomial_tree

def test_known_value():

    S_values = np.array([100])
    K_values = np.array([100])
    T_values = np.array([1])
    r_values = np.array([0.05])
    sigma_values = np.array([0.2])
    C_values = np.array([9.5405])

    N = 2

    for i in range(len(S_values)):
        C_estimate = binomial_tree(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "call",
            N,
        )

        assert np.isclose(
            C_estimate,
            C_values[i],
            rtol=1e-3,
            atol=1e-4,
        )