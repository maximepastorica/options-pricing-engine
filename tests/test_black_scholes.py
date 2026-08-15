import numpy as np
import pytest
from black_scholes import black_scholes

# Check whether the put-call parity C - P = S - K*exp(-r*T) is satisfied.
def test_put_call_parity():

    rng = np.random.default_rng(42)

    N = 100
    S_values = rng.uniform(20, 500, size=N)
    K_values = rng.uniform(20, 500, size=N)
    T_values = rng.uniform(0.05, 5, size=N)
    r_values = rng.uniform(-0.02, 0.15, size=N)
    sigma_values = rng.uniform(0.05, 1.0, size=N)

    for i in range(N):
        C = black_scholes(S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i], "call")
        P = black_scholes(S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i], "put")
        assert np.isclose(C - P, S_values[i] - K_values[i]*np.exp(-r_values[i]*T_values[i]))


# Check Black-Scholes prices against known reference values.
def test_known_value():

    S_values = np.array([100, 100, 100, 50])
    K_values = np.array([100, 110, 90, 50])
    T_values = np.array([1, 1, 1, 0.5])
    r_values = np.array([0.05, 0.05, 0.05, 0.03])
    sigma_values = np.array([0.2, 0.2, 0.2, 0.3])
    C_values = np.array([10.4505835722, 6.0400881297, 16.6994484084, 4.57469928883])
    P_values = np.array([5.5735260223, 10.6753248248, 2.3100966135, 3.83029626898])

    for i in range(4):
        C_estimates = black_scholes(S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i], "call")
        P_estimates = black_scholes(S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i], "put")
        assert np.allclose(np.array([C_estimates, P_estimates]), np.array([C_values[i], P_values[i]]))


# Check no-arbitrage bounds for European call and put options.
def test_no_arbitrage_bounds():

    rng = np.random.default_rng(42)

    N = 100
    S_values = rng.uniform(20, 500, size=N)
    K_values = rng.uniform(20, 500, size=N)
    T_values = rng.uniform(0.05, 5, size=N)
    r_values = rng.uniform(-0.02, 0.15, size=N)
    sigma_values = rng.uniform(0.05, 1.0, size=N)

    for i in range(N):
        C = black_scholes(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "call"
        )

        P = black_scholes(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "put"
        )

        call_lower_bound = max(
            S_values[i] - K_values[i] * np.exp(-r_values[i] * T_values[i]),
            0
        )

        call_upper_bound = S_values[i]

        put_lower_bound = max(
            K_values[i] * np.exp(-r_values[i] * T_values[i]) - S_values[i],
            0
        )

        put_upper_bound = K_values[i] * np.exp(-r_values[i] * T_values[i])

        assert call_lower_bound <= C <= call_upper_bound
        assert put_lower_bound <= P <= put_upper_bound


# Check whether invalid input parameters raise ValueError.
def test_invalid_inputs():

    with pytest.raises(ValueError):
        black_scholes(-100, 100, 1, 0.05, 0.2, "call")

    with pytest.raises(ValueError):
        black_scholes(100, 0, 1, 0.05, 0.2, "call")

    with pytest.raises(ValueError):
        black_scholes(100, 100, -1, 0.05, 0.2, "call")

    with pytest.raises(ValueError):
        black_scholes(100, 100, 1, 0.05, 0, "call")

    with pytest.raises(ValueError):
        black_scholes(100, 100, 1, 0.05, 0.2, "banana")