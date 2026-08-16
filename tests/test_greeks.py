import numpy as np
from black_scholes import black_scholes
from greeks import (
    black_scholes_delta,
    numerical_delta,
    black_scholes_gamma,
    numerical_gamma,
    black_scholes_vega,
    numerical_vega,
    black_scholes_rho,
    numerical_rho,
    black_scholes_theta,
    numerical_theta
)


        
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
        analytical_call_delta = black_scholes_delta(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "call",
        )

        numerical_call_delta = numerical_delta(
            black_scholes,
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "call",
            h,
        )

        analytical_put_delta = black_scholes_delta(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "put",
        )

        numerical_put_delta = numerical_delta(
            black_scholes,
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "put",
            h,
        )

        assert np.isclose(
            analytical_call_delta,
            numerical_call_delta,
            rtol=1e-4,
            atol=1e-8,
        )
        
        assert np.isclose(
            analytical_put_delta,
            numerical_put_delta,
            rtol=1e-4,
            atol=1e-8,
        )


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
        analytical_gamma = black_scholes_gamma(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
        )

        numerical_call_gamma = numerical_gamma(
            black_scholes,
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "call",
            h,
        )

        numerical_put_gamma = numerical_gamma(
            black_scholes,
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "put",
            h,
        )

        assert np.isclose(
            analytical_gamma,
            numerical_call_gamma,
            rtol=1e-4,
            atol=1e-8,
        )
        
        assert np.isclose(
            analytical_gamma,
            numerical_put_gamma,
            rtol=1e-4,
            atol=1e-8,
        )


def test_analytical_vs_numerical_vega():

    rng = np.random.default_rng(42)

    N = 100
    h = 1e-4

    S_values = rng.uniform(20, 500, size=N)
    K_values = rng.uniform(20, 500, size=N)
    T_values = rng.uniform(0.05, 5, size=N)
    r_values = rng.uniform(-0.02, 0.15, size=N)
    sigma_values = rng.uniform(0.05, 1.0, size=N)

    for i in range(N):
        analytical_vega = black_scholes_vega(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
        )

        numerical_call_vega = numerical_vega(
            black_scholes,
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "call",
            h,
        )

        numerical_put_vega = numerical_vega(
            black_scholes,
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "put",
            h,
        )

        assert np.isclose(
            analytical_vega,
            numerical_call_vega,
            rtol=1e-4,
            atol=1e-8,
        )

        assert np.isclose(
            analytical_vega,
            numerical_put_vega,
            rtol=1e-4,
            atol=1e-8,
        )


def test_analytical_vs_numerical_rho():

    rng = np.random.default_rng(42)

    N = 100
    h = 1e-5

    S_values = rng.uniform(20, 500, size=N)
    K_values = rng.uniform(20, 500, size=N)
    T_values = rng.uniform(0.05, 5, size=N)
    r_values = rng.uniform(-0.02, 0.15, size=N)
    sigma_values = rng.uniform(0.05, 1.0, size=N)

    for i in range(N):
        analytical_call_rho = black_scholes_rho(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "call",
        )

        numerical_call_rho = numerical_rho(
            black_scholes,
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "call",
            h,
        )

        analytical_put_rho = black_scholes_rho(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "put",
        )

        numerical_put_rho = numerical_rho(
            black_scholes,
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "put",
            h,
        )

        assert np.isclose(
            analytical_call_rho,
            numerical_call_rho,
            rtol=1e-4,
            atol=1e-8,
        )

        assert np.isclose(
            analytical_put_rho,
            numerical_put_rho,
            rtol=1e-4,
            atol=1e-8,
        )


def test_analytical_vs_numerical_theta():

    rng = np.random.default_rng(42)

    N = 100
    h = 1e-4

    S_values = rng.uniform(20, 500, size=N)
    K_values = rng.uniform(20, 500, size=N)
    T_values = rng.uniform(0.05, 5, size=N)
    r_values = rng.uniform(-0.02, 0.15, size=N)
    sigma_values = rng.uniform(0.05, 1.0, size=N)

    for i in range(N):
        analytical_call_theta = black_scholes_theta(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "call",
        )

        numerical_call_theta = numerical_theta(
            black_scholes,
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "call",
            h,
        )

        analytical_put_theta = black_scholes_theta(
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "put",
        )

        numerical_put_theta = numerical_theta(
            black_scholes,
            S_values[i],
            K_values[i],
            T_values[i],
            r_values[i],
            sigma_values[i],
            "put",
            h,
        )

        assert np.isclose(
            analytical_call_theta,
            numerical_call_theta,
            rtol=1e-4,
            atol=1e-8,
        )

        assert np.isclose(
            analytical_put_theta,
            numerical_put_theta,
            rtol=1e-4,
            atol=1e-8,
        )