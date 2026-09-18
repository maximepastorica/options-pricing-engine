import numpy as np

from binomial_tree import (
    crr_delta,
    crr_gamma,
    crr_vega,
    crr_rho,
    crr_theta,
)

from greeks import (
    black_scholes_delta,
    black_scholes_gamma,
    black_scholes_vega,
    black_scholes_rho,
    black_scholes_theta,
)


def test_crr_delta():
    rng = np.random.default_rng(42)

    S_values = rng.uniform(20, 500, 100)
    K_values = rng.uniform(20, 500, 100)
    T_values = rng.uniform(0.05, 5, 100)
    r_values = rng.uniform(-0.02, 0.15, 100)
    sigma_values = rng.uniform(0.05, 1, 100)

    N = 2000

    for S, K, T, r, sigma in zip(
        S_values,
        K_values,
        T_values,
        r_values,
        sigma_values,
    ):
        analytical_call_delta = black_scholes_delta(
            S, K, T, r, sigma, "call"
        )
        analytical_put_delta = black_scholes_delta(
            S, K, T, r, sigma, "put"
        )

        crr_call_delta = crr_delta(
            S, K, T, r, sigma, "call", N
        )
        crr_put_delta = crr_delta(
            S, K, T, r, sigma, "put", N
        )

        assert np.isclose(
            analytical_call_delta,
            crr_call_delta,
            rtol=1e-2,
            atol=1e-2,
        )

        assert np.isclose(
            analytical_put_delta,
            crr_put_delta,
            rtol=1e-2,
            atol=1e-2,
        )


def test_crr_gamma():
    rng = np.random.default_rng(42)

    S_values = rng.uniform(20, 500, 100)
    K_values = rng.uniform(20, 500, 100)
    T_values = rng.uniform(0.05, 5, 100)
    r_values = rng.uniform(-0.02, 0.15, 100)
    sigma_values = rng.uniform(0.05, 1, 100)

    N = 2000

    for S, K, T, r, sigma in zip(
        S_values,
        K_values,
        T_values,
        r_values,
        sigma_values,
    ):
        analytical_gamma = black_scholes_gamma(
            S, K, T, r, sigma
        )

        crr_call_gamma = crr_gamma(
            S, K, T, r, sigma, "call", N
        )
        crr_put_gamma = crr_gamma(
            S, K, T, r, sigma, "put", N
        )

        assert np.isclose(
            analytical_gamma,
            crr_call_gamma,
            rtol=1e-2,
            atol=1e-3,
        )

        assert np.isclose(
            analytical_gamma,
            crr_put_gamma,
            rtol=1e-2,
            atol=1e-3,
        )


def test_crr_vega():
    rng = np.random.default_rng(42)

    S_values = rng.uniform(20, 500, 50)
    K_values = rng.uniform(20, 500, 50)
    T_values = rng.uniform(0.05, 5, 50)
    r_values = rng.uniform(-0.02, 0.15, 50)
    sigma_values = rng.uniform(0.05, 1, 50)

    N = 2000
    h = 1e-2

    for S, K, T, r, sigma in zip(
        S_values,
        K_values,
        T_values,
        r_values,
        sigma_values,
    ):
        analytical_vega = black_scholes_vega(
            S, K, T, r, sigma
        )

        crr_call_vega = crr_vega(
            S, K, T, r, sigma, "call", N, h
        )
        crr_put_vega = crr_vega(
            S, K, T, r, sigma, "put", N, h
        )

        assert np.isclose(
            analytical_vega,
            crr_call_vega,
            rtol=5e-2,
            atol=1e-2,
        )

        assert np.isclose(
            analytical_vega,
            crr_put_vega,
            rtol=5e-2,
            atol=1e-2,
        )


def test_crr_rho():
    rng = np.random.default_rng(42)

    S_values = rng.uniform(20, 500, 50)
    K_values = rng.uniform(20, 500, 50)
    T_values = rng.uniform(0.05, 5, 50)
    r_values = rng.uniform(-0.02, 0.15, 50)
    sigma_values = rng.uniform(0.05, 1, 50)

    N = 2000
    h = 1e-4

    for S, K, T, r, sigma in zip(
        S_values,
        K_values,
        T_values,
        r_values,
        sigma_values,
    ):
        analytical_call_rho = black_scholes_rho(
            S, K, T, r, sigma, "call"
        )
        analytical_put_rho = black_scholes_rho(
            S, K, T, r, sigma, "put"
        )

        crr_call_rho = crr_rho(
            S, K, T, r, sigma, "call", N, h
        )
        crr_put_rho = crr_rho(
            S, K, T, r, sigma, "put", N, h
        )

        assert np.isclose(
            analytical_call_rho,
            crr_call_rho,
            rtol=1e-2,
            atol=1e-2,
        )

        assert np.isclose(
            analytical_put_rho,
            crr_put_rho,
            rtol=1e-2,
            atol=1e-2,
        )


def test_crr_theta():
    rng = np.random.default_rng(42)

    S_values = rng.uniform(20, 500, 50)
    K_values = rng.uniform(20, 500, 50)
    T_values = rng.uniform(0.05, 5, 50)
    r_values = rng.uniform(-0.02, 0.15, 50)
    sigma_values = rng.uniform(0.05, 1, 50)

    N = 2000

    for S, K, T, r, sigma in zip(
        S_values,
        K_values,
        T_values,
        r_values,
        sigma_values,
    ):
        analytical_call_theta = black_scholes_theta(
            S, K, T, r, sigma, "call"
        )

        analytical_put_theta = black_scholes_theta(
            S, K, T, r, sigma, "put"
        )

        crr_call_theta = crr_theta(
            S, K, T, r, sigma, "call", N
        )

        crr_put_theta = crr_theta(
            S, K, T, r, sigma, "put", N
        )

        assert np.isclose(
            analytical_call_theta,
            crr_call_theta,
            rtol=5e-2,
            atol=1e-2,
        )

        assert np.isclose(
            analytical_put_theta,
            crr_put_theta,
            rtol=5e-2,
            atol=1e-2,
        )