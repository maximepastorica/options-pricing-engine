from utils import (
    validate_model_inputs,
    validate_option_type,
    validate_steps
)
import numpy as np

def binomial_tree(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        N: int 
) -> float:


    validate_steps(N)
    validate_model_inputs(S, K, T, r, sigma)
    validate_option_type(option_type)

    dt = T / N

    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    q = (np.exp(r * dt) - d) / (u - d)

    j = np.arange(N + 1)
    stock_prices = S * u ** (N - j) * d ** j

    if option_type == "call":
        option_values = np.maximum(stock_prices - K, 0)
    else:
        option_values = np.maximum(K - stock_prices, 0)

    temp = option_values

    for _ in range(N):
        temp = np.exp(-r * dt) * (
            q * temp[:-1] + (1 - q) * temp[1:]
        )
    return temp[0]


def crr_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    N: int,
) -> float:

    validate_steps(N)
    validate_model_inputs(S, K, T, r, sigma)
    validate_option_type(option_type)

    dt = T / N

    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))

    q = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)

    # Terminal stock prices
    j = np.arange(N + 1)
    stock_prices = S * u ** (N - j) * d ** j

    # Terminal option values
    if option_type == "call":
        option_values = np.maximum(stock_prices - K, 0)
    else:
        option_values = np.maximum(K - stock_prices, 0)

    # Move backwards until the first level of the tree
    temp = option_values

    for _ in range(N - 1):
        temp = discount_factor * (
            q * temp[:-1]
            + (1 - q) * temp[1:]
        )

    # Option values after one up/down move
    V_u = temp[0]
    V_d = temp[1]

    # Stock prices after one up/down move
    S_u = S * u
    S_d = S * d

    return (V_u - V_d) / (S_u - S_d)


def crr_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    N: int,
) -> float:

    validate_steps(N)
    validate_model_inputs(S, K, T, r, sigma)
    validate_option_type(option_type)

    if N < 2:
        raise ValueError(
            "At least 2 steps are required to calculate CRR Gamma"
        )

    dt = T / N

    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))

    q = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)

    # Terminal stock prices
    j = np.arange(N + 1)
    stock_prices = S * u ** (N - j) * d ** j

    # Terminal option values
    if option_type == "call":
        option_values = np.maximum(stock_prices - K, 0)
    else:
        option_values = np.maximum(K - stock_prices, 0)

    # Move backwards to the second level of the tree
    temp = option_values

    for _ in range(N - 2):
        temp = discount_factor * (
            q * temp[:-1]
            + (1 - q) * temp[1:]
        )

    # Option values at the second level
    V_uu = temp[0]
    V_ud = temp[1]
    V_dd = temp[2]

    # Stock prices at the second level
    S_uu = S * u**2
    S_ud = S * u * d
    S_dd = S * d**2

    # Delta at the upper and lower branches
    delta_u = (V_uu - V_ud) / (S_uu - S_ud)
    delta_d = (V_ud - V_dd) / (S_ud - S_dd)

    # Gamma
    return (delta_u - delta_d) / (
        (S_uu - S_dd) / 2
    )

            
def crr_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    N: int,
    h: float = 1e-3,
) -> float:

    validate_steps(N)
    validate_model_inputs(S, K, T, r, sigma)
    validate_option_type(option_type)

    if h <= 0 or h >= sigma:
        raise ValueError(
            "Step size h must be positive and smaller than sigma"
        )

    price_up = binomial_tree(
        S, K, T, r, sigma + h, option_type, N
    )
    price_down = binomial_tree(
        S, K, T, r, sigma - h, option_type, N
    )

    return (price_up - price_down) / (2 * h)


def crr_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    N: int,
    h: float = 1e-4,
) -> float:

    validate_steps(N)
    validate_model_inputs(S, K, T, r, sigma)
    validate_option_type(option_type)

    if h <= 0:
        raise ValueError("Step size h must be positive")

    price_up = binomial_tree(
        S, K, T, r + h, sigma, option_type, N
    )
    price_down = binomial_tree(
        S, K, T, r - h, sigma, option_type, N
    )

    return (price_up - price_down) / (2 * h)


def crr_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    N: int,
) -> float:

    validate_steps(N)
    validate_model_inputs(S, K, T, r, sigma)
    validate_option_type(option_type)

    if N < 2:
        raise ValueError(
            "At least 2 steps are required to calculate CRR Theta"
        )

    dt = T / N

    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))

    q = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)

    # Terminal stock prices
    j = np.arange(N + 1)
    stock_prices = S * u ** (N - j) * d ** j

    # Terminal option values
    if option_type == "call":
        option_values = np.maximum(stock_prices - K, 0)
    else:
        option_values = np.maximum(K - stock_prices, 0)

    # Move backwards to the second level
    temp = option_values

    for _ in range(N - 2):
        temp = discount_factor * (
            q * temp[:-1]
            + (1 - q) * temp[1:]
        )

    # Middle node after one up and one down move
    V_ud = temp[1]

    # Move from the second level to the initial option value
    first_level = discount_factor * (
        q * temp[:-1]
        + (1 - q) * temp[1:]
    )

    V_0 = discount_factor * (
        q * first_level[0]
        + (1 - q) * first_level[1]
    )

    return (V_ud - V_0) / (2 * dt)