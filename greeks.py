from utils import compute_d1_d2, validate_model_inputs, validate_option_type
from scipy.stats import norm
from collections.abc import Callable
import numpy as np

def black_scholes_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str
) -> float:
    
    """
    Calculate the analytical Delta of a European call or put option
    using the Black-Scholes model.

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.
    option_type : str
        Type of the option: "call" or "put".

    Returns
    -------
    float
        The Black-Scholes Delta of the option.

    Raises
    ------
    ValueError
        If S, K, T, or sigma is not positive, or if option_type
        is neither "call" nor "put".
    """

    validate_model_inputs(S, K, T, r, sigma)
    validate_option_type(option_type)

    d1, _ = compute_d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1

def numerical_delta(
    pricing_function: Callable[[float, float, float, float, float, str], float],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    h: float
) -> float:

    """
    Approximate the Delta of an option using the central finite difference method.

    Parameters
    ----------
    pricing_function : Callable
        Function used to calculate the option price.
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.
    option_type : str
        Type of the option: "call" or "put".
    h : float
        Step size used in the finite difference approximation.

    Returns
    -------
    float
        Numerical approximation of the option Delta.

    Raises
    ------
    ValueError
        If h is not positive or is greater than or equal to S.

    Notes
    -----
    Delta is approximated using the central finite difference formula:

        Delta ≈ [V(S + h) - V(S - h)] / (2h)
    """
    
    if h <= 0 or h >= S:
        raise ValueError("Step size h must be positive and smaller than S")
    
    return ( 
        pricing_function(S + h, K, T, r, sigma, option_type) 
        - pricing_function(S - h, K, T, r, sigma, option_type) 
        ) / ( 2*h )

        
def black_scholes_gamma(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
) -> float:
    """
    Calculate the analytical Gamma of a European option
    using the Black-Scholes model.

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.

    Returns
    -------
    float
        The Black-Scholes Gamma of the option.

    Raises
    ------
    ValueError
        If S, K, T, or sigma is not positive.
    """

    validate_model_inputs(S, K, T, r, sigma)
    d1, _ = compute_d1_d2(S, K, T, r, sigma)

    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def numerical_gamma(
        pricing_function: Callable[[float, float, float, float, float, str], float],
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        h: float
) -> float:
    """
    Approximate the Gamma of an option using the central finite difference method.

    Parameters
    ----------
    pricing_function : Callable
        Function used to calculate the option price.
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.
    option_type : str
        Type of the option: "call" or "put".
    h : float
        Step size used in the finite difference approximation.

    Returns
    -------
    float
        Numerical approximation of the option Gamma.

    Raises
    ------
    ValueError
        If h is not positive or is greater than or equal to S.

    Notes
    -----
    Gamma is approximated using the central finite difference formula:

        Gamma ≈ [V(S + h) - 2V(S) + V(S - h)] / h^2
    """

    if h <= 0 or h >= S:
        raise ValueError("Step size h must be positive and smaller than S")

    return (
        pricing_function(S + h, K, T, r, sigma, option_type)
        - 2 * pricing_function(S, K, T, r, sigma, option_type)
        + pricing_function(S - h, K, T, r, sigma, option_type)
    ) / h**2


def black_scholes_vega(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float
) -> float:
    """
    Calculate the analytical Vega of a European option
    using the Black-Scholes model.

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.

    Returns
    -------
    float
        The Black-Scholes Vega of the option.

    Raises
    ------
    ValueError
        If S, K, T, or sigma is not positive.
    """

    validate_model_inputs(S, K, T, r, sigma)

    d1, _ = compute_d1_d2(S, K, T, r, sigma)

    return S * norm.pdf(d1) * np.sqrt(T)


def numerical_vega(
        pricing_function: Callable[[float, float, float, float, float, str], float],
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str,
        h: float
) -> float:
    """
    Approximate the Vega of an option using the central finite difference method.

    Parameters
    ----------
    pricing_function : Callable
        Function used to calculate the option price.
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.
    option_type : str
        Type of the option: "call" or "put".
    h : float
        Step size used in the finite difference approximation.

    Returns
    -------
    float
        Numerical approximation of the option Vega.

    Raises
    ------
    ValueError
        If h is not positive or is greater than or equal to sigma.

    Notes
    -----
    Vega is approximated using the central finite difference formula:

        Vega ≈ [V(sigma + h) - V(sigma - h)] / (2h)
    """

    if h <= 0 or h >= sigma:
        raise ValueError("Step size h must be positive and smaller than sigma")

    return (
        pricing_function(S, K, T, r, sigma + h, option_type)
        - pricing_function(S, K, T, r, sigma - h, option_type)
    ) / (2 * h)


def black_scholes_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Calculate the analytical Rho of a European call or put option
    using the Black-Scholes model.

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.
    option_type : str
        Type of the option: "call" or "put".

    Returns
    -------
    float
        The Black-Scholes Rho of the option.

    Raises
    ------
    ValueError
        If S, K, T, or sigma is not positive, or if option_type
        is neither "call" nor "put".
    """

    validate_model_inputs(S, K, T, r, sigma)
    validate_option_type(option_type)

    _, d2 = compute_d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)

    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


def numerical_rho(
    pricing_function: Callable[[float, float, float, float, float, str], float],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    h: float,
) -> float:
    """
    Approximate the Rho of an option using the central finite difference method.

    Parameters
    ----------
    pricing_function : Callable
        Function used to calculate the option price.
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.
    option_type : str
        Type of the option: "call" or "put".
    h : float
        Step size used in the finite difference approximation.

    Returns
    -------
    float
        Numerical approximation of the option Rho.

    Raises
    ------
    ValueError
        If h is not positive.

    Notes
    -----
    Rho is approximated using the central finite difference formula:

        Rho ≈ [V(r + h) - V(r - h)] / (2h)
    """

    if h <= 0:
        raise ValueError("Step size h must be positive")

    return (
        pricing_function(S, K, T, r + h, sigma, option_type)
        - pricing_function(S, K, T, r - h, sigma, option_type)
    ) / (2 * h)


def black_scholes_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Calculate the analytical Theta of a European call or put option
    using the Black-Scholes model.

    Parameters
    ----------
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.
    option_type : str
        Type of the option: "call" or "put".

    Returns
    -------
    float
        The Black-Scholes Theta of the option.

    Raises
    ------
    ValueError
        If S, K, T, or sigma is not positive, or if option_type
        is neither "call" nor "put".
    """

    validate_model_inputs(S, K, T, r, sigma)
    validate_option_type(option_type)

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)

    decay_term = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        return (
            decay_term
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        )

    return (
        decay_term
        + r * K * np.exp(-r * T) * norm.cdf(-d2)
    )


def numerical_theta(
    pricing_function: Callable[[float, float, float, float, float, str], float],
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    h: float,
) -> float:
    """
    Approximate the Theta of an option using the central finite difference method.

    Parameters
    ----------
    pricing_function : Callable
        Function used to calculate the option price.
    S : float
        Current price of the underlying asset.
    K : float
        Strike price of the option.
    T : float
        Time to maturity in years.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Annualized volatility of the underlying asset.
    option_type : str
        Type of the option: "call" or "put".
    h : float
        Step size used in the finite difference approximation.

    Returns
    -------
    float
        Numerical approximation of the option Theta.

    Raises
    ------
    ValueError
        If h is not positive or is greater than or equal to T.

    Notes
    -----
    T represents time to maturity. Since calendar time increases as
    time to maturity decreases, Theta is approximated as:

        Theta ≈ [V(T - h) - V(T + h)] / (2h)
    """

    if h <= 0 or h >= T:
        raise ValueError("Step size h must satisfy 0 < h < T")

    return (
        pricing_function(S, K, T - h, r, sigma, option_type)
        - pricing_function(S, K, T + h, r, sigma, option_type)
    ) / (2 * h)