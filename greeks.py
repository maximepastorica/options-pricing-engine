from utils import compute_d1_d2, validate_inputs
from scipy.stats import norm
from collections.abc import Callable

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

    validate_inputs(S, K, T, r, sigma, option_type)

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
    
    return ( pricing_function(S + h, K, T, r, sigma, option_type) - pricing_function(S - h, K, T, r, sigma, option_type) ) / ( 2*h )

    
        