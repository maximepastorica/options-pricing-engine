import numpy as np
from scipy.stats import norm
from utils import compute_d1_d2, validate_inputs

def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:

    """
    Price a European call or put option using the Black-Scholes model.

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
        The Black-Scholes price of the option.

    Raises
    ------
    ValueError
        If S, K, T, or sigma is not positive, or if option_type
        is neither "call" nor "put".
    """
    validate_inputs(S, K, T, r, sigma, option_type)
    
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        C = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return C
    elif option_type == "put":
        P = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        return P
