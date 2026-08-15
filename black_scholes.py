import numpy as np
from scipy.stats import norm

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
    if S <= 0 or K <= 0 or T <= 0  or sigma <= 0:
        raise ValueError("Parameters S, K, T and sigma must be positive")
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == "call":
        C = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        return C
    elif option_type == "put":
        P = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        return P
    else:
        raise ValueError("option_type must be a 'call' or 'put'")
