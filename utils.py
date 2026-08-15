import numpy as np

def compute_d1_d2(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
) -> tuple[float, float]:
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return d1, d2

def validate_inputs(
         S: float,
            K: float,
            T: float,
            r: float,
            sigma: float,
            option_type: str
) -> None:
    if S <= 0 or K <= 0 or T <= 0  or sigma <= 0:
        raise ValueError("Parameters S, K, T and sigma must be positive")
    elif option_type not in ("call", "put"):
        raise ValueError("option_type must be a 'call' or 'put'")