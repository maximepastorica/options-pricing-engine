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
    u = np.exp( sigma * np.sqrt ( dt ) )
    d = np.exp( -sigma * np.sqrt ( dt ) )
    q = ( np.exp( r * dt ) - d ) / ( u - d )
    stock_prices = np.zeros(N + 1)
    option_values = np.zeros(N + 1)

    if option_type == "call":
        for j in range(N + 1):
            stock_prices[j] = S * u ** ( N - j) * d ** j
            option_values[j] = max(stock_prices[j] - K, 0)
    else:
        for j in range(N + 1):
            stock_prices[j] = S * u ** ( N - j) * d ** j
            option_values[j] = max(K - stock_prices[j], 0)

    temp = option_values

    for j in range(N):
        a = np.zeros( len(temp) - 1 )
        for i in range( len(a) ):
            a[i] = np.exp( -r * dt) * ( 
                q * temp[i] + ( 1 - q ) * temp[i+1])
        temp = a

    return temp[0]
            
