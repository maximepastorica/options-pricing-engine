import matplotlib.pyplot as plt, numpy as np
from black_scholes import black_scholes
from binomial_tree import binomial_tree
from utils import (validate_model_inputs, 
                   validate_option_type,
                   validate_steps)

def plot_crr_convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    max_steps: int,
) -> None:
    
    validate_steps(max_steps)
    validate_option_type(option_type)
    validate_model_inputs(S, K, T, r, sigma)

    black_scholes_value = black_scholes(
                                S,
                                K,
                                T,
                                r,
                                sigma,
                                option_type,
                    )

    binomial_tree_values = np.zeros(max_steps)
    abs_error_values = np.zeros(max_steps)
    N_values = np.arange(1, max_steps + 1)

    for i in range(max_steps):
        binomial_tree_values[i] = binomial_tree(
                        S,
                        K,
                        T,
                        r,
                        sigma,
                        option_type,
                        i + 1,
                    )
        abs_error_values[i] = abs(binomial_tree_values[i] - black_scholes_value)

    plt.axhline(
    black_scholes_value,
    label="Black-Scholes",
    )

    plt.plot(
        N_values,
        binomial_tree_values,
        label = "CRR",
        color = "red",
    )

    plt.xlabel("Number of steps (N)")
    plt.ylabel("Option price")
    plt.title("CRR Convergence to Black-Scholes")
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(
        N_values, 
        abs_error_values,
        label = "Absolute error"
        )
    
    plt.xlabel("Number of steps (N)")
    plt.ylabel("Absolute error")
    plt.title("CRR Absolute Error")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng()

    S = rng.uniform(20, 500)
    K = rng.uniform(20, 500)
    T = rng.uniform(0.05, 5)
    r = rng.uniform(-0.02, 0.15)
    sigma = rng.uniform(0.05, 1.0)

    print(f"S = {S:.2f}")
    print(f"K = {K:.2f}")
    print(f"T = {T:.2f}")
    print(f"r = {r:.4f}")
    print(f"sigma = {sigma:.4f}")

    plot_crr_convergence(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type="call",
        max_steps=100,
    )