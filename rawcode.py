import numpy as np
from numba import jit, float64#don't forget to manually add the jitter
@jit((float64, float64, float64, float64, float64, float64, float64, float64), nopython=True)
def compute(
    mu_q: float, sigma_q: float,
    mu_outcome: float, sigma_outcome: float,
    mu_condition: float, sigma_condition: float,
    mu_or: float, sigma_or: float,
    ) -> (float, float):
    "Guess the odds ratio of a lognormal distribution."
    delta_mean = (
    (np.exp(mu_outcome + 0.5*sigma_outcome**2) + np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
)*(np.exp(mu_q + 0.5*sigma_q**2) + np.exp(mu_condition + mu_or + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_or**2 + 0.5*sigma_q**2)
)*(np.log(np.exp(mu_outcome + 0.5*sigma_outcome**2) + np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
) - np.log(np.exp(mu_q + 0.5*sigma_q**2) + np.exp(mu_condition + mu_or + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_or**2 + 0.5*sigma_q**2)
)) + 0.5*(np.exp(mu_outcome + 0.5*sigma_outcome**2) + np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
)*(-(1 - np.exp(sigma_q**2)
)*np.exp(2*mu_q + sigma_q**2) - (1 - np.exp(sigma_condition**2 + sigma_or**2 + sigma_q**2)
)*np.exp(2*mu_condition + 2*mu_or + 2*mu_q + sigma_condition**2 + sigma_or**2 + sigma_q**2) + 1) + 0.5*(np.exp(mu_q + 0.5*sigma_q**2) + np.exp(mu_condition + mu_or + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_or**2 + 0.5*sigma_q**2)
)*(
    (1 - np.exp(sigma_outcome**2)
)*np.exp(2*mu_outcome + sigma_outcome**2) + (1 - np.exp(sigma_condition**2 + sigma_q**2)
)*np.exp(2*mu_condition + 2*mu_q + sigma_condition**2 + sigma_q**2) - 1)
)/(
    (np.exp(mu_outcome + 0.5*sigma_outcome**2) + np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
)*(np.exp(mu_q + 0.5*sigma_q**2) + np.exp(mu_condition + mu_or + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_or**2 + 0.5*sigma_q**2)
))

    delta_sigma = np.sqrt(
    (-(1 - np.exp(sigma_outcome**2)
)*np.exp(2*mu_outcome + sigma_outcome**2) - (1 - np.exp(sigma_condition**2 + sigma_q**2)
)*np.exp(2*mu_condition + 2*mu_q + sigma_condition**2 + sigma_q**2) + 1)/(np.exp(mu_outcome + 0.5*sigma_outcome**2) + np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
)) - np.sqrt(
    (-(1 - np.exp(sigma_q**2)
)*np.exp(2*mu_q + sigma_q**2) - (1 - np.exp(sigma_condition**2 + sigma_or**2 + sigma_q**2)
)*np.exp(2*mu_condition + 2*mu_or + 2*mu_q + sigma_condition**2 + sigma_or**2 + sigma_q**2) + 1)/(np.exp(mu_q + 0.5*sigma_q**2) + np.exp(mu_condition + mu_or + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_or**2 + 0.5*sigma_q**2)
))
    return delta_mean, delta_sigma