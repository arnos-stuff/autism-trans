from typing import List, Tuple
from numba import jit, float64
import sympy as sp
import numpy as np
import scipy.optimize as opt

def from_confidence_intervals(lower: float, upper: float, symbolic=False) -> (float, float):
    """Compute the mean and standard deviation of a lognormal distribution from the
    confidence interval of the mean.

    :param mu: the mean of the lognormal distribution
    :param lower: the lower bound of the confidence interval
    :param upper: the upper bound of the confidence interval
    :return: the standard deviation of the lognormal distribution
    """
    if symbolic:
        log = sp.log
        sqrt = sp.sqrt
    else:
        log = np.log
        sqrt = np.sqrt

    mean = (log(lower) + log(upper)) / 2.0
    std = sqrt(log(upper) - log(lower) )/ (2.0 * 1.96)
    return mean, std

def to_confidence_interval(mu: float, sigma: float, symbolic=False) -> (float, float):
    """Compute the confidence interval of the mean of a lognormal distribution.

    :param mu: the mean of the lognormal distribution
    :param sigma: the standard deviation of the lognormal distribution
    :return: the lower and upper bounds of the confidence interval
    """
    if symbolic:
        exp = sp.exp
        sqrt = sp.sqrt
    else:
        exp = np.exp
        sqrt = np.sqrt

    lower = exp(mu - 1.96 * sigma)
    upper = exp(mu + 1.96 * sigma)
    return lower, upper

def to_mean(mu: float, sigma:float, symbolic=False) -> float:
    """Compute the point prediction of the mean of a lognormal distribution."""

    if symbolic:
        exp = sp.exp
    else:
        exp = np.exp

    return exp(mu + 0.5 * sigma**2)

def to_median(mu: float, sigma:float, symbolic=False) -> float:
    """Compute the point prediction of the median of a lognormal distribution."""

    if symbolic:
        exp = sp.exp
    else:
        exp = np.exp

    return exp(mu)

def prod_lognormals_(
    mu1: float, sigma1: float,
    mu2: float, sigma2: float,
    symbolic=False) -> (float, float):
    """Compute the mean and standard deviation of the product of two lognormal distributions.

    :param mu1: the mean of the first lognormal distribution
    :param sigma1: the standard deviation of the first lognormal distribution
    :param mu2: the mean of the second lognormal distribution
    :param sigma2: the standard deviation of the second lognormal distribution
    :return: the mean and standard deviation of the sum of the two lognormal distributions
    """
    if symbolic:
        sqrt = sp.sqrt
    else:
        sqrt = np.sqrt

    mu = mu1 + mu2
    sigma = sp.sqrt(sigma1**2 + sigma2**2)
    return mu, sigma

def sum_lognormals_(
    mu1: float, sigma1: float,
    mu2: float, sigma2: float,
    symbolic = False) -> (float, float):
    """Compute the mean and standard deviation of the sum of two lognormal distributions.

    :param mu1: the mean of the first lognormal distribution
    :param sigma1: the standard deviation of the first lognormal distribution
    :param mu2: the mean of the second lognormal distribution
    :param sigma2: the standard deviation of the second lognormal distribution
    :return: the mean and standard deviation of the sum of the two lognormal distributions
    """
    if symbolic:
        exp = sp.exp
        log = sp.log
        sqrt = sp.sqrt
    else:
        exp = np.exp
        log = np.log
        sqrt = np.sqrt

    denom = exp(mu1 + 0.5 * sigma1**2) + exp(mu2 + 0.5 * sigma2**2)
    
    var = (
        exp(2 * mu1 + sigma1**2) * (exp(sigma1**2) - 1) +
        exp(2 * mu2 + sigma2**2) * (exp(sigma2**2) - 1) + 1
        ) / denom
    mu = log(denom) - 0.5 * var
    return mu, sqrt(var)

def sum_lognormals(*params: List[Tuple[float]]) -> (float, float):
    """Compute the mean and standard deviation of the sum of multiple lognormal distributions.

    :param params: a list of tuples of the form (mu, sigma) for each lognormal distribution
    :return: the mean and standard deviation of the sum of the lognormal distributions
    """
    params = list(params)
    mu, sigma = params.pop(0)
    for mu_i, sigma_i in params:
        mu, sigma = sum_lognormals_(mu, sigma, mu_i, sigma_i)
    return mu, sigma

def prod_lognormals(*params: List[Tuple[float]]) -> (float, float):
    """Compute the mean and standard deviation of the product of multiple lognormal distributions.
    """
    params = list(params)
    mu, sigma = params.pop(0)
    for mu_i, sigma_i in params:
        mu, sigma = prod_lognormals_(mu, sigma, mu_i, sigma_i)
    return mu, sigma

def symbolic_guess_odds_ratio(
        ) -> sp.Symbol:
    """Compute the odds ratio of lognormal distributions.

    If both p & q are lognormally distributed, then the odds ratio is given by

    p/q ~ lognormal(mu_or, sigma_or)

    in the special that p = prob(outcome|condition) and q = prob(outcome|~condition)
    we can infer the distribution of both p and q from the odds ratio.

    this assumes we know both the marginal distribution p(condition) and p(y)

    this is because p(y) = p(y|condition) * p(condition) + p(y|~condition) * p(~condition)

    which yields the following system of equations:

    lognormal(mu_outcome, sigma_outcome) = 
        lognormal(mu_q, sigma_q) * lognormal(mu_or, sigma_or) * lognormal(mu_condition, sigma_condition) +
        lognormal(mu_q, sigma_q) - lognormal(mu_q, sigma_q) * lognormal(mu_condition, sigma_condition)

    which simplifies to
    sum_lognormals(outcome, product_lognormals(q, condition)) =
    sum_lognormals(q, product_lognormals(q, or, condition))

    :param mu: the mean of the lognormal distribution
    :param sigma: the standard deviation of the lognormal distribution
    :return: the odds ratio of the lognormal distribution
    """
    mu_q, mu_outcome, mu_condition = sp.symbols('mu_q, mu_outcome, mu_condition')
    sigma_q, sigma_outcome, sigma_condition = sp.symbols('sigma_q, sigma_outcome, sigma_condition', positive=True)
    
    params_q = (mu_q, sigma_q)
    params_outcome = (mu_outcome, sigma_outcome)
    params_condition = (mu_condition, sigma_condition)

    mu_lhs, sigma_lhs = sum_lognormals(params_outcome, prod_lognormals(params_q, params_condition))
    mu_rhs, sigma_rhs = sum_lognormals(params_q, prod_lognormals(params_q, params_or, params_condition))

    return sp.simplify(mu_lhs - mu_rhs), sp.simplify(sigma_lhs - sigma_rhs)

@jit((float64, float64, float64, float64, float64, float64), nopython=True)
def raw_guess_odds_ratio(
    mu_q: float, sigma_q: float,
    mu_outcome: float, sigma_outcome: float,
    mu_condition: float, sigma_condition: float,
    ) -> (float, float):
    """Guess the odds ratio of a lognormal distribution."""
    
    params_q = (mu_q, sigma_q)
    params_outcome = (mu_outcome, sigma_outcome)
    params_condition = (mu_condition, sigma_condition)

    mu_expr = (
        (
            np.exp(mu_outcome + 0.5*sigma_outcome**2) + 
            np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
        )*(
            np.exp(mu_q + 0.5*sigma_q**2) + 5.56635651637762*np.exp(
                mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
        )*(-np.log(
            (
                5.56635651637762*np.exp(mu_condition + 0.5*sigma_condition**2) + 1.0
            )*np.exp(mu_q + 0.5*sigma_q**2)) + np.log(
                np.exp(mu_outcome + 0.5*sigma_outcome**2) + np.exp(
                    mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2
                    )
                )
            ) + (
                np.exp(mu_outcome + 0.5*sigma_outcome**2) + 
                np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
                )*(
                    -0.5*(1 - np.exp(sigma_q**2))*np.exp(2*mu_q + sigma_q**2) + 
                    (
                        16.0819550303169*np.exp(sigma_condition**2 + sigma_q**2) - 15.4921624337098
                    )*np.exp(2*mu_condition + 2*mu_q + sigma_condition**2 + sigma_q**2) + 0.5
                    
                ) + 0.5*(np.exp(mu_q + 0.5*sigma_q**2) + 5.56635651637762*np.exp(
                    mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2
                    )
                    )*(
                        (1 - np.exp(sigma_outcome**2))*np.exp(2*mu_outcome + sigma_outcome**2) +
                        (1 - np.exp(sigma_condition**2 + sigma_q**2))*np.exp(2*mu_condition + 2*mu_q + sigma_condition**2 + sigma_q**2) - 1)
                        )/(
                            (
                                np.exp(mu_outcome + 0.5*sigma_outcome**2) +
                                np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
                            )*(
                                np.exp(mu_q + 0.5*sigma_q**2) + 5.56635651637762*np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
                            )
                        )
    sigma_expr = np.sqrt(
        (
            -(1 - np.exp(sigma_outcome**2))*np.exp(2*mu_outcome + sigma_outcome**2) -
            (1 - np.exp(sigma_condition**2 + sigma_q**2))*np.exp(
                2*mu_condition + 2*mu_q + sigma_condition**2 + sigma_q**2
                ) + 1
        )/(
            np.exp(mu_outcome + 0.5*sigma_outcome**2) +
            np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
            )
        ) - np.sqrt(
            (
                -(1 - np.exp(sigma_q**2))*np.exp(2*mu_q + sigma_q**2) +
                30.9843248674196*(1.03807038553403*np.exp(sigma_condition**2 + sigma_q**2) - 1)*np.exp(
                    2*mu_condition + 2*mu_q + sigma_condition**2 + sigma_q**2
                    ) + 1
            )/(
                np.exp(mu_q + 0.5*sigma_q**2) +
                5.56635651637762*np.exp(mu_condition + mu_q + 0.5*sigma_condition**2 + 0.5*sigma_q**2)
                )
            )
    return mu_expr, sigma_expr

def solve_for_remaining_parameters(
    mu_outcome: float, sigma_outcome: float,
    mu_condition: float, sigma_condition: float,
    mu0_q: float, sigma0_q: float,
    logit: bool = False,
    display: bool = False,
    maxiter: int = 1000,
    ):
    """Solve for the remaining parameters."""
    if logit:
        def objective(params_array: np.ndarray):
            delta_mu, delta_sigma = raw_guess_odds_ratio(
                -1-params_array[0], params_array[1],
                mu_outcome, sigma_outcome,
                mu_condition, sigma_condition
                )
            return (delta_mu)**2 + (delta_sigma)**2
    else:
        def objective(params_array: np.ndarray):
            delta_mu, delta_sigma = raw_guess_odds_ratio(
                params_array[0], params_array[1],
                mu_outcome, sigma_outcome,
                mu_condition, sigma_condition
                )
            return (delta_mu)**2 + (delta_sigma)**2
    result = opt.minimize(
        fun=objective,
        x0=np.array([mu0_q, sigma0_q]),
        method='SLSQP',
        options={'disp': display, 'maxiter': maxiter, 'gtol': 1e-4, 'ftol': 1e-4},
        # +inf > mu > -inf, +inf > sigma >= 0
        constraints=[
            opt.LinearConstraint(
            A=np.array([[1, 0], [0, 1]]),
            lb=np.array([-np.inf, 0]),
            ub=np.array([np.inf, np.inf]),
            keep_feasible=True
                ),
            opt.NonlinearConstraint(
                fun=lambda x: np.exp(x[0]),
                lb=0,
                ub=1,
                keep_feasible=True
                ),
            opt.NonlinearConstraint(
                fun=lambda x: np.exp(x[0] + 0.5 * x[1]**2),
                lb=0,
                ub=1,
                keep_feasible=True
                ),
            opt.NonlinearConstraint(
                fun=lambda x: np.exp(x[0] + 1.96*x[1]**2),
                lb=0,
                ub=1,
                keep_feasible=True
                )
            ]
        )
    return result.x

def params_solve(
    conf_interval_or: Tuple[float, float],
    logit: bool = False, display: bool = False,
    maxiter: int = 1000,
    ) -> Tuple[float]:
    conf_interval_autism = (0.01, 0.02)
    conf_interval_gender = (0.004, 0.013)

    params_autism = from_confidence_intervals(*conf_interval_autism)
    params_gender = from_confidence_intervals(*conf_interval_gender)
    params_or = from_confidence_intervals(*conf_interval_or)

    mu_autism, sigma_autism = params_autism
    mu_gender, sigma_gender = params_gender
    mu_or, sigma_or = params_or

    mu0 = (mu_autism + mu_gender) / 2.0
    s0 = np.sqrt(sigma_autism**2 + sigma_gender**2)

    result = solve_for_remaining_parameters(
        mu_autism, sigma_autism,
        mu_gender, sigma_gender,
        mu0, s0,
        logit=logit,
        display=display,
        maxiter=maxiter,
        )
    mu_q, sigma_q = result
    mu_p, sigma_p = mu_q + mu_or, np.sqrt(sigma_q**2 + sigma_or**2)

    return {
        "q": {
            "mu": mu_q,
            "sigma":sigma_q,
            "95%CI": to_confidence_interval(mu_q, sigma_q),
            "mean-proba": to_mean(mu_q, sigma_q),
            "median-proba": to_median(mu_q, sigma_q)
        },
        "p": {
            "mu": mu_p,
            "sigma":sigma_p,
            "95%CI": to_confidence_interval(mu_p, sigma_p),
            "mean-proba": to_mean(mu_p, sigma_p),
            "median-proba": to_median(mu_p, sigma_p)
        },
        "gender": {
            "mu": mu_gender,
            "sigma": sigma_gender,
            "95%CI": conf_interval_gender,
            "mean-proba": to_mean(mu_gender, sigma_gender),
            "median-proba": to_median(mu_gender, sigma_gender)
        },
        "autism": {
            "mu": mu_autism,
            "sigma": sigma_autism,
            "95%CI": conf_interval_autism,
            "mean-proba": to_mean(mu_autism, sigma_autism),
            "median-proba": to_median(mu_autism, sigma_autism)
        }
    }

if __name__ == '__main__':
    conf_interval_autism = (0.01, 0.02)
    conf_interval_gender = (0.004, 0.013)
    conf_interval_or = (4.1, 7.28)
    params_autism = from_confidence_intervals(*conf_interval_autism)
    params_gender = from_confidence_intervals(*conf_interval_gender)
    params_or = from_confidence_intervals(*conf_interval_or)

    mu_autism, sigma_autism = params_autism
    mu_gender, sigma_gender = params_gender

    # print(sp.pycode(mu_expr), end='\n\n')
    # print(sp.pycode(sigma_expr))

    print('\mu_k = {}, \sigma_k = {}'.format(mu_autism, sigma_autism))
    print('\mu_r = {}, \sigma_r = {}'.format(mu_gender, sigma_gender))


    # mu0 = (mu_autism + mu_gender) / 2.0
    # s0 = np.sqrt(sigma_autism**2 + sigma_gender**2)
    # # fwd = raw_guess_odds_ratio(mu0, s0, mu_autism, sigma_autism, mu_gender, sigma_gender)
    # result = solve_for_remaining_parameters(mu_autism, sigma_autism, mu_gender, sigma_gender, mu0, s0) 
    # # print(fwd)
    # print(result)
    # mu_q = -4.35836312,  sigma_q = 0.42120029
    print('\mu_q = {}, \sigma_q = {}'.format(-4.35836312, 0.42120029))

