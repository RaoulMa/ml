import math
import numpy.random as nrand

from utils import ModelParameters
from utils import convert_to_prices
from utils import plot_stochastic_processes

def brownian_motion_log_returns(param):
    """Differentials of Wiener process.
    :param param: the model parameters object
    :return: brownian motion log returns; [dW_0, ..., dW_T]
    """
    sqrt_delta_sigma = math.sqrt(param.all_delta) * param.all_sigma
    return nrand.normal(loc=0, scale=sqrt_delta_sigma, size=param.all_time)

def brownian_motion_levels(param):
    """Returns a price sequence whose returns evolve according to a brownian motion.
    :param param: model parameters object
    :return: returns a price sequence which follows a brownian motion
    """
    return convert_to_prices(param, brownian_motion_log_returns(param))

def geometric_brownian_motion_log_returns(param):
    """
    This method constructs a sequence of log returns which, when exponentiated, produce a random Geometric Brownian
    Motion (GBM). GBM is the stochastic process underlying the Black Scholes options pricing formula.
    :param param: model parameters object
    :return: returns the log returns of a geometric brownian motion process
    """
    assert isinstance(param, ModelParameters)
    wiener_process = numpy.array(brownian_motion_log_returns(param))
    sigma_pow_mu_delta = (param.gbm_mu - 0.5 * math.pow(param.all_sigma, 2.0)) * param.all_delta
    return wiener_process + sigma_pow_mu_delta

def geometric_brownian_motion_levels(param):
    """
    Returns a sequence of price levels for an asset which evolves according to a geometric brownian motion
    :param param: model parameters object
    :return: the price levels for the asset
    """
    return convert_to_prices(param, geometric_brownian_motion_log_returns(param))

def ornstein_uhlenbeck_levels(param):
    """Returns the rate levels of a mean-reverting Ornstein-Uhlenbeck process.
    :param param: the model parameters object
    :return: the interest rate levels for the Ornstein Uhlenbeck process
    """
    ou_levels = [param.all_r0]
    brownian_motion_returns = brownian_motion_log_returns(param)
    for i in range(1, param.all_time):
        drift = param.ou_a * (param.ou_mu - ou_levels[i-1]) * param.all_delta
        randomness = brownian_motion_returns[i - 1]
        ou_levels.append(ou_levels[i - 1] + drift + randomness)
    return ou_levels

params = ModelParameters(100, 10000, 1/252, 0.1, 0.1, ou_a=0.15, all_r0=1)

processes = [ornstein_uhlenbeck_levels(params),
             ornstein_uhlenbeck_levels(params),
             ornstein_uhlenbeck_levels(params)]

plot_stochastic_processes(processes, 'Processes')