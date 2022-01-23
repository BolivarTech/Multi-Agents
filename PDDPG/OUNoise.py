import numpy as np
import random
import copy


class OUNoise:
    """
    Ornstein-Uhlenbeck stochastic process that it is a Gaussian, Markov process and is temporally homogeneous.

    The process have a tendency to move back towards a central location following a 'random walk', with a greater
    attraction when the process is further away from the center.
    """

    def __init__(self, size=1, seed=None, mu=0., theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process.
        :param size (int): sample dimention
        :param seed: seed used on stochastic selection
        :param mu (float): the is long-term mean of the process
        :param theta (float):  the rate reverts towards the mean
        :param sigma (float):the degree of volatility around the mean
        """
        random.seed(seed)
        self.__mu = mu * np.ones(size)
        self.__theta = theta
        self.__sigma = sigma
        self.__state = copy.copy(self.__mu)

    def __del__(self):
        """
        Class Destructor

        """
        del self.__mu
        del self.__theta
        del self.__sigma
        del self.__state

    def reset(self):
        """
        Reset the internal __state (= noise) to mean (__mu).
        """
        self.__state = copy.copy(self.__mu)

    def sample(self):
        """
        Update internal __state and return it as a noise sample.
        """
        x = self.__state
        dx = self.__theta * (self.__mu - x) + self.__sigma * np.array([random.random() for i in range(len(x))])
        self.__state = x + dx
        return self.__state
