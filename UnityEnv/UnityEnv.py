"""
Unity Simulation Environment

By: Julian Bolivar
Version: 1.0.0
"""
import numpy as np
from unityagents import UnityEnvironment
import logging as log

logLevel_ = log.INFO


class UnityEnv:
    """
    Load and configure Unity Simulation Environment
    """

    def __init__(self, train_mode=True, env_filename=".", log_handler=None):
        """
        Class Constructor

        :param train_mode: Set the Environment mode on train, default is True
        :param env_filename: Path to Unity Environment
        :param log_Handler (handlers): Log handler to be used in the logging (Default is None)
        """
        global logLevel_

        # Set the error logger
        self.__logger = log.getLogger('UnityEnv')
        # Add handler to logger
        self.__logHandler = log_handler
        if self.__logHandler is not None:
            self.__logger.addHandler(self.__logHandler)
        else:
            self.__logger.debug(f"logHandler NOT defined (001)")
        # Set Logger Lever
        self.__logger.setLevel(logLevel_)
        self.__env = UnityEnvironment(env_filename)
        # get the default brain
        self.__brain_name = self.__env.brain_names[0]
        self.__brain = self.__env.brains[self.__brain_name]
        self.__env_info = None
        self.__env_info = self.__env.reset(train_mode)[self.__brain_name]
        self.__state = self.__env_info.vector_observations[0]
        self.__num_agents = len(self.__env_info.agents)
        self.__state_size = self.__env_info.vector_observations.shape[1]
        self.__score = np.zeros((1,self.__num_agents))
        self.__done = np.full((1, self.__num_agents), False)

    def __del__(self):
        """
            Class Destructor
            Deletes all the instances

            :return: None
        """

        self.__env.close()
        del self.__env
        del self.__brain_name
        del self.__brain
        del self.__state
        del self.__done
        del self.__score
        del self.__env_info
        del self.__logger

    def reset(self, train_mode=True):
        """
        Reset the environment

        :param train_mode: Set the Environment mode on train, default is True
        :return: Environment __state
        """

        # reset the environment
        self.__score = np.zeros((1,self.__num_agents))
        self.__done = np.full((1, self.__num_agents), False)
        self.__env_info = self.__env.reset(train_mode)[self.__brain_name]
        self.__logger.debug(f"Environment Rested (002)")
        return self.__env_info.vector_observations

    def get_num_agents(self):
        """
        Return the number of agents on the environment

        :return: Number of agents
        """

        return len(self.__env_info.agents)

    def get_agents(self):
        """
        Return list with the agents in the environment

        :return: List with agents
        """
        return self.__env_info.agents

    def get_num_actions(self):
        """
        Return the number of actions available in the environment

        :return: Number of actions available
        """

        return self.__brain.vector_action_space_size

    def get_state(self):
        """
        Returns the current __state

        :return: Current State
        """

        return self.__state

    def get_state_size(self):
        """
        Returns the __state space size

        :return: __state space size
        """

        return len(self.__state)

    def set_action(self, action):
        """
        Perform one action on the environment and update the __state and the score

        :param action: Action to be performed on the environment
        :return: Next __state, reward, score, done
        """

        if action.ndim == 1:
            action = np.expand_dims(action, axis=0)
        num_agents = action.shape[0]
        num_actions = action.shape[1]
        if 0 > num_agents > self.__num_agents:
            self.__logger.error(f"ERROR Number of Agents on Action Out of range: {action}  (003)")
        elif 0 > num_actions > self.__brain.vector_action_space_size:
            self.__logger.error(f"ERROR Number of Actions Out of range: {action}  (004)")
        else:
            self.__env_info = self.__env.step(action)[self.__brain_name]
            self.__state = self.__env_info.vector_observations
            self.__done = self.__env_info.local_done  # see if episode has finished
            self.__score += self.__env_info.rewards  # update the score
        return self.__state, self.__env_info.rewards, self.__score, self.__done

    def get_reward(self):
        """
        Returns the last Action reward

        :return: Last Action reward
        """

        return self.__env_info.rewards

    def get_score(self):
        """
        Return the actual simulation score

        :return: actual score
        """
        return self.__score

    @property
    def is_done(self):
        """
        Return True if environment simulation has finished or False if not

        :return: True or False if simulation is done
        """

        return self.__done
