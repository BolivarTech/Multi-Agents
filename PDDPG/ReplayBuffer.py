import numpy as np
import random
from collections import namedtuple, deque

import logging as log

logLevel_ = log.INFO


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size=int(1e5), batch_size=64, log_handler=None):
        """
        ReplayBuffer object Constructor.

          :param values_size (int): dimension of each action
          :param buffer_size (int): maximum size of buffer
          :param batch_size (int): size of each training batch
          :param log_Handler (handlers): Log handler to be used in the logging (Default is None)
        """

        global logLevel_

        # Set the error logger
        self.__logger = log.getLogger('ReplayBuffer')
        # Add handler to logger
        if log_handler is not None:
            self.__logger.addHandler(log_handler)
        else:
            self.__logger.debug(f"logHandler NOT defined (001)")
        # Set Logger Lever
        self.__logger.setLevel(logLevel_)
        self.__action_size = action_size
        self.__buffer = deque(maxlen=buffer_size)
        self.__batch_size = batch_size
        self.__experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.__lessThatMax = True

    def __del__(self):
        """
        Class destructor

        """
        del self.__action_size
        del self.__buffer
        del self.__batch_size
        del self.__experience
        del self.__logger

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.__experience(state, action, reward, next_state, done)
        self.__buffer.append(e)
        buffLen = len(self.__buffer)
        self.__logger.debug(f"Element added to Buffer, new size {buffLen} (002)")
        if len(self.__buffer) == self.__batch_size:
            self.__logger.info(f"Buffer reached batch size {buffLen} (003)")
        elif self.__lessThatMax and (len(self.__buffer) == self.__buffer.maxlen):
            self.__logger.info(f"Buffer reached maximum size {buffLen} (004)")
            self.__lessThatMax = False

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.__buffer, k=self.__batch_size)

        states = np.stack([e.state for e in experiences if e is not None])
        actions = np.stack([e.action for e in experiences if e is not None])
        rewards = np.stack([e.reward for e in experiences if e is not None])
        next_states = np.stack([e.next_state for e in experiences if e is not None])
        dones = np.stack([e.done for e in experiences if e is not None]).astype(np.uint8)
        return states, actions, rewards, next_states, dones

    def clear(self):
        """
        Clear and empty the buffer
        """
        self.__buffer.clear()
        self.__lessThatMax = True

    def __len__(self):
        """Return the current size of internal __shared_buffer."""
        return len(self.__buffer)
