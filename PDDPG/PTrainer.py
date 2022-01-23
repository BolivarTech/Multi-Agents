import numpy as np
import logging as log

from collections import deque

from UnityEnv import UnityEnv
from .ReplayBuffer import ReplayBuffer

logLevel_ = log.INFO


class PTrainer:
    """
    Agent that interacts with and learns from the environment.
    """

    def __init__(self, agents: list, actions_size =1, buffer_size=int(1e6), batch_size=400, update_target=10,
                 trainer_id="Parallel_Trainer_DDPG", log_handler=None):
        """
        Agent object Constructor.

        :param agents (list): Agents List
        :param actions_size (int): actions size
        :param buffer_size (int): replay buffer size
        :param batch_size (int): training mini-batch size
        :param update_target (int): how often to update the target network
        :param trainer_id (string): ID used to identify the agent's saved files
        :param log_Handler (handlers): Log handler to be used in the logging (Default is None)
        """

        global logLevel_

        self.__trainer_id = trainer_id
        # Set the error logger
        self.__logger = log.getLogger('PTrainer(' + self.__trainer_id + ')')
        # Add handler to logger
        self.__logHandler = log_handler
        if self.__logHandler is not None:
            self.__logger.addHandler(self.__logHandler)
        else:
            self.__logger.debug(f"logHandler NOT defined (001)")
        # Set Logger Lever
        self.__logger.setLevel(logLevel_)
        self.__agents = agents
        self.__actions_size = actions_size
        self.__buffer_size = buffer_size
        self.__batch_size = batch_size
        self.__update_target = update_target

        # Shared Replay memory
        self.__shared_buffer = ReplayBuffer(self.__actions_size, self.__buffer_size, self.__batch_size,
                                            log_handler=self.__logHandler)


        # Initialize time step (for updating every update_target steps)
        self.__time_step = 0

    def __del__(self):
        """
        Class Destructor

        """
        del self.__agents
        del self.__actions_size
        del self.__buffer_size
        del self.__batch_size
        del self.__update_target
        del self.__shared_buffer
        del self.__time_step
        del self.__logHandler
        del self.__logger
        del self.__trainer_id

    def learn_step(self, state, action, reward, next_state, done):
        """
        Perform one step on the learning process

        :param state: Environment current __state
        :param action: Current Action
        :param reward: Reward received by the selected accion
        :param next_state: Environment next __state
        :param done: Episode ended
        """

        # Save experience in replay memory
        self.__shared_buffer.add(state, action, reward, next_state, done)
        self.__logger.debug(f"Replay Buffer increased to: {len(self.__shared_buffer)} (008)")

        # Learn every self.__update_target time steps.
        self.__time_step = (self.__time_step + 1) % self.__update_target
        if self.__time_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.__shared_buffer) >= self.__batch_size:
                num_agents = len(self.__agents)
                experiences = self.__shared_buffer.sample()
                for i in range(num_agents):
                    self.__agents[i].plearn(experiences, i)


    def training(self, envrm: UnityEnv, mean_score_update_timeout=60000, n_episodes=int(2e6), max_time_steps=int(1e10),
                 exploration_tries=8):
        """
        Deep Q-Learning Multi Agent Training

        :param envrm: Object that contain the training environment
        :param mean_score_update_timeout: maximum number of episodes where not improvement on the mean score was found
        :param n_episodes: (int) maximum number of training episodes
        :param max_time_steps: (int) maximum number of time steps per episode
        :return scores: [list] episodes' end scores
        """

        num_agents = len(self.__agents)
        if envrm.get_num_agents() != num_agents:
            self.__logger.error(f"Number of agents is not equal to agent on environment (012)")
            return None
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        mean_score = 0
        mean_score_update_timer = 0
        i_episode = 0
        epsl = 0
        epsl_end = 1
        expl_tries = exploration_tries
        check_point_saved = False
        for i_episode in range(1, n_episodes + 1):
            states = envrm.reset(train_mode=True)
            score = envrm.get_score()
            for t in range(max_time_steps):
                actions = []
                for i in range(num_agents):
                    action = self.__agents[i].actions(states[i])
                    actions.append(action[0])
                self.__logger.debug(f"Episode: {i_episode} Step: {t} Action: {actions} (013)")
                next_states, reward, score, dones = envrm.set_action(np.array(actions))
                self.learn_step(states, actions, reward, next_states, dones)
                states = next_states
                if np.any(dones):
                    score = score[0]
                    self.__logger.debug(f"Episode: {i_episode} DONE after {t} Steps (014)")
                    break
            for i in range(num_agents):
                self.__agents[i].update_epsilon()
            scores_window.append(np.max(score))  # save most recent max score
            scores.append(score)  # save episodes scores
            new_mean_score = np.mean(scores_window)
            self.__logger.info(f'Episode {i_episode:d}\tScore: [{score[0]:.4f}, {score[1]:.4f}] Mean: {new_mean_score:.4f} (015)')
            if i_episode % 100 == 0:
                self.__logger.info(f'Episode {i_episode:d}\tAverage Score: {new_mean_score:.4f} (016)')
            # if has been to much time and not improvement on the mean score then begin a new agent's exploration
            for i in range(num_agents):
                t_epsl, t_epsl_end = self.__agents[i].get_epsilons()
                epsl = max(epsl, t_epsl)
                epsl_end = min(epsl_end,t_epsl_end)
            if (expl_tries > 0) and (epsl <= epsl_end) and \
                    (i_episode % (mean_score_update_timeout / 16) == 0) and \
                    (mean_score_update_timer > (mean_score_update_timeout / 4)):
                for i in range(num_agents):
                    # restart the exploration
                    self.__agents[i].restart_exploration()
                expl_tries -= 1
                self.__logger.info(
                    f'Episode {i_episode:d}\tNew Exploration begun ({expl_tries} remaining) (017)')
            if new_mean_score > mean_score:
                # New improvement found on the mean score, checkpoint is saved and training timer is restarted
                for i in range(num_agents):
                    actor_file_name = f'.\checkpoints\{self.__trainer_id}-{i}-checkpoint-{new_mean_score:.4f}-actor.pth'
                    critic_file_name = f'.\checkpoints\{self.__trainer_id}-{i}-checkpoint-{new_mean_score:.4f}-critic.pth'
                    self.__agents[i].save(actor_file_name, critic_file_name)
                self.__logger.info(f'Environment checkpoint saved at {i_episode:d} episodes\tAverage Score: '
                                   f'{new_mean_score:.4f} (018)')
                mean_score = new_mean_score
                check_point_saved = True
            if check_point_saved and (mean_score_update_timer > mean_score_update_timeout):
                # One checkpoint was saved and the training timer is timeout then finis the training
                self.__logger.info(f'Episode {i_episode:d}\tMean Score Update Time Out (019)')
                break
            else:
                # Continue the training time
                mean_score_update_timer += 1
        self.__logger.info(
            f'Episode {i_episode:d}\tBest Environment checkpoint saved at Average Score: {mean_score:.2f} (020)')
        return scores
