import numpy as np
import random
import logging as log

from collections import deque

from UnityEnv import UnityEnv
from .OUNoise import OUNoise
from .ActorNet import ActorNet
from .CriticNet import CriticNet
from .ReplayBuffer import ReplayBuffer

import torch
import torch.nn.functional as functnl
import torch.nn.utils as tchutils
import torch.optim as optim

logLevel_ = log.INFO


class Agent:
    """
    Agent that interacts with and learns from the environment.
    """

    def __init__(self, actormodel: ActorNet, criticmodel: CriticNet, epsl_start=1.0, epsl_end=0.01, epsl_decay=0.995,
                 buffer_size=int(1e6), batch_size=40, gamma=0.999, tau=1e-3, learn_rate_actor=1e-4,
                 learn_rate_critic=1e-4, weight_decay_actor=0.0, weight_decay_critic=0.0, update_target=20,
                 agent_id="Agent_DDPG", device=None, log_handler=None):
        """
        Agent object Constructor.

        :param actormodel (ActorNet): Actor Netwokr model implementation to be used
        :param criticmodel (CriticNet): Critic Netwokr model implementation to be used
        :param eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        :param eps_end (float): minimum value of epsilon
        :param eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        :param buffer_size (int): replay buffer size
        :param batch_size (int): training mini-batch size
        :param gamma (float): Q Training discount factor
        :param tau (float): soft update of target parameters
        :param learn_rate_actor (float): actor learning rate
        :param learn_rate_critic (float): critic learning rate
        :param weight_decay_actor (float): actor optimizer weight decay rate
        :param weight_decay_critic (float): critic optimizer weight decay rate
        :param update_target (int): how often to update the target network
        :param agent_id (string): ID used to identify the agent's saved files
        :param device: device where to load the network, if not specified the best device available will be selected
                       "cuda" or "cpu"
        :param log_Handler (handlers): Log handler to be used in the logging (Default is None)
        """

        global logLevel_

        self.__agent_id = agent_id
        # Set the error logger
        self.__logger = log.getLogger('Agent('+self.__agent_id+')')
        # Add handler to logger
        self.__logHandler = log_handler
        if self.__logHandler is not None:
            self.__logger.addHandler(self.__logHandler)
        else:
            self.__logger.debug(f"logHandler NOT defined (001)")
        # Set Logger Lever
        self.__logger.setLevel(logLevel_)
        self.__epsl_start = epsl_start
        self.__epsl_end = epsl_end
        self.__epsl_decay = epsl_decay
        self.__epsl = self.__epsl_start  # epsilon, for epsilon-greedy action selection
        self.__buffer_size = buffer_size
        self.__batch_size = batch_size
        self.__gamma = gamma
        self.__tau = tau
        self.__learn_rate_actor = learn_rate_actor
        self.__learn_rate_critic = learn_rate_critic
        self.__weight_decay_actor = weight_decay_actor
        self.__weight_decay_critic = weight_decay_critic
        self.__update_target = update_target
        if device is None:
            self.__logger.debug(f"Pytorch Cuda is available: {torch.cuda.is_available()} (002)")
            self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.__device = device
        self.__logger.debug(f"Pytorch Device set to {self.__device} (003)")

        # Actor Network
        self.__actor_network = actormodel.to(self.__device)
        self.__actor_state_size = self.__actor_network.get_state_size()
        self.__actor_action_size = self.__actor_network.get_action_size()
        self.__actor_optimizer = optim.Adam(self.__actor_network.parameters(), lr=self.__learn_rate_actor,
                                            weight_decay=self.__weight_decay_actor)

        # Target Actor Network
        self.__actor_network_target = type(self.__actor_network)(self.__actor_state_size, self.__actor_action_size,
                                                                 log_handler=self.__logHandler).to(self.__device)
        self.__actor_network_target.load_state_dict(self.__actor_network.state_dict())
        self.__actor_network_target.eval()  # Not calculate gradients on target network

        # Critic Network
        self.__critic_network = criticmodel.to(self.__device)
        self.__critic_state_size = self.__critic_network.get_state_size()
        self.__critic_action_size = self.__critic_network.get_action_size()
        self.__critic_values_size = self.__critic_network.get_values_size()
        self.__critic_optimizer = optim.Adam(self.__critic_network.parameters(), lr=self.__learn_rate_critic,
                                             weight_decay=self.__weight_decay_critic)

        # Target Critic Network
        self.__critic_network_target = type(self.__critic_network)(self.__critic_state_size, self.__critic_action_size,
                                                                   self.__critic_values_size,
                                                                   log_handler=self.__logHandler).to(self.__device)
        self.__critic_network_target.load_state_dict(self.__critic_network.state_dict())
        self.__critic_network_target.eval()  # Not calculate gradients on target network

        # Replay memory
        self.__buffer = ReplayBuffer(self.__actor_action_size, self.__buffer_size, self.__batch_size,
                                     log_handler=self.__logHandler)

        # Noise process
        self.__noise_generator = OUNoise(self.__actor_action_size)

        # Initialize time step (for updating every update_target steps)
        self.__time_step = 0

    def __del__(self):
        """
        Class Destructor

        """
        del self.__actor_state_size
        del self.__actor_action_size
        del self.__critic_state_size
        del self.__critic_action_size
        del self.__critic_values_size
        del self.__device
        del self.__epsl_start
        del self.__epsl_end
        del self.__epsl_decay
        del self.__epsl
        del self.__buffer_size
        del self.__batch_size
        del self.__gamma
        del self.__tau
        del self.__learn_rate_actor
        del self.__learn_rate_critic
        del self.__weight_decay_critic
        del self.__weight_decay_actor
        del self.__update_target
        del self.__actor_network
        del self.__actor_network_target
        del self.__actor_optimizer
        del self.__critic_network
        del self.__critic_network_target
        del self.__critic_optimizer
        del self.__buffer
        del self.__noise_generator
        del self.__time_step
        del self.__logHandler
        del self.__logger
        del self.__agent_id

    def __learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: (Tuple[numpy array]) tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        states = torch.from_numpy(states).float().to(self.__device)
        actions = torch.from_numpy(actions).float().to(self.__device)
        rewards = torch.from_numpy(rewards).float().to(self.__device)
        next_states = torch.from_numpy(next_states).float().to(self.__device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.__device)

        # ---------------------------- update critic network ---------------------------- #
        # Get predicted actions (for next states) from actor target model
        actor_actions_next = self.__actor_network_target(next_states).detach()
        # Get predicted values (for next states) from critic model
        critic_targets_next = self.__critic_network_target(next_states, actor_actions_next).detach()
        # Compute targets values for current states
        q_targets = rewards + (self.__gamma * critic_targets_next * (1 - dones))

        # Get expected values from local model
        q_expected = self.__critic_network(states, actions)

        # Compute Critic loss
        critic_loss = functnl.mse_loss(q_expected, q_targets)
        # Minimize the Critic loss
        self.__critic_optimizer.zero_grad()
        critic_loss.backward()
        tchutils.clip_grad_norm_(self.__critic_network.parameters(), 1)
        self.__critic_optimizer.step()

        # ---------------------------- update actor network ---------------------------- #
        # Update the actor network
        actor_actions = self.__actor_network(states)
        actor_loss = -self.__critic_network(states, actor_actions)
        actor_loss = actor_loss.mean()
        self.__actor_optimizer.zero_grad()
        actor_loss.backward()
        self.__actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.__soft_update(self.__critic_network, self.__critic_network_target)
        self.__soft_update(self.__actor_network, self.__actor_network_target)
        self.__logger.debug(f"Target Networks Updated (004)")

        self.__logger.debug(f"Learn Step Performed (005)")

    def __soft_update(self, local_model, target_model):
        """
        Soft update model parameters.

           θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model:  (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.__tau * local_param.data + (1.0 - self.__tau) * target_param.data)
        self.__logger.debug(f"Soft Update Performed (006)")

    def __hard_update(self, local_model, target_model):
        """
        Hard update model parameters.

           θ_target = θ_local

        :param local_model:  (PyTorch model) weights will be copied from
        :param target_model: (PyTorch model) weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
        self.__logger.debug(f"Hard Update Performed (007)")

    def restart_exploration(self):
        """
        Resume exploration
        """
        self.__epsl = self.__epsl_start
        self.__noise_generator.reset()

    def update_epsilon(self):
        """
            Update Epsilon
        """
        self.__epsl = max(self.__epsl_end, self.__epsl_decay * self.__epsl)  # decrease epsilon

    def get_epsilons(self):
        """
        Return Epsilon boundaries
        :return: epsilon, epsilon_end
        """
        return self.__epsl, self.__epsl_end

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
        self.__buffer.add(state, action, reward, next_state, done)
        self.__logger.debug(f"Replay Buffer increased to: {len(self.__buffer)} (008)")

        # Learn every self.__update_target time steps.
        self.__time_step = (self.__time_step + 1) % self.__update_target
        if self.__time_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.__buffer) >= self.__batch_size:
                experiences = self.__buffer.sample()
                self.__learn(experiences)

    def plearn(self, experiences, index):
        """
        Parallel Learning step
        :param experiences:
        :param index:
        """
        states, actions, rewards, next_states, dones = experiences

        states = states[:, index, :]
        actions = actions[:, index, :]
        next_states = next_states[:, index, :]
        dones = dones[:, index]
        dones = np.expand_dims(dones,axis=1)
        rewards = rewards[:, index]
        rewards = np.expand_dims(rewards, axis=1)
        agent_experiences = [states, actions, rewards, next_states, dones]
        self.__learn(agent_experiences)

    def actions(self, state, add_noise=True):
        """
        Returns action for given state as per current policy.

        :param state: (array_like) current state
        :param add_noise: (boolean) Add noise using the epsilon-greedy policy
        :return: Actions selected by actor network's policy
        """

        if state.ndim == 1:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.__device)
        elif state.ndim == 2:
            state = torch.from_numpy(state).float().to(self.__device)
        self.__actor_network.eval()  # set actor network on eval mode
        with torch.no_grad():
            action_values = self.__actor_network(state)
        self.__actor_network.train()  # set actor network on training mode

        # Continuos Epsilon-greedy action selection
        temp_act = action_values.cpu().data.numpy()
        if add_noise:
            self.__logger.debug(f"Epsilon Values: {self.__epsl}")
            if random.random() <= self.__epsl:
                # N = np.array([random.gauss(0,self.__epsl) for _ in range(self.__actor_action_size)])
                # N = np.array([random.gauss(0, 1) for _ in range(self.__actor_action_size)])
                # N = np.random.randn(self.__actor_action_size)  # select an action (for each agent)
                noise = self.__noise_generator.sample()
                temp_act += noise
                temp_act %= 1
                # temp_act = np.clip(temp_act, -1, 1)  # all actions between -1 and 1
                self.__logger.debug(f"Actor Values plus noise: {temp_act} (009)")

        self.__logger.debug(f"Actor Values: {temp_act} (010)")
        return temp_act

    def values(self, state, actions):
        """
        Returns values for given state and actions as per current critic network.

        :param state: (array_like): current state
        :param actions: (array_like): current actions
        :return: Values selected by critic network's
        """
        _state = torch.from_numpy(state).float().to(self.__device)
        _actions = torch.from_numpy(actions).float().to(self.__device)
        self.__critic_network.eval()  # set critic network on eval mode
        with torch.no_grad():
            critic_values = self.__critic_network(_state, _actions)
        self.__critic_network.train()  # set critic network on training mode

        temp_val = critic_values.cpu().data.numpy()
        self.__logger.debug(f"Critic Values: {temp_val} (011)")
        return temp_val

    def training(self, envrm: UnityEnv, mean_score_update_timeout=20000, n_episodes=int(2e6), max_time_steps=int(1e10),
                 exploration_tries=8):
        """
        Deep Q-Learning Agent Training

        :param envrm: Object that contain the training environment
        :param mean_score_update_timeout: maximum number of episodes where not improvement on the mean score was found
        :param n_episodes: (int) maximum number of training episodes
        :param max_time_steps: (int) maximum number of time steps per episode
        :param exploration_tries: (int) Number of exploration tries
        :return scores: [list] episodes' end scores
        """

        self.__logger.info(f"Training model on {self.__device} device (012)")
        scores = []  # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        mean_score = 0
        mean_score_update_timer = 0
        i_episode = 0
        expl_tries = exploration_tries
        check_point_saved = False
        self.__epsl = self.__epsl_start  # initialize epsilon
        for i_episode in range(1, n_episodes + 1):
            state = envrm.reset(train_mode=True)
            score = envrm.get_score()
            for t in range(max_time_steps):
                action = self.actions(state)
                self.__logger.debug(f"Episode: {i_episode} Step: {t} Action: {action} (013)")
                next_state, reward, score, done = envrm.set_action(action)
                self.learn_step(state, action, reward, next_state, done)
                state = next_state
                if np.any(done):
                    score = np.squeeze(score, axis=(0, 1))
                    self.__logger.debug(f"Episode: {i_episode} DONE after {t} Steps (014)")
                    break
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save episodes scores
            self.__epsl = max(self.__epsl_end, self.__epsl_decay * self.__epsl)  # decrease epsilon
            new_mean_score = np.mean(scores_window)
            self.__logger.info(f'Episode {i_episode:d}\tScore: {score:.2f} (015)')
            if i_episode % 100 == 0:
                self.__logger.info(f'Episode {i_episode:d}\tAverage Score: {new_mean_score:.2f} (016)')
            # if has been to much time and not improvement on the mean score then begin a new agent's exploration
            if (expl_tries > 0) and (self.__epsl <= self.__epsl_end) and \
                    (i_episode % (mean_score_update_timeout / 16) == 0) and \
                    (mean_score_update_timer > (mean_score_update_timeout / 4)):
                # initialize epsilon to restart the exploration
                self.__epsl = self.__epsl_start
                self.__noise_generator.reset()
                expl_tries -= 1
                self.__logger.info(
                    f'Episode {i_episode:d}\tNew Exploration begun ({expl_tries} remaining) (017)')
            if new_mean_score > mean_score:
                # New improvement found on the mean score, checkpoint is saved and training timer is restarted
                self.__actor_network.save(f'.\checkpoints\{self.__agent_id}-checkpoint-{new_mean_score:.4f}-actor.pth')
                self.__critic_network.save(f'.\checkpoints\{self.__agent_id}-checkpoint-{new_mean_score:.4f}-critic.pth')
                self.__logger.info(f'Environment checkpoint saved at {i_episode:d} episodes\tAverage Score: '
                                   f'{new_mean_score:.2f} (018)')
                mean_score = new_mean_score
                mean_score_update_timer = 0
                expl_tries = exploration_tries
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

    def save(self, actor_file_name, critic_file_name):
        """
        Save the network to file_name

        :param actor_file_name:  File where to save the actor network
        :param critic_file_name:  File where to save the critic network
        """
        self.__actor_network.save(actor_file_name)
        if hasattr(self, '_Agent__critic_network'):
            self.__critic_network.save(critic_file_name)
        else:
            self.__logger.info(f'Critic Network NOT saved because is not on Training Mode (021)')

    def load(self, actor_file_name, critic_file_name=None, training_mode=True):
        """
        Load Network from file_name

        :param actor_file_name: File from where to load actor network
        :param critic_file_name: File from where to load critic network
        :param training_mode: if true the critic network is required and the training target networks are created,
               if false only the actor network is requires to save memory
        :return: two NamedTuple with missing_keys and unexpected_keys fields, one for each network (load_actor_result,
                 load_critic_result), if training_mode is False then load_critic_result is None.
        """

        load_critic_result = None
        load_actor_result = self.__actor_network.load(actor_file_name)
        if training_mode:
            # create the target actor network
            self.__actor_network_target = type(self.__actor_network)(self.__actor_state_size, self.__actor_action_size,
                                                                     log_handler=self.__logHandler).to(self.__device)
            self.__hard_update(self.__actor_network, self.__actor_network_target)
            self.__actor_network_target.eval()  # Not calculate gradients on target network
            if critic_file_name is not None:
                # Load the critic network
                load_critic_result = self.__critic_network.load(critic_file_name)
                # create the target actor network
                self.__critic_network_target = type(self.__critic_network)(self.__critic_state_size,
                                                                           self.__critic_action_size,
                                                                           self.__critic_values_size,
                                                                           log_handler=self.__logHandler).to(
                    self.__device)
                self.__hard_update(self.__critic_network, self.__critic_network_target)
                self.__critic_network_target.eval()  # Not calculate gradients on target network
        else:
            if hasattr(self, '_Agent__actor_network_target'):
                del self.__actor_network_target
            if hasattr(self, '_Agent__critic_network'):
                del self.__critic_network
            if hasattr(self, '_Agent__critic_network_target'):
                del self.__critic_network_target
        return load_actor_result, load_critic_result
