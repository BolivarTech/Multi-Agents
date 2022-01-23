"""
Continuos Control

By: Julian Bolivar
Version: 1.0.0
"""
from collections import deque

import numpy as np

from PDDPG import ActorNet, CriticNet, Agent, PTrainer
from UnityEnv import UnityEnv

from argparse import ArgumentParser
import logging as log
import logging.handlers
import sys
import os

import csv
from datetime import datetime

# Main Logger
logHandler = None
logger = None
logLevel_ = logging.INFO
# OS running
OS_ = 'unknown'

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", required=False, action="store_true", help="Perform a model training, if -a "
                                                                                   "not specified a new model is "
                                                                                   "trained.")
    parser.add_argument("-p", "--play", required=False, action="store_true", help="Perform the model playing.")
    parser.add_argument("-a1", "--actor1", required=False, type=str, default=None,
                        help="Path to an pytorch actor 1 model file.")
    parser.add_argument("-a2", "--actor2", required=False, type=str, default=None,
                        help="Path to an pytorch actor 2 model file.")
    parser.add_argument("-c1", "--critic1", required=False, type=str, default=None,
                        help="Path to an pytorch critic 1 model file.")
    parser.add_argument("-c2", "--critic2", required=False, type=str, default=None,
                        help="Path to an pytorch critic 2 model file.")
    return parser


def save_scores(scores, computer_name):
    """

    :param scores:
    :return:
    """

    with open(f'{computer_name}-scores-{datetime.now().strftime("%Y%m%d%H%M%S")}.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(scores)


def train(actors_model_files=None, critics_model_files=None, computer_name="MultiAgent_App"):
    """
    Setup the training environment


    :param actors_model_files: file path with the actors model to be loaded
    :param critics_model_files: file path with the critics model to be loaded
    :param computer_name: String with the comnputer's name
    :return: None
    """

    global logHandler
    global logger
    global OS_
    global logLevel_

    u_env = None
    if OS_ == 'linux':
        u_env = UnityEnv(train_mode=True, env_filename="./SimEnv/Tennis_Linux/Tennis.x86_64", log_handler=logHandler)
    elif OS_ == 'win32':
        u_env = UnityEnv(train_mode=True, env_filename=".\\SimEnv\\Tennis_Windows_x86_64\\Tennis.exe", log_handler=logHandler)
    logger.info(f"Unity Environmet {OS_} loaded (001)")
    # number of agents in the environment
    logger.info(f'Number of agents: {u_env.get_num_agents()}')
    # number of actions
    logger.info(f'Number of actions: {u_env.get_num_actions()}')
    # examine the __state space
    logger.info(f'States look like: {u_env.get_state()}')
    logger.info(f'States have length: {u_env.get_state_size()}')
    # Generate the Agents
    agents = []
    for i in range(u_env.get_num_agents()):
        # Generate the Actor Network
        actNet = ActorNet(u_env.get_state_size(), u_env.get_num_actions(), log_handler=logHandler)
        # Generate the Critic Network
        critNet = CriticNet(u_env.get_state_size(), u_env.get_num_actions(), 1, log_handler=logHandler)
        agn = Agent(actormodel=actNet, criticmodel=critNet, log_handler=logHandler, agent_id=computer_name, device='cpu')
        agents.append(agn)
    if actors_model_files is not None and critics_model_files is not None:
        if len(agents) != len(actors_model_files):
            logger.error(f"Actors files and agensts numbers don't match")
            exit(1)
        elif len(agents) != len(critics_model_files):
            logger.error(f"Critics files and agensts numbers don't match")
            exit(1)
        for i in range(len(agents)):
            agents[i].load(actors_model_files[i],critics_model_files[i])
    # train the agent
    agnparatrain = PTrainer(agents, actions_size=u_env.get_num_actions(), trainer_id=computer_name, log_handler=logHandler)
    scores = agnparatrain.training(u_env)
    save_scores(scores,computer_name)
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

def play(actors_model_files, computer_name):
    """
    Perform an play using the agent

    :param actors_model_files: file path with the actor_model to be loaded
    :param computer_name: String with the comnputer's name
    """

    global logHandler
    global logger
    global OS_
    global logLevel_

    num_episodes = 100
    scores_window = deque(maxlen=num_episodes)  # last num_plays scores
    u_env = None
    if OS_ == 'linux':
        u_env = UnityEnv(train_mode=False, env_filename="./SimEnv/Tennis_Linux/Tennis.x86_64", log_handler=logHandler)
    elif OS_ == 'win32':
        u_env = UnityEnv(train_mode=False, env_filename=".\\SimEnv\\Tennis_Windows_x86_64\\Tennis.exe", log_handler=logHandler)
    logger.info(f"Unity Environmet {OS_} loaded (002)")
    # Generate the Actor Network
    actors = []
    num_agents = u_env.get_num_agents()
    for i in range(num_agents):
        # Generate the Actor Network
        actNet = ActorNet(u_env.get_state_size(), u_env.get_num_actions(), log_handler=logHandler, device='cpu')
        actors.append(actNet)
    if actors_model_files is not None:
        if len(actors) != len(actors_model_files):
            logger.error(f"Actors files and agents numbers don't match")
            exit(1)
        for i in range(len(actors)):
            actors[i].load(actors_model_files[i])
            actors[i].eval()
            actors[i].set_grad_enabled(False)
    for i_episode in range(1, num_episodes + 1):
        states = u_env.reset(train_mode=False)
        scores = u_env.get_score()
        t = 0
        while True:
            actions = []
            for i in range(num_agents):
                action = actors[i].get_action(states[i])
                actions.append(action)
            logger.debug(f"Episode: {i_episode} Step: {t} Action: {actions} (013)")
            states, _, scores, dones = u_env.set_action(np.array(actions))
            if np.any(dones):
                scores = scores[0]
                logger.debug(f"Episode: {i_episode} DONE after {t} Steps (005)")
                logger.info(f"Episode: {i_episode} DONE with [{scores[0]:.4f} {scores[1]:.4f}] score (006)")
                break
            t += 1
        scores_window.append(np.max(scores))  # save most recent score
    mean_score = np.mean(scores_window)
    print(f"The mean score was {mean_score:.2f} over {num_episodes} episodes ")
    save_scores(scores_window,computer_name)

def main(computer_name):
    """
     Run the main function

    :param computer_name: String with the comnputer's name
    """

    global logger

    args = build_argparser().parse_args()
    actor1_model_file = args.actor1
    actor2_model_file = args.actor2
    critic1_model_file = args.critic1
    critic2_model_file = args.critic2
    if args.train and args.play:
        logger.error("Options Train and Play can't be used togethers (007)")
    elif args.train:
        if actor1_model_file is not None:
            logger.debug(f"Training option selected with actor file {actor1_model_file} (008)")
        else:
            logger.debug(f"Training option selected with new actor 1 model (009)")
        if critic1_model_file is not None:
            logger.debug(f"Training option selected with critic file {critic1_model_file} (010)")
        else:
            logger.debug(f"Training option selected with new critic 1 model (011)")
        if actor2_model_file is not None:
            logger.debug(f"Training option selected with actor file {actor2_model_file} (008)")
        else:
            logger.debug(f"Training option selected with new actor 2 model (009)")
        if critic2_model_file is not None:
            logger.debug(f"Training option selected with critic file {critic2_model_file} (010)")
        else:
            logger.debug(f"Training option selected with new critic 2 model (011)")
        actors_model_files = []
        critics_model_files = []
        actors_model_files.append(actor1_model_file)
        actors_model_files.append(actor2_model_file)
        critics_model_files.append(critic1_model_file)
        critics_model_files.append(critic2_model_file)
        train(actors_model_files, critics_model_files,computer_name)
    elif args.play:
        if critic1_model_file is not None:
            logger.warning(f"On Play mode critic file {critic1_model_file} is NOT needed (012)")
        if critic2_model_file is not None:
            logger.warning(f"On Play mode critic file {critic2_model_file} is NOT needed (012)")
        if actor1_model_file is None:
            logger.error(f"Play option selected without actor1 file (013)")
            exit(1)
        elif actor2_model_file is None:
            logger.error(f"Play option selected without actor2 file (013)")
            exit(1)
        actors_model_files = []
        actors_model_files.append(actor1_model_file)
        actors_model_files.append(actor2_model_file)
        play(actors_model_files, computer_name)
    else:
        logger.debug(f"Not option was selected with actor file {actor1_model_file} (015)")
        logger.debug(f"Not option was selected with critic file {critic1_model_file} (016)")


if __name__ == '__main__':

    computer_name = os.environ['COMPUTERNAME']
    loggPath = "."
    LogFileName = loggPath + '/' + computer_name +'-ccontrol.log'
    # Check where si running
    if sys.platform.startswith('freebsd'):
        OS_ = 'freebsd'
    elif sys.platform.startswith('linux'):
        OS_ = 'linux'
    elif sys.platform.startswith('win32'):
        OS_ = 'win32'
    elif sys.platform.startswith('cygwin'):
        OS_ = 'cygwin'
    elif sys.platform.startswith('darwin'):
        OS_ = 'darwin'
    if OS_ == 'linux':
        # loggPath = '/var/log/DNav'
        loggPath = './log'
        LogFileName = loggPath + '/' + computer_name +'-ccontrol.log'
    elif OS_ == 'win32':
        # loggPath = os.getenv('LOCALAPPDATA') + '\\DNav'
        loggPath = '.\\log'
        LogFileName = loggPath + '\\' + computer_name +'-ccontrol.log'

    # Configure the logger
    os.makedirs(loggPath, exist_ok=True)  # Create log path
    logger = log.getLogger('DCCtrl')  # Get Logger
    # Add the log message file handler to the logger
    logHandler = log.handlers.RotatingFileHandler(LogFileName, maxBytes=10485760, backupCount=5)
    # Logger Formater
    logFormatter = log.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                                 datefmt='%Y/%m/%d %H:%M:%S')
    logHandler.setFormatter(logFormatter)
    # Add handler to logger
    if 'logHandler' in globals():
        logger.addHandler(logHandler)
    else:
        logger.debug(f"logHandler NOT defined (017)")
        # Set Logger Lever
    # logger.setLevel(logging.INFO)
    logger.setLevel(logLevel_)
    # Start Running
    logger.debug(f"Running in {OS_} (018)")
    main(computer_name)
