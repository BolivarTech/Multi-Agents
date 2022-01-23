# Project Details

This project implements two agent that interact with a Unity simulated
environment where it needs to play tennis one against the other.

The max score between the agents needs to be an average score of 0.5 or
more over 100 episodes to consider the environment solved.

# Getting Started

The Multi-Agents to run needs some dependencies to be installed before.

Because it uses the Unity Engine to run the environment simulator is
necessary to install it and the ML-Agents toolkit.

Clone the project's repository local on your machine

> git clone https://github.com/BolivarTech/Multi-Agents.git
>
> cd Multi-Agents

Run the virtual environment on the console

## Windows:

> .\\venv\\Scripts\\activate.bat

## Linux:

> source ./venv/Scripts/activate

Install the following modules on python.

-   Numpy (v1.19.5+)

-   torch (v1.8.2)

-   tensorflow (v1.7.1)

-   mlagents (v0.27.0)

-   unityagents

-   protobuf (v3.5.2)

To install it, on the projects root you need to run on one console

> pip install .

# Instructions

To run the agents you first need to activate the virtual environment.

## Windows:

> .\\venv\\Scripts\\activate.bat

## Linux:

> source ./venv/Scripts/activate

On the repository's root you can run the agent to get a commands'
description

usage: main.py \[-h\] \[-t\] \[-p\] \[-a1 ACTOR1\] \[-a2 ACTOR2\] \[-c1
CRITIC1\] \[-c2 CRITIC2\]

optional arguments:

-h, \--help show this help message and exit

-t, \--train Perform a model training, if -a not specified a new

model is trained.

-p, \--play Perform the model playing.

-a1 ACTOR1, \--actor1 ACTOR1 Path to a pytorch actor 1 model file.

-a2 ACTOR2, \--actor2 ACTOR2 Path to a pytorch actor 2 model file.

-c1 CRITIC1, \--critic1 CRITIC1 Path to a pytorch critic 1 model file.

-c2 CRITIC2, \--critic2 CRITIC2 Path to a pytorch critic 2 model file.

The -t option is used to train a new model or if -a1, -a2, -c1 and -c2
option are selected continues training the selected model.

The -p option is used to play the agent on the environment using the
model specified on the -a1 and -a2 flag, the -c flag is not needed.

The -h option shows the command's flags help

The suggested play models are the follow:

> python .\\main.py -p -a1 .\\models\\model-0-checkpoint-2.5200-actor.pth
-a2 .\\models\\model-1-checkpoint-2.4980-actor.pth
