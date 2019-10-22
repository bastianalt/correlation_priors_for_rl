# Introduction
This is the companion code to the paper 'Correlation Priors for Reinforcement Learning' [B. Alt, A. Sosic and H Koeppl].

Simulations similar to the ones used in the paper can be run with this Python code.

# Installation
The code has been tested on MacOs 10.14.6.
## Prerequisites
This code has been tested with Python 3.6.8. For a local installation of a new Python version use, e.g., pyenv with virtualenv.

**Run:**

`pip install pyenv`

`pyenv install 3.6.8`

`pip install virtualenv`

`virtualenv venv -p ~/.pyenv/versions/3.6.8/bin/python`

`source venv/bin/activate`

## Installing the Dependencies
For the installation of the dependencies run the bash script 'setup.sh'.

**Run:** 

`./setup.sh`

# Simulation
The simulation files can be found in the simulation folder. For creating plots the scripts in the evaluation folder can be used.
## Imitation Learning
### Learning from Demonstration
**Run:** 

`python simulations/sim_lfd.py`
### Sub-Goal Modelling
**Run:** 

`python simulations/sim_subgoal.py`

## System Identification
**Run:** 
 
`python simulations/sim_transition_model_learning.py`

For evaluation:
  
**Run:** 

`python evaluation/eval_transition_model_learning.py`

## Bayesian Reinforcement Learning
### Grid World
**Run:** 

`python simulations/sim_full_planning.py`

### Queueing Problem
**Run:** 
 
`python simulations/sim_queuing.py`


For evaluation:
  
**Run:** 

`python evaluation/eval_full_planning.py`
