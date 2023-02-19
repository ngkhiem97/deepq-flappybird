This repository has the Python implemetation of a Deep Q Network which is a reinforcement policy to train an agent with deep neural network.The resulted agent is able to play the famous 2013 mobile game Flappy Bird by deciding between pressing the command or not. 

Deep Q Networks (DQNs) are a type of artificial neural network that combines the power of deep learning with the principles of Q-learning. At their core, DQNs are designed to approximate the optimal Q-function, which is a mathematical formula used to evaluate the expected reward of taking a particular action in a given state. In traditional Q-learning, the Q-function is represented as a table that stores the expected reward for every possible combination of state and action. However, this approach quickly becomes impractical when dealing with high-dimensional state spaces or continuous action spaces, which are common in real-world scenarios. This is where DQNs come in. Instead of representing the Q-function as a table, they use a neural network to approximate it. This neural network takes in the current state as input and produces a set of Q-values for each possible action. The action with the highest Q-value is then selected as the optimal action.

Please see below the procedure on how to run the Python program to train the agent as well as how to run the demonstration of the network playing the video game.

## Requirements
```
numpy==1.19.5
opencv_python==4.6.0.66
pygame==2.1.0
torch==1.13.0
tqdm==4.60.0
```

## How to run the program

### Training the agent
```
python3 learn.py
```
