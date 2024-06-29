# Deep Q Network AI Flappy Bird Project

## Overview

This project implements a Deep Q Network (DQN) to play the Flappy Bird game using reinforcement learning. The AI is trained to maximize its score by learning the optimal actions to take at each state of the game.

## Motivation

My motivation behind this project was to delve deeper into reinforcement learning, specifically the implementation and training of Deep Q-Networks.
I also wanted a visual representation of what the agent is doing.
I saw the classic Flappy Bird game project and some implementations using the NEAT algorithm to play the game.
I wanted to take this further by using a DQN to play the game because I believed it offered a suitable training environment that would be easy to change and maintain. 
I also wanted a project with a nice visible aspect so I could share it with my family, allowing them to understand what exactly was going on as a learning tool.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features

- Uses Deep Q Network (DQN) for training an AI to play Flappy Bird
- Includes experience replay and epsilon-greedy strategy
- Saves and loads model weights to/from files
- Visualizes training progress with real-time plots

## Installation

To get started with this project, clone the repository and install the required dependencies following the commands below. 

```sh
git clone https://github.com/yourusername/AI-Flappy-Bird-Python-Project.git
cd AI-Flappy-Bird-Python-Project
pip install -r requirements.txt
```
## Usage

To run this project simply execute the main.py file to begin.
The model can be trained by running the `main.py` script whuch involves the following steps.
  - Initialize the DQN, target network and optimizer.
  - load saved weights if available.
  - Starts the game and begins performing actions based on the epsilon-greedy strategy.
  - Store experiences in replay memory.
  - Sample a batch from the replay memory and perform a training episode.
  - Updates the target network, save weights and learning rates and specified intervals to allow continued training over sessions.

## Project Structure

AI-Flappy-Bird-Python-Project/
|
|--- FlappyBird.py           # Flappy Bird game environment
|--- main.py                 # Main script for training and running the AI
|--- bg.png                  # Background image
|--- pipe.png                # Pipe image
|--- bird1.png               # Bird image with wings up
|--- bird2.png               # Bird image with wings neutral
|--- bird3.png               # Bird image with wings down
|--- base.png                # Image used for the floor
|--- Policy_weight/          # Directory to save policy network weights
|    |--- weights.pt
|--- Target_weight/          # Directory to save target network weights
|    |--- weights.pt
|--- Optimizer_weight/       # Directory to save optimizer state
|    |--- weights.pt
|--- requirements.txt        # Python dependencies
|--- README.md               # Project documentation

## Reasoning and Aim

The primary aim of this project, other than gaining hands-on experience with reinforcement learning algorithms and techniques, was to create a fun learning tool with a visual aspect.

## Challenges and Future Work
- Exploration vs. Exploitation: Balancing exploration and exploitation to ensure the agent learns effectively.
- Hyperparameter Tuning: Finding the optimal hyperparameters for training the DQN.
- Scalability: Adapting the model to more complex enviroments and games.
- Reward Structure Choices: Adapting the reward structure to prevent the agent from flying into the ceiling and encouraging it to aim for a higher game score than in previous runs 

Future work may involve experimenting with different architectures, optimization techniques, and applying the model to more complex tasks.
There is also potential for creating an executable and a possible stretch goal of including an interface for end-users to change certain parameters of the agent and the game without breaking something.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Any contributions and help are welcome!
