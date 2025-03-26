from typing import DefaultDict, Dict, Iterable, List
from collections import defaultdict
from abc import ABC, abstractmethod
from copy import deepcopy
import os.path

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
import torch.nn
from torch.optim import Adam

from rl2025.exercise3.networks import FCNetwork
from rl2025.exercise3.replay import Transition

class Agent(ABC):
    """Base class for Deep RL Exercise 3 Agents

    **DO NOT CHANGE THIS CLASS**

    :attr action_space (gym.Space): action space of used environment
    :attr observation_space (gym.Space): observation space of used environment
    :attr saveables (Dict[str, torch.nn.Module]):
        mapping from network names to PyTorch network modules

    Note:
        see https://gymnasium.farama.org/api/spaces/ for more information on Gymnasium spaces
    """

    def __init__(self, action_space: gym.Space, observation_space: gym.Space):
        """The constructor of the Agent Class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        """
        self.action_space = action_space
        self.observation_space = observation_space

        self.saveables = {}

    def save(self, path: str, suffix: str = "") -> str:
        """Saves saveable PyTorch models under given path

        The models will be saved in directory found under given path in file "models_{suffix}.pt"
        where suffix is given by the optional parameter (by default empty string "")

        :param path (str): path to directory where to save models
        :param suffix (str, optional): suffix given to models file
        :return (str): path to file of saved models file
        """
        torch.save(self.saveables, path)
        return path

    def restore(self, save_path: str):
        """Restores PyTorch models from models file given by path

        :param save_path (str): path to file containing saved models
        """
        dirname, _ = os.path.split(os.path.abspath(__file__))
        save_path = os.path.join(dirname, save_path)
        checkpoint = torch.load(save_path)
        for k, v in self.saveables.items():
            v.load_state_dict(checkpoint[k].state_dict())

    @abstractmethod
    def act(self, obs: np.ndarray):
        ...

    @abstractmethod
    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        ...

    @abstractmethod
    def update(self):
        ...


class DQN(Agent):
    """The DQN agent for exercise 3

    **YOU NEED TO IMPLEMENT FUNCTIONS IN THIS CLASS**

    :attr critics_net (FCNetwork): fully connected DQN to compute Q-value estimates
    :attr critics_target (FCNetwork): fully connected DQN target network
    :attr critics_optim (torch.optim): PyTorch optimiser for DQN critics_net
    :attr learning_rate (float): learning rate for DQN optimisation
    :attr update_counter (int): counter of updates for target network updates
    :attr target_update_freq (int): update frequency (number of iterations after which the target
        networks should be updated)
    :attr batch_size (int): size of sampled batches of experience
    :attr gamma (float): discount rate gamma
    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        learning_rate: float,
        hidden_size: Iterable[int],
        target_update_freq: int,
        batch_size: int,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay_strategy: str = "constant",
        epsilon_decay: float = None,
        exploration_fraction: float = None,
        **kwargs,
        ):
        """The constructor of the DQN agent class

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param learning_rate (float): learning rate for DQN optimisation
        :param hidden_size (Iterable[int]): list of hidden dimensionalities for fully connected DQNs
        :param target_update_freq (int): update frequency (number of iterations after which the target
            networks should be updated)
        :param batch_size (int): size of sampled batches of experience
        :param gamma (float): discount rate gamma
        :param epsilon_start (float): initial value of epsilon for epsilon-greedy action selection
        :param epsilon_min (float): minimum value of epsilon for epsilon-greedy action selection
        :param epsilon_decay (float, optional): decay rate of epsilon for epsilon-greedy action. If not specified,
                                                epsilon will be decayed linearly from epsilon_start to epsilon_min.
        """
        super().__init__(action_space, observation_space)

        STATE_SIZE = observation_space.shape[0]
        ACTION_SIZE = action_space.n

        # ######################################### #
        #  BUILD YOUR NETWORKS AND OPTIMIZERS HERE  #
        # ######################################### #
        self.critics_net = FCNetwork(
            (STATE_SIZE, *hidden_size, ACTION_SIZE), output_activation=None
            )

        self.critics_target = deepcopy(self.critics_net)

        self.critics_optim = Adam(
            self.critics_net.parameters(), lr=learning_rate, eps=1e-3
            )

        # ############################################# #
        # WRITE ANY HYPERPARAMETERS YOU MIGHT NEED HERE #
        # ############################################# #
        self.learning_rate = learning_rate
        self.update_counter = 0
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min

        self.epsilon_decay_strategy = epsilon_decay_strategy
        if epsilon_decay_strategy == "constant":
            assert epsilon_decay is None, "epsilon_decay should be None for epsilon_decay_strategy == 'constant'"
            assert exploration_fraction is None, "exploration_fraction should be None for epsilon_decay_strategy == 'constant'"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = None
        elif self.epsilon_decay_strategy == "linear":
            assert epsilon_decay is None, "epsilon_decay is only set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is not None, "exploration_fraction must be set for epsilon_decay_strategy='linear'"
            assert exploration_fraction > 0, "exploration_fraction must be positive"
            self.epsilon_exponential_decay_factor = None
            self.exploration_fraction = exploration_fraction
        elif self.epsilon_decay_strategy == "exponential":
            assert epsilon_decay is not None, "epsilon_decay must be set for epsilon_decay_strategy='exponential'"
            assert exploration_fraction is None, "exploration_fraction is only set for epsilon_decay_strategy='linear'"
            self.epsilon_exponential_decay_factor = epsilon_decay
            self.exploration_fraction = None
        else:
            raise ValueError("epsilon_decay_strategy must be either 'linear' or 'exponential'")
        # ######################################### #
        self.saveables.update(
            {
                "critics_net"   : self.critics_net,
                "critics_target": self.critics_target,
                "critic_optim"  : self.critics_optim,
                }
            )

    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**
        ** Implement both epsilon_linear_decay() and epsilon_exponential_decay() functions **
        ** You may modify the signature of these functions if you wish to pass additional arguments **

        This function is called before every episode and allows you to schedule your
        hyperparameters.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """

        def epsilon_linear_decay(timestep, max_timestep, epsilon_start, epsilon_min, exploration_fraction, cur_epsilon):
            """Linear epsilon decay strategy
            
            Linearly decays epsilon from epsilon_start to epsilon_min based on the exploration_fraction
            """
            ### PUT YOUR CODE HERE ###
            if cur_epsilon > epsilon_min:  # calculate only if epsilon hasn't reached epsilon_min
                # Linear decay: epsilon = m * current_timestep + b
                m = (epsilon_min - epsilon_start) / (exploration_fraction * max_timestep)
                b = epsilon_start
                return max(m * timestep + b, epsilon_min)   # constant after reaching epsilon_min
            return epsilon_min


        def epsilon_exponential_decay(timestep, max_timestep, epsilon_min, decay_factor, cur_epsilon):
            """Exponential epsilon decay strategy
            
            Exponentially decays epsilon from epsilon_start to epsilon_min using decay_factor
            """
            ### PUT YOUR CODE HERE ###
            if cur_epsilon > epsilon_min:  # calculate only if epsilon hasn't reached epsilon_min
                # Exponential Decay: epsilon_t+1 = r^(t/T) * epsilon_t
                r_expon = decay_factor ** (timestep / max_timestep)
                epsilon_next_t = r_expon * cur_epsilon
                return max(epsilon_next_t, epsilon_min) # constant after reaching epsilon_min
            return epsilon_min

        if self.epsilon_decay_strategy == "constant":
            pass
        elif self.epsilon_decay_strategy == "linear":
            # linear decay
            self.epsilon = epsilon_linear_decay(
                timestep, 
                max_timestep, 
                self.epsilon_start, 
                self.epsilon_min, 
                self.exploration_fraction,
                self.epsilon
            )
        elif self.epsilon_decay_strategy == "exponential":
            # exponential decay
            self.epsilon = epsilon_exponential_decay(
                timestep,
                max_timestep,
                self.epsilon_min, 
                self.epsilon_exponential_decay_factor,
                self.epsilon
            )
        else:
            raise ValueError("epsilon_decay_strategy must be either 'constant', 'linear' or 'exponential'")

    def act(self, obs: np.ndarray, explore: bool):
        """Returns an action (should be called at every timestep)

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        When explore is False you should select the best action possible (greedy). However, during
        exploration, you should be implementing an exploration strategy (like e-greedy). Use
        schedule_hyperparameters() for any hyperparameters that you want to change over time.

        :param obs (np.ndarray): observation vector from the environment
        :param explore (bool): flag indicating whether we should explore
        :return (sample from self.action_space): action the agent should perform
        """

        ### PUT YOUR CODE HERE ###
        if explore and np.random.rand() < self.epsilon:    #using numpy
            # Random action (exploration)
            action = self.action_space.sample()
        else:
            # Greedy action (exploitation)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)  # Convert obs to tensor
            with torch.no_grad():
                q_values = self.critics_net(obs_tensor)

            action = torch.argmax(q_values).item()
        return action

    def update(self, batch: Transition) -> Dict[str, float]:
        """Update function for DQN

        **YOU MUST IMPLEMENT THIS FUNCTION FOR Q3**

        This function is called after storing a transition in the replay buffer. This happens
        every timestep. It should update your network, update the target network at the given
        target update frequency, and return the Q-loss in the form of a dictionary.

        :param batch (Transition): batch vector from replay buffer
        :return (Dict[str, float]): dictionary mapping from loss names to loss values
        """
        # Unpack the batch
        states, actions, next_states, rewards, dones = batch
        
        # Convert actions to long tensor for gathering
        actions = actions.long()

        # Calculate current Q-values
         ### PUT YOUR CODE HERE ###
        rewards = rewards.view(64)
        dones = dones.view(64)
        q_values = self.critics_net(states).gather(1, actions.view(-1, 1)).squeeze(-1)

        # Calculate target Q-values
        with torch.no_grad():
            # Get max Q-value for next state from target network
            ### PUT YOUR CODE HERE ###
            next_q_values = self.critics_target(next_states)
            max_q_value = torch.max(next_q_values, dim=1)[0]
            # Calculate target Q-value using Bellman equation
            ### PUT YOUR CODE HERE ###
            targets = rewards + (1 - dones) * self.gamma * max_q_value
        
        # Calculate loss (Mean Squared Error)
        ### PUT YOUR CODE HERE ###
        q_loss = torch.nn.functional.mse_loss(q_values, targets)
        
        # Optimize the critic network
        ### PUT YOUR CODE HERE ###
        # Backpropagation
        self.critics_optim.zero_grad()  # Reset previous gradients to 0
        q_loss.backward()  # Backpropagate the loss
        self.critics_optim.step()  # Gradient descent step (updates the parameters)

        # Update target network if it's time
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            ### PUT YOUR CODE HERE ###
            self.critics_target.hard_update(self.critics_net)


        return {"q_loss": q_loss.item()} # item() turns a one-element tensor into a standard python number


class DiscreteRL(Agent):
    """The DiscreteRL Agent for Ex 3 using tabular Q-Learning without neural networks
    
    This agent implements standard Q-learning with a discretized state space for
    environments with continuous state spaces. Suitable for small state-action spaces.
    
    :attr gamma (float): discount factor for future rewards
    :attr epsilon (float): probability of choosing a random action for exploration
    :attr alpha (float): learning rate for Q-value updates
    :attr n_acts (int): number of possible actions in the environment
    :attr q_table (DefaultDict): table storing Q-values for state-action pairs
    :attr position_bins (np.ndarray): bins for discretizing position dimension
    :attr velocity_bins (np.ndarray): bins for discretizing velocity dimension

        ** YOU CAN CHANGE THE PROVIDED SETTINGS **

    """

    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        gamma: float = 0.99,
        epsilon: float = 0.99,
        learning_rate: float = 0.02,    #changed to learning_rate to match name in config, otherwise it isn't used
        **kwargs
    ):
        """Constructor of DiscreteRL agent

        :param action_space (gym.Space): environment's action space
        :param observation_space (gym.Space): environment's observation space
        :param gamma (float): discount factor gamma
        :param epsilon (float): epsilon for epsilon-greedy action selection
        :param alpha (float): learning rate alpha
        """
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.alpha: float = learning_rate
        self.n_acts: int = action_space.n
        
        super().__init__(action_space=action_space, observation_space=observation_space)
        
        # Initialize Q-table as defaultdict with default value of 0 for any new state-action pair
        # This avoids having to initialize all possible state-action pairs explicitly
        self.q_table: DefaultDict = defaultdict(lambda: 0)

        # For mountain car environment discretization - creates k bins for each dimension, e.g. k=8
        k=8
        # Position range: -1.2 to 0.6 (8 bins)
        self.position_bins = np.linspace(-1.2, 0.6, k)
        # Velocity range: -0.07 to 0.07 (8 bins)
        self.velocity_bins = np.linspace(-0.07, 0.07, k)

    def discretize_state(self, obs: np.ndarray) -> int:
        """Discretizes a continuous state observation into a unique integer identifier.

        Converts continuous observation values into discrete bins and creates
        a unique integer identifier for the discretized state.

        :param obs (np.ndarray): continuous state observation (position, velocity)
        :return (int): unique integer identifier for the discretized state
        """
        # Convert continuous position to discrete bin index
        position_idx = np.digitize(obs[0], self.position_bins) - 1
        
        # Convert continuous velocity to discrete bin index
        velocity_idx = np.digitize(obs[1], self.velocity_bins) - 1
        
        # Create a unique integer ID by combining position and velocity indices
        # This creates a unique ID for each discretized state using a simple hash function
        unique_state_id = position_idx * len(self.velocity_bins) + velocity_idx
        return unique_state_id

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        """Returns an action using epsilon-greedy action selection.

        With probability epsilon, selects a random action for exploration.
        Otherwise, selects the action with the highest Q-value for the current state.

        :param obs (np.ndarray): current observation state
        :param explore (bool): flag indicating whether exploration should be enabled
        :return (int): action the agent should perform (index from action space)
        """
        # Discretize the observation
        state = self.discretize_state(obs)

        # Epsilon-greedy action selection
        if explore and np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            # Get Q-values for all actions in current state
            q_values = [self.q_table[(state, a)] for a in range(self.n_acts)]
            # Return action with highest Q-value (randomly break ties)
            return np.random.choice(np.flatnonzero(q_values == np.max(q_values)))

    def update(
        self, obs: np.ndarray, action: int, reward: float, n_obs: np.ndarray, done: bool
    ) -> float:
        """Updates the Q-table based on agent experience using Q-learning algorithm.

         ** YOU NEED TO IMPLEMENT THIS FUNCTION FOR Q3 BUT YOU CAN REUSE YOUR Q LEARNING CODE FROM Q2 (you can include it here or you adapt the files from Q2 to work of the mountain car problem **

        Implements the Q-learning update equation:
        Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        :param obs (np.ndarray): current observation state
        :param action (int): applied action
        :param reward (float): received reward
        :param n_obs (np.ndarray): next observation state
        :param done (bool): flag indicating whether episode is done
        :return (float): updated Q-value for current observation-action pair
        """


        # Convert continuous observations to discrete state identifiers
        state = self.discretize_state(obs)         # Current state
        next_state = self.discretize_state(n_obs)   # Next state

        ### PUT YOUR CODE HERE ###
        q_old = self.q_table[(state, action)]
        q_max = max([self.q_table[(next_state, a)] for a in range(self.n_acts)])
        self.q_table[(state, action)] = q_old + self.alpha * (reward + self.gamma * q_max - q_old)
        return self.q_table[(state, action)]

        return {f"Q_value_{state}" : self.q_table[(state, action)]}


    def schedule_hyperparameters(self, timestep: int, max_timestep: int):
        """Updates the hyperparameters (specifically epsilon for exploration).

        ** YOU CAN CHANGE THE PROVIDED SCHEDULING **

        Implements a linear decay schedule for epsilon, reducing from 1.0 to 0.01
        over the first 20% of total timesteps.

        :param timestep (int): current timestep at the beginning of the episode
        :param max_timestep (int): maximum timesteps that the training loop will run for
        """
        decay_progress = min(1.0, timestep / (0.20 * max_timestep))
        self.epsilon = 1.0 - decay_progress * 0.99  # Decays from 1.0 to 0.01

