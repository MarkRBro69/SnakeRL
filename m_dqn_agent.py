import logging
import numpy as np
import random

from collections import deque
from tensorflow.keras import models

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


class ReplayBuffer:
    def __init__(self, max_size: int) -> None:
        """
        Initialize the replay buffer.

        Parameters:
            max_size (int): Maximum size of the buffer.
        """
        self._max_size = max_size  # Set the maximum size of the buffer
        self._buffer = deque(maxlen=max_size)  # Initialize the buffer with a maximum length

    def add(self, experience: tuple) -> None:
        """
        Add experience to the buffer.

        Parameters:
            experience (tuple): Experience to be stored in the buffer.
                                 Each experience is a tuple (states, actions, rewards, next_states, dones).
        """
        self._buffer.append(experience)  # Append each experience to the buffer

    def sample(self, batch_size: int) -> list:
        """
        Sample a batch of experiences from the buffer.

        Parameters:
            batch_size (int): Number of experiences to sample.

        Returns:
            list: A batch of sampled experiences.
        """
        return random.sample(self._buffer, batch_size)  # Randomly sample experiences from the _buffer

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer = deque(maxlen=self._max_size)  # Reinitialize the _buffer, clearing its contents

    def __len__(self) -> int:
        """
        Get the current size of the buffer.

        Returns:
            int: Current size of the buffer.
        """
        return len(self._buffer)  # Return the current number of experiences in the buffer


class DQNAgent:
    def __init__(self, model: models.Sequential, batch_size: int = 32, gamma: float = 0.95,
                 epsilon: float = 0, epsilon_decay: float = 0.995, epsilon_min: float = 0,
                 environment_count: int = 1, memory_size: int = 0, data_to_learn: int = 0,
                 replay_batch_size: int = 0, agent_turns: int = 1,
                 use_target_model: bool = False, target_model_upd_freq: int = 0,
                 use_replays: bool = False) -> None:
        """
        Initialize the DQN agent.

        Parameters:
            model (models.Sequential): Tensorflow NN model.
            batch_size (int): Batch size setup for the "fit" method.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
            epsilon_decay (float): Decay rate for the exploration rate.
            epsilon_min (float): Minimum exploration rate.
            environment_count (int): Number of environments.
            memory_size (int): Size of the replay buffer.
            data_to_learn (int): Data in memory to start learning
            replay_batch_size (int): Size of replay sample
            agent_turns (int): Number of agent turns.
            use_target_model (bool): Using target model flag (use if you have small amount of environments).
            target_model_upd_freq (int): Update frequency of target model.
            use_replays (bool): Using replays to learn.
        """
        # Initialize model
        self._model = model  # Tensorflow NN model

        # Initialize agent configuration
        self._batch_size = batch_size  # Batch size for model.fit
        self._gamma = gamma  # Discount factor for future rewards
        self._epsilon = float(epsilon)  # Initial exploration rate for epsilon-greedy policy
        self._epsilon_decay = float(epsilon_decay)  # Decay rate for the exploration rate
        self._epsilon_min = float(epsilon_min)  # Minimum value for the exploration rate
        self._environments_count = environment_count  # Number of parallel environments
        self._agent_turns = agent_turns  # Number of turns the agent takes
        self._use_target_model = use_target_model  # Using target model flag
        self._use_replays = use_replays  # Using replays to learn

        self._steps_made = 0  # Train steps made

        if use_replays:
            # Create replay _buffer
            self._memory_size = memory_size  # Maximum size of the replay buffer
            self._data_to_learn = data_to_learn  # Data in memory to start learning
            self._replay_batch_size = replay_batch_size  # Size of replay sample
            self._memory = ReplayBuffer(memory_size)  # Instantiate the replay buffer

        # Clone model if using target model
        if use_target_model:
            self._target_model = models.clone_model(self._model)  # Clone agent model
            self._target_model_upd_freq = target_model_upd_freq  # Initialize update frequency for target model

    def _update_target_model(self) -> None:
        """Update target model with weights from the main model."""
        self._target_model.set_weights(self._model.get_weights())

    def save_model(self, path):
        """
        Save trained model.

        Parameters:
            path (str): Path to the model.
        """
        self._model.save(path)

    def set_agent_turns(self, turns: int) -> None:
        """
        Set the number of agent turns and clear the memory.

        Parameters:
            turns (int): Number of agent turns.
        """
        self._agent_turns = turns  # Set the number of turns the agent takes
        self._memory.clear()  # Clear the replay buffer

    def act(self, states: np.ndarray, available_actions: list = None) -> (np.ndarray, list, list):
        """
        Choose actions based on the current policy.

        Parameters:
            states (np.ndarray): Current states.
            available_actions (list): List of available actions.

        Returns:
            Q-values  (np.ndarray), chosen actions (list), and random action flags (list).
        """
        q_values = self._model.predict(states, verbose=0)  # Predict Q-values for the given states
        actions = []  # type: list
        rand_actions = []  # type: list
        if available_actions is None:  # If available_actions is None, do not use epsilon
            for env_index in range(self._environments_count):
                rand_action = False
                action = np.argmax(q_values[env_index])  # Select the action with the highest Q-value
                actions.append(action)
                rand_actions.append(rand_action)
        else:
            for env_index in range(self._environments_count):
                if np.random.rand() <= self._epsilon:
                    rand_action = True
                    action = random.choice(available_actions)  # Select a random action
                else:
                    rand_action = False
                    action = np.argmax(q_values[env_index])  # Select the action with the highest Q-value

                actions.append(action)
                rand_actions.append(rand_action)

        return q_values, actions, rand_actions

    def model_fit(self, states: np.ndarray, q_values: np.ndarray, actions: list,
                  rewards: list, next_states: np.ndarray, dones: list):
        """
        Learning from current data and replay if enabled.

        Parameters:
            states (np.ndarray): States.
            q_values (np.ndarray): List of Q-values predicted for states
            actions (list): List of actions for states
            rewards (list): List of rewards after actions
            next_states (np.ndarray): List of next states after actions
            dones (list): List of dones after actions

        Returns:
            q_values (nd.array): Updated Q-Values.
        """
        if self._use_replays:
            if len(self._memory) < self._data_to_learn:
                return

        # Update Q-values based on current experiences
        updated_q_values = self._update_q_values(q_values, actions, rewards, next_states, dones)

        # Replay experiences if enabled
        if self._use_replays:
            replay_states, replay_q_values = self._replay()

            # Concatenate states and replay states
            combined_states = np.concatenate((states, replay_states), axis=0)
            # Concatenate updated Q-values and replay Q-values
            combined_q_values = np.concatenate((updated_q_values, replay_q_values), axis=0)
            # Shuffle data
            indices = np.random.permutation(len(combined_states))
            combined_states = combined_states[indices]
            combined_q_values = combined_q_values[indices]
        else:
            combined_states = states
            combined_q_values = updated_q_values

        # Model learning
        history = self._model.fit(combined_states, combined_q_values, epochs=1, batch_size=self._batch_size, verbose=0)
        logger.debug(f'Loss: {history.history["loss"]}')

        if self._use_target_model:
            if self._steps_made % self._target_model_upd_freq == 0 and self._steps_made > 0:
                self._update_target_model()  # Update target model

        # Epsilon decrease
        if self._epsilon > self._epsilon_min:
            self._epsilon *= self._epsilon_decay

        self._steps_made += 1

    def _update_q_values(self, q_values: np.ndarray, actions: list,
                         rewards: list, next_states: np.ndarray, dones: list) -> np.ndarray:
        """
        Updating Q-Values.

        Parameters:
            q_values (np.ndarray): List of Q-values predicted for states
            actions (list): List of actions for states
            rewards (list): List of rewards after actions
            next_states (np.ndarray): List of next states after actions
            dones (list): List of dones after actions

        Returns:
            q_values (nd.array): Updated Q-Values.
        """
        # Predict Q-values for next states
        if self._use_target_model:
            next_q_values = self._target_model.predict(next_states, verbose=0)
        else:
            next_q_values = self._model.predict(next_states, verbose=0)

        # Calculate target Q-values
        for env_index in range(len(q_values)):
            if not dones[env_index]:
                reward = rewards[env_index] + self._gamma * np.max(next_q_values[env_index])
                q_values[env_index, actions[env_index]] = reward
            else:
                q_values[env_index, actions[env_index]] = rewards[env_index]

        return q_values

    def memorize(self, experience):
        """
        Memorize experience by adding it to the memory.

        Parameters:
            experience tuple (state, action, reward, next state, done).
        """
        if self._use_replays:
            self._memory.add(experience)

    def _replay(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns states and Q-Values for learning.

        Returns:
            states (np.ndarray): States.
            q_values (np.ndarray): Q-Values.
        """
        batch = self._memory.sample(self._replay_batch_size)  # Sample a batch of experiences from the replay buffer
        states, actions, rewards, next_states, dones = zip(*batch)  # Unpack the batch into individual components

        # Convert each component into numpy array
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int8)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)

        # Predict Q-values
        q_values = self._model.predict(states, verbose=0)  # Predict Q-values for current states

        # Predict Q-values for next states
        if self._use_target_model:
            next_q_values = self._target_model.predict(next_states, verbose=0)
        else:
            next_q_values = self._model.predict(next_states, verbose=0)

        # Calculate target Q-values for each state
        for exp_index in range(self._replay_batch_size):
            if not dones[exp_index]:
                reward = rewards[exp_index] + self._gamma * np.max(next_q_values[exp_index])  # Calculate target Q-value
                q_values[exp_index, actions[exp_index]] = reward  # Update Q-value for the taken action
            else:
                q_values[exp_index, actions[exp_index]] = rewards[exp_index]  # If done, Q-value is the reward

        return states, q_values

    def get_agent_turns(self) -> int:
        """
        Get agent turns.

        Returns:
            agent_turns (int): Agent turns.
        """
        return self._agent_turns

    def get_epsilon(self) -> float:
        """
        Get current epsilon.

        Returns:
            epsilon (float): Agent epsilon.
        """
        return self._epsilon
