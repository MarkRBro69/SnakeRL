import numpy as np


class Environment:
    """
    Base environment class.

    Methods to overwrite:
        reset - Reset the environment to its initial state.
        step - Perform one step in the environment.
    """
    def __init__(self):
        self._state: np.ndarray = np.array([])  # State of environment

    def reset(self, *args) -> np.ndarray:
        """
        Reset the environment to its initial state.

        Parameters:
            *args: Custom data

        Returns:
            state (np.ndarray): State of environment
        """
        raise NotImplementedError("Subclasses must implement (reset) method")

    def step(self, action: int) -> tuple:
        """
        Perform one step in the environment.

        Parameters:
            action (int): Actions to do in environment

        Returns:
            tuple: state, reward, done, other data
        """
        raise NotImplementedError("Subclasses must implement (step) method")


class Environments:
    """
    Base environments class.

    Methods to overwrite:
        reset - Restart all environments and return their initial states.
        step - Perform actions in all environments and return observations after the actions.
    """
    def __init__(self, environment_count: int, environment_type: type, environment_config: tuple):
        """
        Initialization of environments.

        Parameters:
            environment_count (int): Number of environments.
            environment_type (type): Type of environments.
            environment_config (tuple): Configuration of environments.
        """
        self._environment_count: int = environment_count  # Number of environments
        self._environment_type: type = environment_type  # Environments type
        self._environment_config: tuple = environment_config  # Environments configuration
        self._environments: list = []  # Environments list

    def reset(self) -> list:
        """
        Restart all environments and return their initial states.

        Returns:
            list: Initial states of the restarted environments.
        """
        raise NotImplementedError("Subclasses must implement (reset) method")

    def step(self, actions: list) -> tuple:
        """
        Perform actions in all environments and return observations after the actions.

        Parameters:
            actions (list): List of actions to perform in each environment.

        Returns:
            tuple: states, rewards, dones, other data
        """
        raise NotImplementedError("Subclasses must implement (reset) method")
