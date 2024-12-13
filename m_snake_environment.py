import random
import numpy as np

from m_environments import Environment, Environments

_BOARD_DICT = {0: ' W ',
               1: '   ',
               2: ' X ',
               3: ' O ',
               4: ' C ',
               5: ' A '}

_ACTIONS = {0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1)}


def get_available_actions() -> list:
    """
    Get available actions.

    Returns:
        list: Available actions.
    """
    return list(_ACTIONS.keys())


class SnakeEnvironment(Environment):
    def __init__(self, configuration: tuple) -> None:
        """
        Initialize the SnakeEnvironment class.

        Parameters:
            configuration (tuple): Configuration parameters for the environment.
        """
        super().__init__()

        # Initialize configuration
        self._height = configuration[0]  # type: int
        self._width = configuration[1]  # type: int
        self._apple_reward = configuration[2]  # type: float
        self._apples_amount = configuration[3]  # type: int
        self._turn_reward = configuration[4]  # type: float
        self._death_reward = configuration[5]  # type: float
        self._max_game_turns = configuration[6]  # type: int

        # Initialize state
        self._empty_fields = []  # type: list
        self._snake_pos = []  # type: list
        self._alive = False  # type: bool
        self._score = 0  # type: float
        self._turns = 0  # type: int

    def _environment_reset(self) -> None:
        """Reset the environment to its initial state."""
        self._state = np.zeros((self._height + 2, self._width + 2, 6), dtype=np.int8)
        self._empty_fields = []
        self._snake_pos = []
        self._alive = False
        self._score = 0
        self._turns = 0

    def _create_walls(self) -> None:
        """Create walls around the game board."""
        # Create left and right walls
        for i in range(self._height + 2):
            self._state[i, 0, 0] = 1
            self._state[i, self._width + 1, 0] = 1

        # Create top and bottom walls
        for i in range(1, self._width + 1):
            self._state[0, i, 0] = 1
            self._state[self._height + 1, i, 0] = 1

        # Fill empty fields
        for i in range(1, self._height + 1):
            for j in range(1, self._width + 1):
                self._state[i, j, 1] = 1
                self._empty_fields.append((i, j))

    def _spawn_snake(self) -> None:
        """Spawn the snake in the middle of the game board."""
        head_h = self._height // 2
        head_w = self._width // 2

        # Spawn snake from tail to head, from left to right
        for i in range(3):
            width = head_w - 2 + i
            self._state[head_h, width, 1] = 0  # Remove empty marker
            self._state[head_h, width, 4 - i] = 1  # Add body marker (4 - tail, 3 - body, 2 - head)
            self._empty_fields.remove((head_h, width))
            self._snake_pos.append((head_h, width))

        self._alive = True

    def _spawn_apples(self, amount: int = 1) -> bool:
        """
        Spawn apples on the game board.

        Parameters:
            amount (int): Number of apples to spawn.

        Returns:
            bool: True if apples were successfully spawned, False otherwise.
        """
        for _ in range(amount):
            if len(self._empty_fields) == 0:
                return False

            rnd_coords = random.choice(self._empty_fields)
            apple_h = rnd_coords[0]  # type: int
            apple_w = rnd_coords[1]  # type: int

            self._state[apple_h, apple_w, 1] = 0  # Remove empty marker
            self._state[apple_h, apple_w, 5] = 1  # Add apple marker
            self._empty_fields.remove((apple_h, apple_w))

        return True

    def _snake_move(self, direction: int) -> (bool, bool):
        """
        Move the snake in the specified direction.

        Parameters:
            direction (int): Direction of movement (0 - up, 1 - right, 2 - down, 3 - left).

        Returns:
            is_alive (bool): True while snake is not dead
            got_apple (bool): False until snake get apple

        """
        is_alive = True  # type: bool
        got_apple = False  # type: bool
        direction = _ACTIONS[direction]  # type: tuple # tuple with direction on hight and _width to move
        head_next_pos = list(map(lambda x, y: x + y, self._snake_pos[-1], direction))  # Calculate next position

        if self._state[head_next_pos[0], head_next_pos[1], 5] == 1:  # If an apple in position
            self._state[head_next_pos[0], head_next_pos[1], 5] = 0  # Remove apple marker
            self._state[head_next_pos[0], head_next_pos[1], 2] = 1  # Add head marker
            self._state[self._snake_pos[-1][0], self._snake_pos[-1][1], 2] = 0  # Remove head marker
            self._state[self._snake_pos[-1][0], self._snake_pos[-1][1], 3] = 1  # Add body marker
            self._snake_pos.append((head_next_pos[0], head_next_pos[1]))

            got_apple = True

        elif self._state[head_next_pos[0], head_next_pos[1], 1] == 1:  # If empty field in position
            self._state[head_next_pos[0], head_next_pos[1], 1] = 0  # Remove empty marker
            self._state[head_next_pos[0], head_next_pos[1], 2] = 1  # Add head marker
            self._state[self._snake_pos[-1][0], self._snake_pos[-1][1], 2] = 0  # Remove head marker
            self._state[self._snake_pos[-1][0], self._snake_pos[-1][1], 3] = 1  # Add body marker
            self._state[self._snake_pos[1][0], self._snake_pos[1][1], 3] = 0  # Remove last body marker
            self._state[self._snake_pos[1][0], self._snake_pos[1][1], 4] = 1  # Add tail marker
            self._state[self._snake_pos[0][0], self._snake_pos[0][1], 4] = 0  # Remove tail marker
            self._state[self._snake_pos[0][0], self._snake_pos[0][1], 1] = 1  # Add empty marker
            self._empty_fields.remove((head_next_pos[0], head_next_pos[1]))
            self._empty_fields.append((self._snake_pos[0][0], self._snake_pos[0][1]))
            del self._snake_pos[0]
            self._snake_pos.append((head_next_pos[0], head_next_pos[1]))

        elif self._state[head_next_pos[0], head_next_pos[1], 4] == 1:  # If a tail in position
            self._state[head_next_pos[0], head_next_pos[1], 4] = 0  # Remove tail marker
            self._state[head_next_pos[0], head_next_pos[1], 2] = 1  # Add head marker
            self._state[self._snake_pos[-1][0], self._snake_pos[-1][1], 2] = 0  # Remove head marker
            self._state[self._snake_pos[-1][0], self._snake_pos[-1][1], 3] = 1  # Add body marker
            self._state[self._snake_pos[1][0], self._snake_pos[1][1], 3] = 0  # Remove last body marker
            self._state[self._snake_pos[1][0], self._snake_pos[1][1], 4] = 1  # Add tail marker
            del self._snake_pos[0]
            self._snake_pos.append((head_next_pos[0], head_next_pos[1]))

        else:  # If an obstacle
            is_alive = False

        return is_alive, got_apple

    def reset(self) -> np.ndarray:
        """
        Start the environment by creating walls, spawning snake and apples.

        Returns:
            SnakeEnvironment: Current instance of the SnakeEnvironment.
        """
        self._environment_reset()
        self._create_walls()
        self._spawn_snake()
        self._spawn_apples(amount=self._apples_amount)
        return self._state.copy()

    def step(self, direction: int) -> tuple:
        """
        Perform one step in the environment by moving the snake in the given direction.

        Parameters:
            direction (int): Direction of movement (0 - up, 1 - right, 2 - down, 3 - left).

        Returns:
            tuple: Tuple containing the current state, reward, number of turns,
                   flag indicating if the episode is done, flag indicating if the game
                   was won, and flag indicating if the game was lost due to turns limit.
        """
        game_won = False  # type: bool
        lost_by_turns = False  # type: bool
        self._turns += 1
        self._alive, got_apple = self._snake_move(direction)

        if got_apple:
            self._score += self._apple_reward
            have_space = self._spawn_apples(amount=1)

            if not have_space:
                current_reward = self._apple_reward
                done = True
                game_won = True

            else:
                current_reward = self._apple_reward
                done = False

        elif not self._alive:
            current_reward = self._death_reward
            done = True

        else:
            current_reward = self._turn_reward
            done = False
            self._score += self._turn_reward

        if self._turns >= self._max_game_turns:
            done = True
            lost_by_turns = True

        return self._state.copy(), current_reward, done, self._turns, game_won, lost_by_turns

    def get_state(self) -> np.ndarray:
        """Return the current state of the environment."""
        return self._state.copy()

    def get_score(self) -> float:
        """Return the current score of the environment."""
        return self._score

    def get_turns(self) -> float:
        """Return the current turns of the environment."""
        return self._turns

    def state_print(self) -> None:
        """
        Print the current state of the environment.

        This method prints the current state using the BOARD_DICT mapping.
        """
        ch = ''
        for height in range(self._height + 2):
            for width in range(self._width + 2):
                for marker in range(len(_BOARD_DICT)):
                    if self._state[height, width, marker] == 1:
                        ch = _BOARD_DICT[marker]
                        break

                print(ch, end='')
            print()


class SnakeEnvironments(Environments):
    def __init__(self, environment_count: int = 1,
                 environment_type: type = None,
                 environment_config: tuple = None,
                 statistics_interval: int = 0) -> None:
        """
        Initialize the Environments class.

        Parameters:
            environment_count (int): Number of environments to manage.
            environment_type (class): Type of environment class to instantiate.
            environment_config (tuple): Configuration parameters for the environment.
            statistics_interval (int): Interval of statistics update.
        """
        super().__init__(environment_count, environment_type, environment_config)

        #  Initialize configuration
        self._environment_height = environment_config[0]
        self._environment_width = environment_config[1]
        self._environment_max_turns = environment_config[6]
        # Tuple of (environments count, timesteps, states)
        self._states_shape = (environment_count,
                              environment_config[6],
                              (environment_config[0] + 2) * (environment_config[1] + 2) * 6)
        self._converted_states = np.zeros(self._states_shape, dtype=np.bool_)
        self._last_turns = np.zeros(environment_count, dtype=np.int8)

        # Statistics
        self._scores = []  # type: list
        self._turns = []  # type: list
        self._avg_scores = []  # type: list
        self._avg_turns = []  # type: list
        self._games_played = 0  # type: int
        self._games_won = 0  # type: int
        self._lost_by_death = 0  # type: int
        self._lost_by_turns = 0  # type: int

        self._statistics_counter = 0  # Count games played
        self._statistics_interval = statistics_interval  # Interval of statistics update

        # Create and add new environments
        self._add_environments()

    def _clear_environments_data(self) -> None:
        """
        Clear all environment-related data.
        """
        self._converted_states = np.zeros(self._states_shape, dtype=np.bool_)
        self._last_turns = np.zeros(self._environment_count, dtype=np.int8)
        self._scores = []
        self._turns = []
        self._avg_scores = []
        self._avg_turns = []
        self._games_played = 0
        self._games_won = 0
        self._lost_by_death = 0
        self._lost_by_turns = 0
        self._statistics_counter = 0

    def _update_converted_state(self, env_index: int, timestep: int, state: np.ndarray) -> None:
        """
        Update the converted states array with a new _state.

        Parameters:
            env_index (int): Index of the environment.
            timestep (int): Index of the timestep.
            state (np.ndarray): New _state of the environment.
        """
        self._converted_states[env_index, timestep, :] = state.reshape(1, -1)

    def _restart_environment(self, env_index: int) -> np.ndarray:
        """
        Restart selected environment and return his _state

        Parameters:
            env_index (int): Index of the environment.
        """
        state = self._environments[env_index].reset()
        self._converted_states[env_index, :, :].fill(0)

        return state

    def _end_game_handler(self, score: float, turns: int, game_won: bool, lost_by_turns: bool) -> None:
        """
        Handle end game conditions

        Parameters:
            score (float): Environment _score.
            turns (int): Environment _turns.
            game_won (bool): Won|lost flag.
            lost_by_turns (bool): Lost due _turns limit flag.
        """
        self._scores.append(score)
        self._turns.append(turns)
        self._games_played += 1

        if game_won:
            self._games_won += 1
        else:
            if lost_by_turns:
                self._lost_by_turns += 1
            else:
                self._lost_by_death += 1

    def _add_environments(self) -> list:
        """
        Add multiple environments to the list and return their initial states.

        Returns:
            states (list): Initial states of the added environments.
        """
        # Clear environments
        states = []  # type: list
        self._clear_environments_data()

        # Create new environments
        for env_index in range(self._environment_count):
            new_environment = self._environment_type(self._environment_config)
            state = new_environment.reset()
            self._environments.append(new_environment)
            states.append(state)
            self._update_converted_state(env_index, 0, state)

        return states

    def reset(self) -> list:
        """
        Restart all environments and return their initial states.

        Returns:
            states (list): Initial states of the restarted environments.
        """
        # Clear environments
        states = []  # type: list
        self._clear_environments_data()

        # Restart environments
        for env_index in range(self._environment_count):
            state = self._environments[env_index].reset()
            states.append(state)
            self._update_converted_state(env_index, 0, state)

        return states

    def step(self, actions: list) -> (list, list, list, list, any):
        """
        Perform actions in all environments and return observations after the actions.

        Parameters:
            actions (list): List of actions to perform in each environment.

        Returns:
            states (list), rewards (list), dones(list), turns_list (list), statistics (None or tuple)
        """
        states = []  # type: list
        rewards = []  # type: list
        dones = []  # type: list
        turns_list = []  # type: list

        # Activate each environment by the action
        for env_index in range(self._environment_count):
            # Activate environment and get observation
            state, reward, done, turns, game_won, lost_by_turns\
                = self._environments[env_index].step(actions[env_index])

            if not done:  # if not done update converted_states and add turn
                self._update_converted_state(env_index, turns, state)
                self._last_turns[env_index] += 1
            else:  # else append scores and _turns. update data and restart environment
                score = self._environments[env_index].get_score()
                self._end_game_handler(score, turns, game_won, lost_by_turns)
                state = self._restart_environment(env_index)
                self._update_converted_state(env_index, 0, state)
                self._last_turns[env_index] = 0

            states.append(state)
            rewards.append(reward)
            dones.append(done)
            turns_list.append(turns)

        statistics = None
        if self._games_played - self._statistics_counter > self._statistics_interval:
            statistics = self.update_statistics()
            self._statistics_counter = self._games_played

        return states, rewards, dones, turns_list, statistics

    def convert_states(self, turns):
        """
        Calculate and convert the current states for the agent.

        Returns:
            converted_states (np.ndarray): States ready for agent use.
        """
        states_shape = (self._environment_count,
                        turns,
                        (self._environment_height + 2) * (self._environment_width + 2) * 6)

        converted_states = np.zeros(states_shape, dtype=np.bool_)

        for env_index in range(self._environment_count):
            if self._last_turns[env_index] < turns:
                converted_states[env_index, :, :]\
                    = self._converted_states[env_index, 0:turns, :]
            else:
                start_index = self._last_turns[env_index] - turns + 1
                end_index = self._last_turns[env_index] + 1

                converted_states[env_index, :, :]\
                    = self._converted_states[env_index, start_index:end_index, :]

        return converted_states

    def get_first_stats(self) -> (float, int):
        """
        Return the current score and turns of the first environment.

        Returns:
            float: Score
            int: Turns
        """
        return self._environments[0].get_score(), self._environments[0].get_turns()

    def get_environment_size(self) -> (int, int):
        """
        Get environment size (height, width).

        Returns:
            environment_height (int): Environment height.
            environment_width (int): Environment width.
        """
        return self._environment_height, self._environment_width

    def get_environment_count(self) -> int:
        """
        Get environment count.

        Returns:
            environment_count (int): Environment count.
        """
        return self._environment_count

    def get_environment_max_turns(self) -> int:
        """
        Get environment maximum turns.

        Returns:
            environment_max_turns (int): Environment maximum turns.
        """
        return self._environment_max_turns

    def get_info(self) -> (int, int, int, int):
        """
        Get environments info.

        Returns:
            games_played (int): Games played.
            games_won (int): Games won by agent.
            lost_by_turns (int): Games lost due turns limit.
            lost_by_death (int): Games lost due death.
        """
        return self._games_played, self._games_won, self._lost_by_turns, self._lost_by_death

    def update_statistics(self):
        length = len(self._scores)
        scores = self._scores.copy()
        turns = self._turns.copy()
        avg_scores = sum(self._scores) / length
        avg_turns = sum(self._turns) / length
        self._scores = list()
        self._turns = list()
        self._avg_scores.append(avg_scores)
        self._avg_turns.append(avg_turns)

        return scores, turns, self._avg_scores, self._avg_turns
