import logging

import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from m_dqn_agent import DQNAgent
from m_snake_environment import get_available_actions, SnakeEnvironments


# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def all_eqs(arrays: list, length: int) -> tuple:
    """
    Check if all arrays are equal and return difference indices.

    Parameters:
        arrays (list): Arrays to check.
        length (int): Number of arrays.

    Returns:
        tuple: all equals, difference indices.
    """
    difference_indices = list()
    for i in range(length - 1):
        comparison = arrays[0] == arrays[i + 1]
        difference_indices.append(np.where(~comparison))

    all_equals = all(np.array_equal(arrays[0], st) for st in arrays[1:])

    return all_equals, difference_indices


class SnakeGUI(tk.Tk):
    def __init__(self, environments: SnakeEnvironments, agent: DQNAgent,
                 train: bool = False, render_flag: bool = False, graphs: bool = False,
                 save_every: int = 0, iterations: int = 1) -> None:
        """
        Initialize interface for agent and environments

        Parameters:
            environments (SnakeEnvironments): Environments.
            agent (DQNAgent): Agent.
            train (bool): Flag for training _model.
            render_flag (bool): Flag for rendering.
            graphs (bool): Flag for graph updating.
            save_every (int): Save model interval.
            iterations (int): Iterations to do.
        """
        super().__init__()

        # Environments and agent initialization
        self.environments = environments
        self.agent = agent

        # Calculate height and width of environment state (state size + walls)
        env_h, env_w = self.environments.get_environment_size()
        self.height = env_h + 2
        self.width = env_w + 2

        self._environment_count = environments.get_environment_count()
        self._environment_max_turns = environments.get_environment_max_turns()
        self._available_actions = get_available_actions()

        # Main frame configuration
        main_frame = tk.Frame(self, highlightbackground="black", highlightthickness=1)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Left frame configuration
        left_frame = tk.Frame(main_frame, highlightbackground="black", highlightthickness=1)
        left_frame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.W+tk.E)

        # Right frame configuration
        right_frame = tk.Frame(main_frame, highlightbackground="black", highlightthickness=1)
        right_frame.grid(row=0, column=1, sticky=tk.N+tk.S+tk.W+tk.E)

        # Bottom frame configuration
        bottom_frame = tk.Frame(main_frame, highlightbackground="black", highlightthickness=1)
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky=tk.N+tk.S+tk.W+tk.E)

        # Text frame in the right frame
        text_frame = tk.Frame(right_frame)
        text_frame.pack(side=tk.LEFT)

        # Canvas for environment rendering
        self.canvas = tk.Canvas(left_frame, width=self.width * 20, height=self.height * 20, bg="black")
        self.canvas.pack(side=tk.RIGHT)

        # Text setup
        self.score_label = tk.Label(text_frame, text="Score: 0", font=("Helvetica", 16))
        self.score_label.grid(row=0, column=0, sticky=tk.W)

        self.turns_label = tk.Label(text_frame, text="Turns: 0", font=("Helvetica", 16))
        self.turns_label.grid(row=1, column=0, sticky=tk.W)

        self.games_played = tk.Label(text_frame, text="Games played: 0", font=("Helvetica", 16))
        self.games_played.grid(row=2, column=0, sticky=tk.W)

        self.games_won = tk.Label(text_frame, text="Games won: 0", font=("Helvetica", 16))
        self.games_won.grid(row=3, column=0, sticky=tk.W)

        self.lost_by_turns = tk.Label(text_frame, text="Lost by turns: 0", font=("Helvetica", 16))
        self.lost_by_turns.grid(row=4, column=0, sticky=tk.W)

        self.lost_by_death = tk.Label(text_frame, text="Lost by death: 0", font=("Helvetica", 16))
        self.lost_by_death.grid(row=5, column=0, sticky=tk.W)

        # Buttons setup
        self.reset_button = tk.Button(text_frame, text="Reset Game", command=self.reset_game)
        self.reset_button.grid(row=6, column=0, sticky=tk.W)

        self.save_and_exit_button = tk.Button(text_frame, text="Save and Exit", command=self.save_and_exit)
        self.save_and_exit_button.grid(row=7, column=0, sticky=tk.W)

        self.bind("<KeyPress>", self.key_pressed)

        # Graphs setup
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.ax_scores = self.figure.add_subplot(221)
        self.ax_turns = self.figure.add_subplot(222)
        self.ax_avg_scores = self.figure.add_subplot(223)
        self.ax_avg_turns = self.figure.add_subplot(224)
        self.canvas_figure = FigureCanvasTkAgg(self.figure, bottom_frame)
        self.canvas_figure.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Check train condition
        if train:
            self.train_loop(render_flag=render_flag, graphs=graphs, save_every=save_every, epochs=iterations)
        else:
            self.game_loop(epochs=iterations)

    def reset_game(self) -> None:
        """Restart the environments."""
        self.environments.reset()

    def save_and_exit(self) -> None:
        """Save the model and exit the GUI."""
        self.agent.save_model('snake_model.keras')
        self.destroy()

    @staticmethod
    def key_pressed(event: tk.Event) -> None:
        """
        Handle key press events to control the game direction.

        Parameters:
            event (tk.Event): Event.
        """
        key_map = {"Up": 0, "Right": 1, "Down": 2, "Left": 3}
        if event.keysym in key_map:
            direction = key_map[event.keysym]
            print(direction)

    def game_loop(self, epochs: int = 1) -> None:
        """
        Main game loop for running the Snake game.

        Parameters
            epoch (int): Number of iterations to do.
        """
        states_list = self.environments.reset()
        state = states_list[0]  # Update state of first environment

        for e in range(epochs):
            # Update window
            self.render(state)
            self.update_idletasks()
            self.update()

            agent_turns = self.agent.get_agent_turns()
            converted_states = self.environments.convert_states(agent_turns)  # Prepare states for the agent

            # Activate agent with current states
            _, actions, _ = self.agent.act(converted_states)

            # Activate environments with agent actions
            states, _, _, _, statistics = self.environments.step(actions=actions)
            state = states[0]  # Update state of first environment

            # Update statistics
            if statistics is not None:
                self.update_plots(*statistics)

    def train_loop(self, render_flag: bool = False, graphs: bool = False, save_every: int = 0, epochs: int = 1) -> None:
        """
        Main training loop for the agent.

        Parameters:
            render_flag (bool): Flag for rendering.
            graphs (bool): Flag for graphs  updating.
            save_every (int): Save _model interval.
            epochs (int): Iterations to do.
        """
        states_list = self.environments.reset()
        state = states_list[0]  # Update state of first environment

        for e in range(epochs):
            print(f"Episode: {e + 1}/{epochs}")

            if render_flag:
                # Update window
                self.render(state)
                self.update_idletasks()
                self.update()

            agent_turns = self.agent.get_agent_turns()
            converted_states = self.environments.convert_states(agent_turns)  # Prepare states for the agent

            # Activate agent with current states
            q_values, actions, rand_actions = self.agent.act(converted_states, self._available_actions)

            # Activate environments with agent actions
            states, rewards, dones, turns, statistics = self.environments.step(actions=actions)
            state = states[0]  # Update state of first environment

            next_converted_states = self.environments.convert_states(agent_turns)  # Prepare next states for the agent

            # Add new experiences to the memory
            for env_index in range(self._environment_count):
                self.agent.memorize((converted_states[env_index], actions[env_index], rewards[env_index],
                                     next_converted_states[env_index], dones[env_index]))

            # Model learning
            self.agent.model_fit(converted_states, q_values, actions, rewards, next_converted_states, dones)

            # Update statistics
            if statistics is not None and graphs:
                self.update_plots(*statistics)

                # If average agent turns > current agent turns -> increase agent turns by 10
                avg_turns = statistics[3][-1]
                if avg_turns > self.agent.get_agent_turns():
                    a_t = int(avg_turns + 10)
                    if a_t <= self._environment_max_turns:
                        self.agent.set_agent_turns(a_t)
                    else:
                        a_t = self._environment_max_turns
                        self.agent.set_agent_turns(a_t)

            if e % save_every == 0 and save_every != 0:
                self.agent.save_model('snake_model.keras')

            logger.debug(self.agent.get_epsilon())

    def render(self, state: np.ndarray) -> None:
        """
        Render the current state of the environment.

        Parameter:
            state (np.ndarray): State to render
        """
        try:
            self.canvas.delete("all")
        except Exception as e:
            print("Error on deleting canvas:", e)
        colors = {0: "grey", 1: "white", 2: "red", 3: "blue", 4: "yellow", 5: "green"}

        for height in range(self.height):
            for width in range(self.width):
                for marker in range(6):
                    if state[height, width, marker] == 1:
                        self.canvas.create_rectangle(
                            width * 20, height * 20, (width + 1) * 20, (height + 1) * 20, fill=colors[marker]
                        )

        self.update_labels()

    def update_labels(self) -> None:
        """Update the text labels with the current game statistics."""
        score, turns = self.environments.get_first_stats()
        games_played, games_won, lost_by_turns, lost_by_death = self.environments.get_info()

        self.score_label.config(text=f"Score: {score:.2f}")
        self.turns_label.config(text=f"Turns: {turns}")
        self.games_played.config(text=f"Games played: {games_played}")
        self.games_won.config(text=f"Games won: {games_won}")
        self.lost_by_turns.config(text=f"Lost by turns: {lost_by_turns}")
        self.lost_by_death.config(text=f"Lost by death: {lost_by_death}")

    def update_plots(self, scores: list, turns: list, avg_scores: list, avg_turns: list) -> None:
        """
        Update the Matplotlib plots with the current scores and turns.

        Parameters:
            scores (list): Current environments scores.
            turns (list): Current environments turns.
            avg_scores (list): Current environments average scores.
            avg_turns (list): Current environments average turns.
        """
        self.ax_scores.clear()
        self.ax_turns.clear()
        self.ax_avg_scores.clear()
        self.ax_avg_turns.clear()

        self.ax_scores.plot(scores)
        self.ax_turns.plot(turns)
        self.ax_avg_scores.plot(avg_scores)
        self.ax_avg_turns.plot(avg_turns)

        self.ax_scores.set_title("Scores")
        self.ax_turns.set_title("Turns")
        self.ax_avg_scores.set_title("Average Scores")
        self.ax_avg_turns.set_title("Average Turns")

        self.ax_scores.set_xlabel("Episodes")
        self.ax_scores.set_ylabel("Score")

        self.ax_turns.set_xlabel("Episodes")
        self.ax_turns.set_ylabel("Turns")

        self.ax_avg_scores.set_xlabel("Episodes")
        self.ax_avg_scores.set_ylabel("Avg Score")

        self.ax_avg_turns.set_xlabel("Episodes")
        self.ax_avg_turns.set_ylabel("Avg Turns")

        self.canvas_figure.draw()
