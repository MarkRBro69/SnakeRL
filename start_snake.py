import time

import tensorflow as tf

import m_snake_interface
from m_dqn_agent import DQNAgent
from m_snake_environment import SnakeEnvironment, SnakeEnvironments

from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam


# def time_test(iterations, ag, env, env_count):
#     s_time = time.perf_counter()
#     for i in range(iterations):
#         environments._converted_states.copy()
#     ex_time = time.perf_counter() - s_time
#     print(f'ex time copy: {ex_time}')
#
#     s_time = time.perf_counter()
#     for i in range(iterations):
#         ag.act(environments._converted_states, list(environments._environments[0].directions.keys()))
#     ex_time = time.perf_counter() - s_time
#     print(f'ex time act_agent: {ex_time}')
#
#     actions = np.zeros(env_count)
#     s_time = time.perf_counter()
#     for i in range(iterations):
#         environments.step(actions=actions)
#     ex_time = time.perf_counter() - s_time
#     print(f'ex time act_env: {ex_time}')
#
#     q_values = np.zeros((env_count, 4))
#     s_time = time.perf_counter()
#     for i in range(iterations):
#         ag._model.fit(env._converted_states, q_values, epochs=1, batch_size=32, verbose=0)
#     ex_time = time.perf_counter() - s_time
#     print(f'ex time fit: {ex_time}')


def build_model(state_size, action_size, learning_rate) -> models.Sequential:
    """
    Build the neural network model.

    Returns:
        models.Sequential: The constructed neural network model.
    """
    nn_model = models.Sequential([
        layers.LSTM(128, input_shape=state_size, return_sequences=True),
        layers.LSTM(128),
        layers.Dense(action_size, activation='linear')
    ])

    nn_model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return nn_model


if __name__ == "__main__":
    # tf.config.set_visible_devices([], 'GPU')

    print(tf.config.list_physical_devices('GPU'))

    print(f"TensorFlow version: {tf.__version__}")

    all_devices = tf.config.list_physical_devices()

    gpu_devices = [device.name for device in all_devices if device.device_type == 'GPU']

    print("Available GPU:", gpu_devices)

    if gpu_devices:
        with tf.device('/GPU:0'):
            a = tf.random.normal([10000, 10000])
            b = tf.random.normal([10000, 10000])
            c = tf.matmul(a, b)

        start_time = time.time()
        with tf.device('/GPU:0'):
            _ = tf.matmul(a, b)
        elapsed_time = time.time() - start_time
        print(f"Time on GPU: {elapsed_time} seconds")
    else:
        print("No GPU devices.")

    height = 6
    width = 6
    apple_reward = 10
    apples_amount = 3
    turn_reward = -0.1
    death_reward = -10
    max_game_turns = 100

    statistic_interval = 1000

    agent_turns = 10

    environments_count = 100

    environment_config = (height, width, apple_reward, apples_amount, turn_reward, death_reward, max_game_turns)

    environments = SnakeEnvironments(environment_count=environments_count,
                                     environment_type=SnakeEnvironment,
                                     environment_config=environment_config,
                                     statistics_interval=statistic_interval)

    # model = build_model(state_size=(None, (height + 2) * (width + 2) * 6),
    #                     action_size=4, learning_rate=0.001)

    model = models.load_model('snake_model.keras')

    agent = DQNAgent(model=model, batch_size=32, gamma=0.95,
                     epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.001,
                     environment_count=environments_count,
                     memory_size=100_000, replay_batch_size=environments_count,
                     agent_turns=agent_turns, use_target_model=True, target_model_upd_freq=4, use_replays=True)

    # time_test(10, agent, environments, environments_count)

    gui = m_snake_interface.SnakeGUI(environments, agent, train=False, render_flag=True,
                                     graphs=True, save_every=100, iterations=100_000)

    gui.mainloop()
