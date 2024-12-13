# 🐍 DQN Snake Game

This project implements a Deep Q-Network (DQN) agent to play the classic Snake game. The agent learns to play by interacting with the environment and maximizing the rewards over time using reinforcement learning. This project demonstrates how reinforcement learning can be applied to a simple game like Snake to train an AI agent.

---

## ✨ Features

- **🎯 Reinforcement Learning**:  
  The agent uses DQN (Deep Q-Network) to learn how to play the Snake game by receiving rewards for actions that help it eat food and avoid collisions.

- **🎮 Game Environment**:  
  Custom implementation of the Snake game, where the agent controls the snake to eat food and grow, while avoiding collisions with walls and itself.

- **📈 Training and Evaluation**:  
  The agent is trained over multiple episodes to improve its performance. The trained model can be saved and evaluated after training.

- **👀 Real-time Visualization**:  
  The game environment is visualized in real-time during both training and evaluation phases.

---

## 🛠️ Technology Stack

- **📝 Programming Language**: Python
- **🤖 Reinforcement Learning**: TensorFlow, Keras
- **📚 Algorithm**: Deep Q-Network (DQN)
- **🎲 Game Engine**: Custom Python implementation of the Snake game
- **🔧 Libraries**: NumPy, tkinter (for visualization)

---

## 🚀 Installation and Usage

### 📥 Clone the Repository
1. Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/MarkRBro69/SnakeRL.git
    cd SnakeRL
    ```

### ⚙️ Set Up the Environment
2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 🎮 Run the Game
4. To **test the trained model** (`snake_model.keras`):  
    ```bash
    python snake_start.py
    ```

5. To **train your own model**:  
    1. Open `snake_start.py` and update the following lines:  
        - Uncomment:
          ```python
          # model = build_model(state_size=(None, (height + 2) * (width + 2) * 6),
          #                     action_size=4, learning_rate=0.001)
          ```
        - Comment:
          ```python
          model = models.load_model('snake_model.keras')
          ```
        - Set `train` to `True` in the configuration:
          ```python
          gui = m_snake_interface.SnakeGUI(environments, agent, train=True, render_flag=True,
                                     graphs=True, save_every=100, iterations=100_000)
          ```
    2. Run the script:
        ```bash
        python snake_start.py
        ```

---

## 👀 Real-Time Visualization

The Snake game environment will be visualized during both training and evaluation. The game window will display the snake's movements, score, and other details in real-time.

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests for:  
- 🌟 New features  
- 🐛 Bug fixes  
- 🚀 Performance improvements  

---

Enjoy coding! 💻🐍
