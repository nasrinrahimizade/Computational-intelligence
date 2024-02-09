import numpy as np
import random

# Constants to represent players and states
AGENT = 1
OPPONENT = -1
NO_PLAYER = 0

class TicTacToe:
    def __init__(self):
        """Initialize the game board as a 3x3 array and set the current player."""
        self.board = np.zeros((3, 3))
        self.current_player = AGENT

    def make_move(self, move):
        """Make a move on the game board and switch players."""
        row, col = move
        self.board[row, col] = self.current_player
        self.current_player *= -1

    def valid_moves(self):
        """Return a list of valid moves (empty cells) on the current board."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == NO_PLAYER]

    def is_finished(self):
        """Check if the game is finished (either a player wins or it's a draw)."""
        return self.get_winner() != NO_PLAYER or np.all(self.board != NO_PLAYER)

    def get_winner(self):
        """Check for a winner and return the winner of the game."""
        for i in range(3):
            # Check rows and columns for a winner
            if np.all(self.board[i, :] == AGENT) or np.all(self.board[:, i] == AGENT):
                return AGENT
            if np.all(self.board[i, :] == OPPONENT) or np.all(self.board[:, i] == OPPONENT):
                return OPPONENT
            # Check diagonals for a winner
        if np.all(np.diag(self.board) == AGENT) or np.all(np.diag(np.fliplr(self.board)) == AGENT):
            return AGENT
        if np.all(np.diag(self.board) == OPPONENT) or np.all(np.diag(np.fliplr(self.board)) == OPPONENT):
            return OPPONENT
        return NO_PLAYER

class RLAgent:
    def __init__(self, epsilon=0.1, alpha=0.8):
        """Initialize the agent with exploration rate (epsilon) and learning rate (alpha)."""
        self.epsilon = epsilon
        self.alpha = alpha
        self.values = {}  # Dictionary to store state values (Q-values)

    def choose_action(self, env):
        """Choose an action based on the epsilon-greedy strategy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(env.valid_moves())
        else:
            return max(env.valid_moves(), key=lambda move: self.get_state_value(env.board))

    def get_state_value(self, state):
        """Get the value of a given state."""
        state_str = str(state)
        return self.values.get(state_str, 0.5)  # Default value for unseen states

    def update_state_value(self, state, reward):
        """Update the value of a given state using temporal difference."""
        state_str = str(state)
        self.values[state_str] = self.values.get(state_str, 0.5) + self.alpha * (reward - self.values.get(state_str, 0.5))

    def play(self, env):
        """Play a game using the RL agent."""
        while not env.is_finished():
            action = self.choose_action(env)
            env.make_move(action)
            reward = env.get_winner()
            self.update_state_value(env.board, reward)

def train_agent(agent, num_episodes=1000):
    """Train the agent for a specified number of episodes."""
    for _ in range(num_episodes):
        game = TicTacToe()
        agent.play(game)

def evaluate_agent(agent, num_games=1000):
    """Evaluate the agent's performance."""
    wins, draws = 0, 0
    for _ in range(num_games):
        game = TicTacToe()
        while not game.is_finished():
            if game.current_player == AGENT:
                action = agent.choose_action(game)
                game.make_move(action)
            else:
                action = random.choice(game.valid_moves())
                game.make_move(action)
        winner = game.get_winner()
        if winner == AGENT:
            wins += 1
        elif winner == NO_PLAYER:
            draws += 1
    return wins, draws

if __name__ == '__main__':
    rl_agent = RLAgent()
    train_agent(rl_agent, num_episodes=50000)
    wins, draws = evaluate_agent(rl_agent, num_games=1000)
    print(f"Agent Wins: {wins * 100 / 1000}%, Draws: {draws * 100 / 1000}%")
