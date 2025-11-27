import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class TwentyFortyEightEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.board_size = 4

        # Actions: 0 = up, 1 = down, 2 = left, 3 = right
        self.action_space = spaces.Discrete(4)

        # Board is 4x4 integers; observation is a flattened vector of size 16
        self.observation_space = spaces.Box(
            low=0,
            high=2**16,
            shape=(16,),
            dtype=np.int32
        )

        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        self.score = 0

        self._add_tile()
        self._add_tile()

        return self._get_obs(), {}

    def step(self, action):
        old_board = self.board[:][:]

        if action == 0:
            reward = self._move_up()
        elif action == 1:
            reward = self._move_down()
        elif action == 2:
            reward = self._move_left()
        elif action == 3:
            reward = self._move_right()

        # Invalid move (board unchanged)
        if np.array_equal(self.board, old_board):
            reward = -2  # small penalty for useless actions

        else:
            # Only add a tile after a valid move
            self._add_tile()

        done = not self._moves_available()

        return self._get_obs(), reward, done, False, {"score": self.score}

    # -------- Rendering -------- #

    def render(self):
        print("\nScore:", self.score)
        print("-" * 25)
        for row in self.board:
            print("|" + "|".join(f"{num:^5}" if num != 0 else "     " for num in row) + "|")
            print("-" * 25)

    # -------- Helper Methods -------- #

    def _get_obs(self):
        return self.board.flatten()

    def _add_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return
        i, j = random.choice(empty)
        self.board[i, j] = 4 if random.random() < 0.1 else 2

    def _moves_available(self):
        if np.any(self.board == 0):
            return True
        for i in range(4):
            for j in range(4):
                if j < 3 and self.board[i, j] == self.board[i, j + 1]:
                    return True
                if i < 3 and self.board[i, j] == self.board[i + 1, j]:
                    return True
        return False

    # -------- Movement Logic -------- #

    def _compress(self, row):
        new = row[row != 0]
        return np.concatenate([new, np.zeros(4 - len(new), dtype=np.int32)])

    def _merge(self, row):
        score_gain = 0
        for i in range(3):
            if row[i] != 0 and row[i] == row[i + 1]:
                row[i] *= 2
                row[i + 1] = 0
                score_gain += row[i]
        return row, score_gain

    def _move_left(self):
        total_gain = 0
        new_board = np.zeros((self.board_size, self.board_size), dtype=np.int32)

        for i in range(4):
            row = self._compress(self.board[i])
            row, gain = self._merge(row)
            row = self._compress(row)

            new_board[i] = row
            total_gain += gain

        self.board = new_board
        self.score += total_gain
        return float(total_gain)

    def _move_right(self):
        self.board = np.fliplr(self.board)
        reward = self._move_left()
        self.board = np.fliplr(self.board)
        return reward

    def _move_up(self):
        self.board = self.board.T
        reward = self._move_left()
        self.board = self.board.T
        return reward

    def _move_down(self):
        self.board = self.board.T
        reward = self._move_right()
        self.board = self.board.T
        return reward
