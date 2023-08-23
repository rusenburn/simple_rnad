import numpy as np


class CircularBuffer():
    def __init__(self, max_len: int, observation_shape: tuple, n_actions) -> None:
        self.max_len = max_len
        self.observation_shape = observation_shape
        self._current_size = 0
        self._index = 0
        self._observations = np.zeros(
            (max_len, *observation_shape), dtype=np.float32)
        self._actions = np.zeros((max_len,), dtype=np.int32)
        self._new_obs = np.zeros(
            (max_len, *observation_shape), dtype=np.float32)
        self._rewards = np.zeros((max_len,), dtype=np.float32)
        self._players = np.zeros((max_len,), dtype=np.int32)
        self._new_players = np.zeros((max_len,), dtype=np.int32)
        self._legal_actions = np.zeros((max_len, n_actions), dtype=np.int32)
        self._new_legal_actions = np.zeros(
            (max_len, n_actions), dtype=np.int32)
        self._terminals = np.zeros((max_len,), dtype=np.bool8)

    def save(self, obs: np.ndarray, new_obs: np.ndarray, action: int, reward: int, player: int, new_player: int, legal_actions: np.ndarray, new_legal_actions: np.ndarray, is_terminal: bool):
        index = self._index
        self._observations[index] = obs
        self._new_obs[index] = new_obs
        self._actions[index] = action
        self._rewards[index] = reward
        self._players[index] = player
        self._new_players[index] = new_player
        self._legal_actions[index] = legal_actions
        self._new_legal_actions[index] = new_legal_actions
        self._terminals[index] = is_terminal

        self._index += 1
        self._current_size += 1
        if self._current_size >= self.max_len:
            self._current_size = self.max_len
        if self._index >= self.max_len:
            self._index = 0

    def sample(self, sample_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if sample_size > self._current_size:
            sample_size = self._current_size

        indices = np.random.choice(
            self._current_size, size=sample_size, replace=False)

        return (self._observations[indices].copy(), self._new_obs[indices].copy(), self._actions[indices].copy(),
                self._rewards[indices].copy(), self._players[indices].copy(
        ), self._new_players[indices].copy(), self._legal_actions[indices].copy(),
            self._new_legal_actions[indices].copy(),
            self._terminals[indices].copy())
