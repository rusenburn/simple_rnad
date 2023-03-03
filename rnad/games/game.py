from abc import ABC, abstractmethod
from typing import Callable, Sequence
import numpy as np
from .state import State


class Game(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def step(self, action: int) -> State:
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_actions(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def player_turn(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_space(self) -> tuple:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> State:
        raise NotImplementedError()

    @abstractmethod
    def game_result(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def is_terminal(self) -> bool:
        raise NotImplementedError()


class VecGame:
    def __init__(self, game_fns: Sequence[Callable[[], Game]]) -> None:
        self.game_fns = game_fns
        self.games = [game_fn() for game_fn in game_fns]
        self._n_games = len(game_fns)
        g = game_fns[0]()
        self._single_obs_space = g.observation_space
        self._n_actions = g.n_actions
        self.reset()

    def reset(self) -> tuple[np.ndarray, np.ndarray,np.ndarray,np.ndarray]:
        channels, rows, cols = self._single_obs_space
        partial_obs = np.zeros((self._n_games, channels, rows, cols), dtype=np.float32)
        full_obs = np.zeros((self._n_games, channels, rows, cols), dtype=np.float32)
        players = np.zeros((self._n_games), dtype=np.int32)
        legal_actions_masks = np.zeros((self._n_games,self._n_actions),dtype=np.int32)
        for i,game in enumerate(self.games):
            state = game.reset()
            player = game.player_turn
            partial_obs[i] = state.to_player_obs()
            full_obs[i] = state.to_full_obs()
            players[i] = player
            legal_actions_masks[i] = state.legal_actions_masks()
        return partial_obs,full_obs,legal_actions_masks,players

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray,np.ndarray, np.ndarray,np.ndarray, np.ndarray]:
        # player_partial_obs , full_obs , new_legal_actions_masks,rewards ,dones , current players
        action: int
        channels, rows, cols = self._single_obs_space
        player_obs = np.zeros((self._n_games, channels, rows, cols), dtype=np.float32)
        full_obs = np.zeros((self._n_games, channels, rows, cols), dtype=np.float32)
        new_legal_actions_masks = np.zeros((self.n_games,self._n_actions),dtype=np.int32)
        rewards = np.zeros((self._n_games, 2), dtype=np.float32)
        dones = np.zeros((self._n_games,), dtype=np.int32)
        players = np.zeros((self._n_games), dtype=np.int32)
        for i, (action, game) in enumerate(zip(actions, self.games)):
            # player = game.player_turn
            state = game.step(action)
            done = game.is_terminal()
            if done:
                rewards[i] = game.game_result()
                state = game.reset()
            new_legal_actions_masks[i]=  state.legal_actions_masks()
            player_obs[i] = state.to_player_obs()
            full_obs[i] = state.to_full_obs()
            dones[i] = done
            players[i] = game.player_turn
        return player_obs,full_obs, new_legal_actions_masks,rewards, dones, players
    
    @property
    def n_games(self)->int:
        return self._n_games
