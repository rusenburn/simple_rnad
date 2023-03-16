from abc import ABC, abstractmethod
from typing import Callable, Sequence
import numpy as np
import torch as T
from torch.distributions import Categorical
from rnad.networks import NetworkBase, PytorchNetwork
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

    @abstractmethod
    def render(self,full:bool)->None:
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
        # channels, rows, cols = self._single_obs_space
        partial_obs = np.zeros((self._n_games, *self._single_obs_space), dtype=np.float32)
        full_obs = np.zeros((self._n_games, *self._single_obs_space), dtype=np.float32)
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
        # channels, rows, cols = self._single_obs_space
        player_obs = np.zeros((self._n_games,* self._single_obs_space), dtype=np.float32)
        full_obs = np.zeros((self._n_games, *self._single_obs_space), dtype=np.float32)
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

class SinglePlayerGame(Game):
    def __init__(self,game_fn:Callable[[],Game],nn:PytorchNetwork) -> None:
        super().__init__()
        self.our_player = 0
        self.game = game_fn()
        self.nn = nn
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
    
    def step(self, action: int) -> State:
        state = self.step_until(player=self.our_player,action=action)
        return state
    
    def step_until(self,*,player:int,action:int)->State:
        assert self.player_turn == player
        new_state = self.game.step(action)
        self.game.is_terminal()
        while not new_state.is_terminal() and self.game.player_turn != player:
            obs = new_state.to_player_obs()
            obs_t = T.tensor(np.array([obs]),dtype=T.float32,device=self.device)
            with T.no_grad():
                probs:T.Tensor = self.nn(obs_t)
            probs_ar:np.ndarray = probs.cpu().numpy()[0]
            legal_actions = new_state.legal_actions_masks()
            probs_ar = probs_ar * legal_actions
            probs_ar/= probs_ar.sum()
            ca:int = np.random.choice(len(probs_ar),p=probs_ar)
            new_state = self.game.step(ca)
        return new_state

    @property
    def n_actions(self) -> int:
        return self.game.n_actions

    @property
    def observation_space(self) -> tuple:
        return self.game.observation_space

    @property
    def player_turn(self) -> int:
        assert self.our_player == self.game.player_turn
        return self.game.player_turn

    def reset(self) -> State:
        self.our_player = 1 - self.our_player
        return self.reset_until(self.our_player)
    
    def reset_until(self,player:int) -> State:
        new_state = self.game.reset()
        while not new_state.is_terminal() and self.game.player_turn != player:
            obs = new_state.to_player_obs()
            obs_t = T.tensor(np.array([obs]),dtype=T.float32,device=self.device)
            with T.no_grad():
                probs:T.Tensor = self.nn(obs_t)
            dist = Categorical(probs)
            chosen_action = dist.sample()
            ca = int(chosen_action.cpu().item())
            new_state = self.game.step(ca)
        return new_state

    def game_result(self) -> np.ndarray:
        return self.game.game_result()

    def is_terminal(self) -> bool:
        return self.game.is_terminal()

    def render(self, full: bool) -> None:
        self.game.render(full)






            

