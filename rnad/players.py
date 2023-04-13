from abc import ABC,abstractmethod
from typing import Sequence
import torch as T
import numpy as np
from .games.state import State
from .networks import PytorchNetwork,ActorNetwork
class PlayerBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def choose_action(self, state: State) -> int:
        '''
        Takes a state and returns an action
        '''
        raise NotImplementedError()

class NNPlayer(PlayerBase):
    def __init__(self,nnet:PytorchNetwork) -> None:
        super().__init__()
        self.nnet = nnet
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
    
    def choose_action(self, state: State) -> int:
        partial_observation = state.to_player_obs()
        partial_observation_tensor = T.tensor(np.array([partial_observation]),dtype=T.float32,device=self.device)
        with T.no_grad():
            probs :T.Tensor= self.nnet(partial_observation_tensor)
        if isinstance(probs,tuple):
            probs = probs[0]
        probs_ar:np.ndarray = probs.cpu().numpy()[0]
        legal_actions_masks = state.legal_actions_masks()
        masked_probs_ar = probs_ar * legal_actions_masks
        fixed_probs_ar = masked_probs_ar / masked_probs_ar.sum()
        action = np.random.choice(len(fixed_probs_ar),p=fixed_probs_ar)
        return action


class CheaterPlayer(PlayerBase):
    def __init__(self,nnet:PytorchNetwork) -> None:
        super().__init__()
        self.nnet = nnet
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
    
    def choose_action(self, state: State) -> int:
        full_obs = state.to_full_obs()
        full_obs_t = T.tensor(np.array([full_obs]),dtype=T.float32,device=self.device)
        with T.no_grad():
            probs :T.Tensor= self.nnet(full_obs_t)
        
        probs_ar:np.ndarray = probs.cpu().numpy()[0]
        legal_actions_masks = state.legal_actions_masks()
        probs_ar = probs_ar * legal_actions_masks
        probs_ar /= probs_ar.sum()
        action = np.random.choice(len(probs_ar),p=probs_ar)
        return action

class TurnBasedNNPlayer(PlayerBase):
    def __init__(self,actors:Sequence[ActorNetwork]) -> None:
        super().__init__()
        self.actors = actors
        self.players = [NNPlayer(actor) for actor in actors]
    
    def choose_action(self, state: State) -> int:
        player = self.players[state.player_turn]
        return player.choose_action(state)


class RandomPlayer(PlayerBase):
    def __init__(self) -> None:
        super().__init__()
    
    def choose_action(self, state: State) -> int:
        masks = state.legal_actions_masks()
        legal_actions = np.argwhere(masks==1)
        legal_action = np.random.choice(len(legal_actions))
        action = legal_actions[legal_action][0]
        return action

class HumanPlayer(PlayerBase):
    def __init__(self,full:bool) -> None:
        super().__init__()
        self.full = full
    
    def choose_action(self, state: State) -> int:
        state.render(self.full)
        action = int(input("Enter your action:\n"))
        return action
    




