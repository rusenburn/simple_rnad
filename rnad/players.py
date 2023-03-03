from abc import ABC,abstractmethod
import torch as T
import numpy as np
from .games.state import State
from .networks import PytorchNetwork
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
        
        probs_ar:np.ndarray = probs.cpu().numpy()[0]
        legal_actions_masks = state.legal_actions_masks()
        probs_ar = probs_ar * legal_actions_masks
        probs_ar /= probs_ar.sum()
        action = np.random.choice(len(probs_ar),p=probs_ar)
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
class RandomPlayer(PlayerBase):
    def __init__(self) -> None:
        super().__init__()
    
    def choose_action(self, state: State) -> int:
        masks = state.legal_actions_masks()
        legal_actions = np.argwhere(masks==1)
        legal_action = np.random.choice(len(legal_actions))
        action = legal_actions[legal_action]
        return action


