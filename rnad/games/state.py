from abc import ABC,abstractmethod
import numpy as np

class State(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @property
    @abstractmethod
    def n_actions(self)->int:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def observation_space(self)->tuple:
        raise NotImplementedError()
    
    @property
    @abstractmethod
    def player_turn(self)->int:
        raise NotImplementedError()

    @abstractmethod
    def step(self,action:int)->'State':
        raise NotImplementedError()
    
    @abstractmethod
    def is_terminal(self)->bool:
        raise NotImplementedError()
    
    @abstractmethod
    def game_result(self)->np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def legal_actions_masks(self)->np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def to_player_obs(self)->np.ndarray:
        raise NotImplementedError()
    
    @abstractmethod
    def to_player_short(self)->tuple:
        raise NotImplementedError()
    
    @abstractmethod
    def to_full_short(self)->tuple:
        raise NotImplementedError()

    @abstractmethod
    def to_full_obs(self)->np.ndarray:
        raise NotImplementedError()
    
    @staticmethod
    @abstractmethod
    def new_state()->'State':
        raise NotImplementedError()
    
    
