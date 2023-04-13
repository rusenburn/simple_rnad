from rnad.games.game import Game
from .state import OthelloState
import numpy as np

class OthelloGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.state = OthelloState.new_state()
    
    def step(self, action: int) -> OthelloState:
        self.state = self.state.step(action)
        return self.state

    @property
    def n_actions(self) -> int:
        return self.state.n_actions
    
    @property
    def observation_space(self) -> tuple:
        return self.state.observation_space
    
    @property
    def player_turn(self) -> int:
        return self.state.player_turn
    
    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    
    def game_result(self) -> np.ndarray:
        return self.state.game_result()
    
    def render(self, full: bool) -> None:
        return self.state.render(full)
    
    def reset(self) -> OthelloState:
        self.state = OthelloState.new_state()
        return self.state
    