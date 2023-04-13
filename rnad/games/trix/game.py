from rnad.games.game import Game
import numpy as np
from .state import TrixState

class TrixGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.state = TrixState.new_state()
    
    def reset(self) -> TrixState:
        self.state = TrixState.new_state()
        return self.state
    
    def render(self, full: bool) -> None:
        return self.state.render(full=full)
    
    @property
    def n_actions(self) -> int:
        return self.state.n_actions
    
    @property
    def observation_space(self) -> tuple:
        return self.state.observation_space
    
    @property
    def player_turn(self) -> int:
        return self.state.player_turn
    
    def step(self, action: int) -> TrixState:
        self.state = self.state.step(action)
        return self.state

    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    
    def game_result(self) -> np.ndarray:
        return self.state.game_result()