from rnad.games.game import Game
import numpy as np
from .state import GoofSpielState

class GoofSpielGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self._state = GoofSpielState.new_state()
    
    def step(self, action: int) -> GoofSpielState:
        self._state = self._state.step(action)
        return self._state


    @property
    def n_actions(self) -> int:
        return self._state.n_actions
    
    @property
    def observation_space(self) -> tuple:
        return self._state.observation_space
    
    @property
    def player_turn(self) -> int:
        return self._state.player_turn
        
    def reset(self) -> GoofSpielState:
        self._state = GoofSpielState.new_state()
        return self._state


    def game_result(self) -> np.ndarray:
        return self._state.game_result()
    
    def is_terminal(self) -> bool:
        return self._state.is_terminal()

    def render(self, full: bool):
        self._state.render(full)