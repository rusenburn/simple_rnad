import numpy as np
from rnad.games.game import Game
from .state import PhantomTicTacToeState

class PhantomTicTacToeGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self._state = PhantomTicTacToeState.new_state()
    
    def step(self, action: int) -> PhantomTicTacToeState:
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
        
    def reset(self) -> PhantomTicTacToeState:
        self._state = PhantomTicTacToeState.new_state()
        return self._state


    def game_result(self) -> np.ndarray:
        return self._state.game_result()
    
    def is_terminal(self) -> bool:
        return self._state.is_terminal()

