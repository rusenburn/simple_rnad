import numpy as np
from rnad.games.game import Game
from .state import LeducPokerState

class LeducPokerGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.state = self.reset()
    

    def step(self, action: int) -> LeducPokerState:
        self.state = self.state.step(action)
        return self.state
    def reset(self) -> 'LeducPokerState':
        self.state = LeducPokerState.new_state()
        return self.state

    @property
    def observation_space(self) -> tuple:
        return self.state.observation_space

    @property    
    def n_actions(self) -> int:
        return self.state.n_actions

    @property
    def player_turn(self) -> int:
        return self.state.player_turn
    def game_result(self) -> np.ndarray:
        return self.state.game_result()

    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    def render(self, full: bool) -> None:
        self.state.render(full)
