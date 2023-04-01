import numpy as np
from .state import KuhnPokerState
from rnad.games.game import Game

class KuhnPokerGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self.state = KuhnPokerState.new_state()
    
    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    
    @property
    def n_actions(self) -> int:
        return self.state.n_actions
    
    @property
    def observation_space(self) -> tuple:
        return self.state.observation_space
    
    @property
    def player_turn(self) -> int:
        return self.state.player_turn
    
    def reset(self) -> KuhnPokerState:
        self.state = KuhnPokerState.new_state()
        return self.state
    
    def step(self, action: int) -> KuhnPokerState:
        self.state = self.state.step(action)
        return self.state

    def game_result(self) -> np.ndarray:
        return self.state.game_result()

    def render(self, full: bool) -> None:
        self.state.render(full)
