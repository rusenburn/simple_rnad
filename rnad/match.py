
import numpy as np
from typing import Callable

from rnad.games.game import Game
from rnad.players import PlayerBase


class Match():
    def __init__(self,game_fn:Callable[[],Game],player_1:PlayerBase,player_2:PlayerBase,n_sets=1,render=False) -> None:
        self.player_1 = player_1
        self.player_2 = player_2
        self.game_fn = game_fn
        self.game = game_fn()
        self.n_sets = n_sets
        self.render = render
        self.scores = np.zeros((3,),dtype=np.int32) # W - D - L for player_1
    
    def start(self)->np.ndarray:
        starting_player = 0
        s = self.game.reset()
        for _ in range(self.n_sets):
            scores = self._play_set(starting_player)
            self.scores += scores
            starting_player = 1-starting_player
        return self.scores
    
    def _play_set(self,starting_player:int)->np.ndarray:
        if starting_player == 0:
            players = [self.player_1,self.player_2]
        else:
            players = [self.player_2,self.player_1]
        
        state = self.game.reset()
        done = False
        current_player = 0
        while not done:
            if self.render:
                ...
            current_player = state.player_turn
            player = players[current_player]
            a = player.choose_action(state)
            legal_actions = state.legal_actions_masks()
            if not legal_actions[a]:
                print(f'player {current_player+1} chose wrong action {a}\n')
                continue
            
            state = self.game.step(a)
            done = state.is_terminal()
        
        rewards = self.game.game_result()

        if rewards[0] > rewards[1]:
            scores = np.array([1,0,0])
        elif rewards[1]>rewards[0]:
            scores = np.array([0,0,1])
        else:
            scores = np.array([0,1,0])
        
        if starting_player != 0:
            scores = scores[::-1]
        return scores.copy()

