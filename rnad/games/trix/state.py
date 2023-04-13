from rnad.games.state import State
import random
import numpy as np
from enum import IntEnum
import copy


N_ACTIONS = 52
N_PLAYERS = 4
N_TEAMS = 2

PLAYER_0_HAND = 0
PLAYER_1_HAND = 1
PLAYER_2_HAND = 2
PLAYER_3_HAND = 3
FLOOR = 4
TEAM_0_TAKES = 5
TEAM_1_TAKES = 6
OBSERVATION_SPACE = (64,56)
class SUITS(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    SPADES = 2
    HEARTS = 3

class VALUES(IntEnum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12

'''
board consists of 7 lists of cards , player 0 hand , player 1 hand , player 2 hand and player 3 hand , floor ,players0-2 trash , players 1-3 trash
while the game consists of 4 players , a player controls  

'''
class TrixState(State):
    def __init__(self,board:list[list[int]],player_turn:int,actions_history:list[tuple[int,int]],constraints:np.ndarray) -> None:
        super().__init__()
        self._board = board
        self._player_turn = player_turn
        self._actions_history = actions_history

        self._contstraints = constraints
        # CACHE
        self._cached_legal_actions_masks : np.ndarray|None = None
        self._cached_score : np.ndarray|None = None
        self._cached_is_terminal: bool|None = None

        self._cached_full_obs : np.ndarray|None = None
        self._cached_player_obs : np.ndarray|None = None

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def observation_space(self) -> tuple:
        return OBSERVATION_SPACE
    
    @property
    def player_turn(self) -> int:
        return self._player_turn % N_TEAMS
    
    @staticmethod
    def new_state()->'TrixState':
        board = TrixState._deal()
        first_player = -1
        for player,hand in enumerate(board):
            for card in hand:
                if card == 0:
                    first_player = player
                    break
        action_history :list[tuple[int,int]]= []
        constrain : np.ndarray = np.zeros((N_PLAYERS,4),dtype=np.int32)
        return TrixState(board,first_player,action_history,constrain)
    def legal_actions_masks(self) -> np.ndarray:
        assert not self.is_terminal()
        if self._cached_legal_actions_masks is not None:
            return self._cached_legal_actions_masks.copy()
        
        player = self._player_turn
        player_hand = self._board[player]
        floor = self._board[FLOOR]

        # TODO check
        player_hand_masks = np.zeros((N_ACTIONS,),dtype=np.int32)
        player_hand_masks[player_hand] = 1
        actions_masks = np.zeros((N_ACTIONS,),dtype=np.int32)
        if len(floor) == 0:
            actions_masks[player_hand_masks==1] = 1
            self._cached_legal_actions_masks = actions_masks
            assert self._cached_legal_actions_masks.max() > 0
            return self._cached_legal_actions_masks.copy()
        
        # if have same suit

        first_card_suit,_ = TrixState._suit_and_value(floor[0])
        same_suit_masks = np.zeros((N_ACTIONS,),dtype=np.int32)
        for card in player_hand:
            player_card_suit , _ = TrixState._suit_and_value(card)
            if first_card_suit == player_card_suit:
                same_suit_masks[card] = 1
        
        if same_suit_masks.max() > 0:
            self._cached_legal_actions_masks = same_suit_masks
        else:
            self._cached_legal_actions_masks = player_hand_masks
        
        assert self._cached_legal_actions_masks.max() > 0
        return self._cached_legal_actions_masks.copy()

    def step(self, action: int) -> 'TrixState':
        assert self.legal_actions_masks()[action] == 1

        player = self._player_turn
        new_board = copy.deepcopy(self._board)
        new_actions_history = copy.deepcopy(self._actions_history)
        player_hand = new_board[player]
        player_hand.remove(action)
        floor = new_board[FLOOR]
        floor.append(action)
        new_actions_history.append((action,player))
        new_constrains = self._contstraints.copy()

        action_suit , action_value = self._suit_and_value(action)
        first_suit , first_value = self._suit_and_value(floor[0])
        if action_suit != first_suit:
            new_constrains[player][first_suit] = 1

        if len(floor) < 4:
            new_player = (player+1) % N_PLAYERS
            return TrixState(new_board,new_player,new_actions_history,new_constrains)
        
        ### SHOWDOWN ###
        first_floor_card = floor[0]
        first_suit , _ = self._suit_and_value(first_floor_card)
        first_player = player - 1 if player > 0 else 3
        highest_value = -1
        highest_player = -1
        for increment,floor_card in enumerate(floor):
            floor_suit , floor_value = self._suit_and_value(floor_card)
            floor_player = (first_player + increment) % N_PLAYERS
            if floor_suit == first_suit and floor_value > highest_value:
                highest_value = floor_value
                highest_player = floor_player

        assert highest_player != -1
        highest_team = highest_player % N_TEAMS
        
        team_takes = TEAM_0_TAKES+highest_team
        for card in floor:
            new_board[team_takes].append(card)
        new_board[FLOOR] = []
        return TrixState(new_board,highest_player,new_actions_history,new_constrains)

    def is_terminal(self) -> bool:
        if self._cached_is_terminal is not None:
            return self._cached_is_terminal
        
        players_cards = self._board[:N_PLAYERS]
        for player_card in players_cards:
            if len(player_card) > 0:
                return False
        return True
    
    def game_result(self) -> np.ndarray:
        if self._cached_score is not None:
            return self._cached_score.copy()
        scores = np.zeros((N_TEAMS,),dtype=np.float32)
        for team in range(N_TEAMS):
            team_takes = self._board[TEAM_0_TAKES+team]
            n_takes = 0
            diamonds_score = 0
            queen_score = 0
            king_score = 0
            takes_score = 0
            for take in team_takes:
                n_takes+=1
                take_suit , take_value = self._suit_and_value(take)
                if take_suit == SUITS.DIAMONDS:
                    diamonds_score +=10
                if take_value == VALUES.QUEEN:
                    queen_score += 25
                if take_suit == SUITS.HEARTS and take_value == VALUES.KING:
                    king_score += 75
            assert (n_takes % 4) == 0
            takes_score = (n_takes // 4) * 15

            team_score = -diamonds_score - queen_score - king_score - takes_score
            scores[team] = team_score
        # normalized_score = (scores + 250) / 25
        # self._cached_score = normalized_score
        # self._cached_score = normalized_score
        if scores[0] > scores[1]:
            self._cached_score = np.array([1,-1],dtype=np.int32)
        else:
            self._cached_score = np.array([-1,1],dtype=np.int32)
        return self._cached_score.copy()

    def to_full_obs(self) -> np.ndarray:
        # 52 X 10 (4players,2takes,4floors) 52 turns X 56 (52 Actions + 4players who took these actions)
        # 10 X 56 + 52 X 56
        if self._cached_full_obs is not None:
            return self._cached_full_obs.copy()
        current_player = self._player_turn
        current_team = self._player_turn % N_TEAMS
        obs = np.zeros((64,56),dtype=np.float32)
        takes_start = N_PLAYERS
        floor_start = takes_start + N_TEAMS
        actions_start = floor_start + N_PLAYERS
        for relative_player_idx in range(N_PLAYERS):
            true_player_idx = (current_player + relative_player_idx ) % N_PLAYERS
            player_cards = self._board[true_player_idx]
            for card in player_cards:
                obs[relative_player_idx,card] = 1
        for relative_player_team in range(N_TEAMS):
            true_player_team = (current_team + relative_player_team) % N_TEAMS
            team_takes = self._board[TEAM_0_TAKES+true_player_team]
            for take in team_takes:
                obs[takes_start+true_player_team,take] = 1

        floor = copy.deepcopy(self._board[FLOOR])
        relative_player_floor = N_PLAYERS
        # Player did not play yet , his floor should always be empty
        while len(floor)>0:
            relative_player_floor -=1
            card = floor.pop()
            true_player_floor = (current_player + relative_player_floor + N_PLAYERS) % N_PLAYERS
            obs[floor_start+relative_player_floor,card] = 1
        
        for idx,(action,player) in enumerate(self._actions_history):
            relative_player = ( player - current_player + N_PLAYERS ) % N_PLAYERS
            obs[actions_start+idx,action] = 1
            obs[actions_start+idx,N_ACTIONS+relative_player] = 1
        # self._cached_full_obs = obs.reshape(OBSERVATION_SPACE)
        self._cached_full_obs = obs.reshape(OBSERVATION_SPACE)
        return self._cached_full_obs.copy()

    def to_player_obs(self) -> np.ndarray:
        return self.to_full_obs()
        current_player = self._player_turn
        current_team = self._player_turn % N_TEAMS
        obs = np.zeros((64,56),dtype=np.float32)
        takes_start = N_PLAYERS
        floor_start = takes_start + N_TEAMS
        actions_start = floor_start + N_PLAYERS
        # current player card
        for card in self._board[current_player]:
            obs[0,card] = 1
        for suit in range(4):
            # check how many card of this suit we have
            suit_count = 0
            for card in self._board[current_player]:
                card_suit,card_value = self._suit_and_value(card)
                if card_suit == suit:
                    suit_count+=1
            for card in self._board[TEAM_0_TAKES]:
                card_suit,card_value = self._suit_and_value(card)
                if card_suit == suit:
                    suit_count+=1
            for card in self._board[TEAM_1_TAKES]:
                card_suit,card_value = self._suit_and_value(card)
                if card_suit == suit:
                    suit_count+=1
            remaining_cards = 13 - suit_count
            if remaining_cards == 0:
                continue

            # check how many player can have the remaining cards
            n_players = 0
            for player in range(N_PLAYERS):
                if player == current_player:
                    continue
                if self._contstraints[player,suit] == 0:
                    n_players+=1

            if n_players == 0:
                continue
            prob = 1/n_players
            for value in range(13):
                card = suit * 13 + value
                if obs[0,card] == 1:
                    continue
                for player in range(N_PLAYERS):
                    if player == current_player:
                        continue
                    relative_player = (player-current_player + N_PLAYERS) % N_PLAYERS
                    obs[relative_player,card] = prob
            
        
        for relative_player_team in range(N_TEAMS):
            true_player_team = (current_team + relative_player_team) % N_TEAMS
            team_takes = self._board[TEAM_0_TAKES+true_player_team]
            for take in team_takes:
                obs[takes_start+true_player_team,take] = 1

        floor = copy.deepcopy(self._board[FLOOR])
        relative_player_floor = N_PLAYERS
        # Player did not play yet , his floor should always be empty
        while len(floor)>0:
            relative_player_floor -=1
            card = floor.pop()
            true_player_floor = (current_player + relative_player_floor + N_PLAYERS) % N_PLAYERS
            obs[floor_start+relative_player_floor,card] = 1
        
        for idx,(action,player) in enumerate(self._actions_history):
            relative_player = ( player - current_player + N_PLAYERS ) % N_PLAYERS
            obs[actions_start+idx,action] = 1
            obs[actions_start+idx,N_ACTIONS+relative_player] = 1

        c:np.ndarray = obs[:10,:52]
        z = c.sum(axis=0)
        v = np.argwhere(z<0.99)
        assert len(v) == 0

        self._cached_player_obs = obs.reshape(OBSERVATION_SPACE)
        return self._cached_player_obs.copy()

    
    def render(self, full: bool) -> None:
        return super().render(full)
    def to_full_short(self) -> tuple:
        return super().to_full_short()
    
    def to_player_short(self) -> tuple:
        return super().to_player_short()
    
    @staticmethod
    def _suit_and_value(action:int)->tuple[int,int]:
        suit = action // 13
        value = action % 13
        return suit,value
    @staticmethod
    def _deal():
        cards = [a for a in range(52)]
        random.shuffle(cards)
        players:list[list[int]] = [[],[],[],[]]
        for player in range(4):
            for _ in range(13):
                players[player].append(cards.pop())
        assert len(cards) == 0
        board:list[list[int]] = [
            players[0], # player 0 hand
            players[1], # player 1 hand
            players[2], # player 2 hand
            players[3], # player 3 hand
            [], # Floor cards
            [], # Team 0 Takes  
            [] # Team 1 Takes
        ]
        return board

    