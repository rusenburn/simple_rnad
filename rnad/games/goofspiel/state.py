import numpy as np
from rnad.games.state import State

PLAYER_0_HAND = 0
PLAYER_0_TAKES = 1
PLAYER_1_HAND = 2
PLAYER_1_TAKES = 3
DECK = 4
FLOOR = 5
LAST_TURN = 2*13 - 1
N_PLAYERS = 2


CARDS = {
    0:"A",
    1:"2",
    2:"3",
    3:"4",
    4:"5",
    5:"6",
    6:"7",
    7:"8",
    8:"9",
    9:"X",
    10:"J",
    11:"Q",
    12:"K",
}

class GoofSpielState(State):
    def __init__(self, board: np.ndarray, player_0_action: int | None, turn: int) -> None:
        super().__init__()
        self._board = board
        self._player_0_action = player_0_action
        self._turn = turn


        # cached 
        self._cached_legal_actions_masks : np.ndarray |None = None
        self._cached_game_result : np.ndarray | None = None
        self._cached_full_obs : np.ndarray|None = None
        # self._board = np.array([
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # player_0 probability of having that card
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # player_0 takes
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # player_1 probability of having that card
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # player_1 takes
        #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # deck_remaining_cards
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # floor cards
        # ])

    @property
    def n_actions(self) -> int:
        return 13

    @property
    def observation_space(self) -> tuple:
        return (6, 13)

    @property
    def player_turn(self) -> int:
        return self._turn % 2

    def is_terminal(self) -> bool:
        return self._turn > LAST_TURN

    def step(self, action: int) -> 'GoofSpielState':
        assert self.legal_actions_masks()[action] == 1
        new_board = self._board.copy()
        if self.player_turn == 0:
            player_0_action = action
            turn = self._turn+1
            return GoofSpielState(new_board, player_0_action, turn)
        assert self._player_0_action is not None
        draw = False
        if self._player_0_action > action:
            player_0_takes = np.argwhere(self._board[FLOOR] == 1)
            new_board[PLAYER_0_TAKES][player_0_takes] = 1
        elif self._player_0_action < action:
            player_1_takes = np.argwhere(self._board[FLOOR] == 1)
            new_board[PLAYER_1_TAKES][player_1_takes] = 1
        else:  # draw
            draw = True

        new_board[PLAYER_0_HAND][self._player_0_action] = 0
        new_board[PLAYER_1_HAND][action] = 0
        if not draw:
            new_board[FLOOR] *= 0 # set it to 0
        
        if self._turn < LAST_TURN:
            next_floor_cards = np.argwhere(new_board[DECK] == 1)
            choice = np.random.choice(len(next_floor_cards))
            next_floor_card = next_floor_cards[choice]
            new_board[DECK][next_floor_card] = 0
            new_board[FLOOR][next_floor_card] = 1
        
        return GoofSpielState(new_board,None,self._turn+1)


    def legal_actions_masks(self) -> np.ndarray:
        if self._cached_legal_actions_masks is not None:
            return self._cached_legal_actions_masks.copy()
        
        current_player = self.player_turn
        if current_player == 0:
            player_hand = PLAYER_0_HAND
        else:
            player_hand = PLAYER_1_HAND
        
        legal_actions :np.ndarray= (self._board[player_hand] == 1)
        self._cached_legal_actions_masks = legal_actions.copy()
        return self._cached_legal_actions_masks.copy()


    def game_result(self) -> np.ndarray:
        assert self.is_terminal()

        player_0_takes = np.argwhere(self._board[PLAYER_0_TAKES] == 1)
        player_1_takes = np.argwhere(self._board[PLAYER_1_TAKES] == 1)
        # increase all by 1 , because Ace is represented by 0 , but it is actually 1 , and so on
        player_0_takes += 1
        player_1_takes += 1

        player_0_total = player_0_takes.sum()
        player_1_total = player_1_takes.sum()
        
        scores = np.zeros((N_PLAYERS,),dtype=np.int32)
        if player_0_total > player_1_total:
            scores[0] = 1
            scores[1] = -1
        elif player_0_total < player_1_total:
            scores[0] = -1
            scores[1] = 1
        else:
            # draw
            ...
        self._cached_game_result = scores
        return self._cached_game_result.copy()

    def to_full_obs(self) -> np.ndarray:
        if self._cached_full_obs is not None:
            return self._cached_full_obs.copy()
        if self.player_turn == 0:
            player_hand = self._board[PLAYER_0_HAND]
            player_takes = self._board[PLAYER_0_TAKES]
            opponent_hand = self._board[PLAYER_1_HAND]
            opponent_takes = self._board[PLAYER_1_TAKES]
        else:
            player_hand = self._board[PLAYER_1_HAND]
            player_takes = self._board[PLAYER_1_TAKES]
            opponent_hand = self._board[PLAYER_0_HAND]
            opponent_takes = self._board[PLAYER_0_TAKES]
        
        deck = self._board[DECK]
        floor = self._board[FLOOR]

        obs = np.stack([player_hand,player_takes,opponent_hand,opponent_takes,deck,floor],axis=0)
        self._cached_full_obs = obs.copy()
        return self._cached_full_obs.copy()

    def to_player_obs(self) -> np.ndarray:
        return self.to_full_obs()

    def to_player_short(self) -> tuple:
        return super().to_player_short()

    def to_full_short(self) -> tuple:
        return super().to_full_short()
    
    @staticmethod
    def new_state():
        new_board = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # player_0 probability of having that card
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # player_0 takes
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # player_1 probability of having that card
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # player_1 takes of having that card
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # deck_remaining_cards
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # floor cards
        ])

        next_floor_cards = np.argwhere(new_board[DECK] == 1)
        choice = np.random.choice(len(next_floor_cards))
        next_floor_card = next_floor_cards[choice]
        new_board[DECK][next_floor_card] = 0
        new_board[FLOOR][next_floor_card] = 1
        turn = 0
        return GoofSpielState(new_board,None,turn)

    def render(self, full: bool) -> None:
        result : list[str] = []
        result.append("\n")
        result.append("###########################################################\n")
        result.append("***************************************\n")
        player_0_score = (np.argwhere(self._board[PLAYER_0_TAKES] == 1) + 1).sum()
        player_1_score = (np.argwhere(self._board[PLAYER_1_TAKES] == 1) + 1).sum()
        for i in range(13):
            if self._board[PLAYER_0_HAND][i] == 1:
                result.append(f" {CARDS[i]} ")
            else:
                result.append(f" . ")
        result.append("\n")
        result.append("\n")
        result.append("***************************************\n")
        result.append("\n")
        for i in range(13):
            if self._board[FLOOR][i] == 1:
                result.append(f" {CARDS[i]} ")
            else:
                result.append(f" . ")
        result.append("\n")
        result.append("\n")
        result.append("***************************************\n")
        for i in range(13):
            if self._board[PLAYER_1_HAND][i] == 1:
                result.append(f" {CARDS[i]} ")
            else:
                result.append(f" . ")
        result.append("\n")
        result.append("\n")
        result.append(f"Scores : {player_0_score}\t {player_1_score}\n")
        result.append(f"Player {self.player_turn} turn\n")
        result.append("###########################################################\n")
        result.append("\n")
        print("".join(result))
        
