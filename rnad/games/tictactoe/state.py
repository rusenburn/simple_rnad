import numpy as np
from rnad.games.state import State

N_PLAYERS = 2
N_ROWS = 3
N_COLS = 3
N_ACTIONS = N_ROWS * N_COLS
OBSERVATION_SPACE = (N_PLAYERS, N_ROWS, N_COLS)
MAX_TURNS = 8

class TicTacToeState(State):
    '''
    board is 2*3*3 showing probability of having cell played for 2 players 3 rows , 3 cols
    '''

    def __init__(self, board: np.ndarray, player_turn: int, turn: int) -> None:
        super().__init__()
        self._board = board
        self._player_turn = player_turn
        self._turn = turn
        self._cached_legal_actions_masks: None | np.ndarray = None
        self._cached_is_terminal: None | bool = None
        self._cached_game_result: None | np.ndarray = None
        self._cached_full_obs :None|np.ndarray = None
        self._cached_full_short : None|tuple = None


    @staticmethod
    def new_state()->'TicTacToeState':
        board = np.zeros((N_PLAYERS,N_ROWS,N_COLS),dtype=np.int32)
        turn = 0
        player_turn = 0
        return TicTacToeState(
            board=board,
            turn=turn,
            player_turn=player_turn)
    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def observation_space(self) -> tuple:
        return OBSERVATION_SPACE

    @property
    def player_turn(self) -> int:
        return self._player_turn

    def legal_actions_masks(self) -> np.ndarray:
        if self._cached_legal_actions_masks is not None:
            return self._cached_legal_actions_masks.copy()

        player = self.player_turn
        opponent = 1 - self.player_turn
        masks = np.logical_and(self._board[player] == 0 , self._board[opponent]==0)
        masks = masks.copy().flatten()
        self._cached_legal_actions_masks = masks
        return self._cached_legal_actions_masks.copy()

    def step(self, action: int) -> 'TicTacToeState':
        if self.is_terminal():
            raise ValueError("Game is over")
        legal_actions_masks = self.legal_actions_masks()
        if legal_actions_masks[action] != 1:
            raise ValueError("Action is not legal")
        
        new_board = self._board.copy()

        row = action // N_COLS
        col = action % N_COLS 
        player = self._player_turn
        opponent = 1-player
        next_player = opponent

        # set player cell mark on board to 1
        new_board[player,row,col] = 1
        
        # switch players and their boards
        next_player = opponent
        
        return TicTacToeState(
            board=new_board,
            player_turn=next_player,
            turn= self._turn+1)
    
    def is_terminal(self) -> bool:
        if self._cached_is_terminal is not None:
            return self._cached_is_terminal
        opponent = 1-self._player_turn
        if self._is_win(opponent):
            self._cached_is_terminal = True
        elif self._is_draw():
            self._cached_is_terminal = True
        else:
            self._cached_is_terminal = False
        return self._cached_is_terminal
    
    def game_result(self) -> np.ndarray:
        ''' returns a numpy array for the scores of all players '''
        assert self.is_terminal()
        if self._cached_game_result is not None:
            return self._cached_game_result.copy()
        opponent = 1-self._player_turn
        game_result = np.zeros((N_PLAYERS,),dtype=np.int32)
        if self._is_win(opponent):
            game_result[opponent] = 1
            game_result[self._player_turn] = -1
        elif self._is_draw():
            ...
        else:
            raise AssertionError("Unreachable code")
        self._cached_game_result = game_result
        return self._cached_game_result.copy()
        

    def to_full_obs(self) -> np.ndarray:
        if self._cached_full_obs is not None:
            return self._cached_full_obs.copy()
        player = self._player_turn
        if player != 0:
            current_player_full_obs = self._board[::-1].copy()
        else:
            current_player_full_obs = self._board.copy()
        full_obs = np.concatenate([current_player_full_obs],axis=0)
        self._cached_full_obs = full_obs
        return self._cached_full_obs.copy()
    
    def to_player_obs(self) -> np.ndarray:
        return self.to_full_obs()
    
    def to_player_short(self) -> tuple:
        return self.to_full_short()
    
    def to_full_short(self) -> tuple:
        if self._cached_full_short is not None:
            return self._cached_full_short
        player_0_cells = np.argwhere(self._board[0] == 1)
        player_1_cells = np.argwhere(self._board[1] == 1)
        board_short = np.zeros((N_ROWS,N_COLS),dtype=np.int32)
        board_short[player_0_cells] = 1
        board_short[player_1_cells] = -1
        board_short = board_short.flatten()
        self._cached_full_short = (self._player_turn,self._turn,*board_short)
        return self._cached_full_short

    def render(self, full: bool) -> None:
        self._render_full()
        

    def _render_full(self)->None:
        if self._player_turn == 0:
            player_rep = "x"
        else:
            player_rep = "o"
        result :list[str]= []
        result.append("****************************\n")
        result.append(f"*** Player {player_rep} has to move ***\n")
        result.append("****************************\n")
        result.append("\n")
        for row in range(N_ROWS):
            for col in range(N_COLS):
                if self._board[0][row][col] == 1:
                    result.append(" x ")
                elif self._board[1][row][col] == 1:
                    result.append(" o ")
                else:
                    result.append(" . ")
                if col == N_COLS-1:
                    result.append("\n")
        result.append("\n")
        print("".join(result))

    def _is_draw(self)->bool:
        # Note: check if we have a winner first
        return self._turn == MAX_TURNS
    def _is_win(self,player:int)->bool:
        if (self._board[player,0,0] != 0 and self._board[player,0,0] == self._board[player,0,1] == self._board[player,0,2]) or (
            self._board[player,1,0] != 0 and self._board[player,1,0] == self._board[player,1,1] == self._board[player,1,2]) or (
            self._board[player,2,0] != 0 and self._board[player,2,0] == self._board[player,2,1] == self._board[player,2,2]
            ):
            return True
        if (self._board[player,0,0] != 0 and self._board[player,0,0] == self._board[player,1,0] == self._board[player,2,0]) or(
            self._board[player,0,1] != 0 and self._board[player,0,1] == self._board[player,1,1] == self._board[player,2,1]) or (
            self._board[player,0,2] != 0 and self._board[player,0,2] == self._board[player,1,2] == self._board[player,2,2]
            ):
            return True
        if (self._board[player,0,0] != 0 and self._board[player,0,0] == self._board[player,1,1] == self._board[player,2,2]):
            return True
        if (self._board[player,0,2] !=0 and self._board[player,0,2] == self._board[player,1,1] == self._board[player,2,0]):
            return True
        return False
        

