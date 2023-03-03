import numpy as np
from rnad.games.state import State

N_PLAYERS = 2
N_ROWS = 3
N_COLS = 3
N_ACTIONS = N_ROWS * N_COLS
OBSERVATION_SPACE = (N_PLAYERS+1, N_ROWS, N_COLS)
MAX_TURNS = 20

class PhantomTicTacToeState(State):
    '''
    board is 2*3*3 showing probability of having cell played for 2 players 3 rows , 3 cols
    '''

    def __init__(self, board: np.ndarray, player_board: np.ndarray, opponent_board: np.ndarray, player_turn: int, turn: int) -> None:
        super().__init__()
        self._board = board
        self._player_board = player_board
        self._opponent_board = opponent_board
        self._player_turn = player_turn
        self._turn = turn

        self._cached_legal_actions_masks: None | np.ndarray = None
        self._cached_is_terminal: None | bool = None
        self._cached_game_result: None | np.ndarray = None

        self._cached_full_obs :None|np.ndarray = None
        self._cached_player_obs:None|np.ndarray = None

        self._cached_full_short : None|tuple = None
        self._cached_player_short : None|tuple = None

    @staticmethod
    def new_state()->'PhantomTicTacToeState':
        board = np.zeros((N_PLAYERS,N_ROWS,N_COLS),dtype=np.int32)
        player_board = np.zeros((N_PLAYERS,N_ROWS,N_COLS),dtype=np.float32)
        opponent_board = np.zeros((N_PLAYERS,N_ROWS,N_COLS),dtype=np.float32)
        turn = 0
        player_turn = 0
        return PhantomTicTacToeState(
            board=board,
            player_board=player_board,
            opponent_board=opponent_board,
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
        masks: np.ndarray = (self._board[self._player_turn] != 1)
        masks = masks.copy().flatten()
        self._cached_legal_actions_masks = masks
        return self._cached_legal_actions_masks

    def step(self, action: int) -> 'PhantomTicTacToeState':
        if self.is_terminal():
            raise ValueError("Game is over")
        legal_actions_masks = self.legal_actions_masks()
        if legal_actions_masks[action] != 1:
            raise ValueError("Action is not legal")
        
        new_board = self._board.copy()
        new_player_board = self._player_board.copy()
        new_opponent_board = self._opponent_board.copy()

        row = action // N_COLS
        col = action % N_COLS 
        player = self._player_turn
        opponent = 1-player
        next_player = opponent

        if new_board[opponent,row,col] == 1 : # if opponent has mark on this cell
            # remove opponent cell mark from global_board
            new_board[opponent,row,col] = 0
            # set opponent cell in his board and on his player board to 0
            # channel 0 refers to whom the board belongs and 1 refers to their opponent
            new_opponent_board[0,row,col] = 0
            new_player_board[0,row,col] = 0

            
            new_opponent_board[1,row,col] = 0
            prob = 1-new_player_board[1,row,col]
            new_player_board[1,row,col] = 0
            possible_cells = new_player_board[1] > 0
            sum_ = np.sum(possible_cells)
            prob = prob/sum_ if sum_ > 0 else 0
            new_player_board[1,possible_cells] -=prob

            next_player = player
        else:
            # set player cell mark on global board to 1
            new_board[player,row,col] = 1

            # set player cell mark on his board to 1
            new_player_board[0,row,col] = 1

            # scatter the probability of current cell into all other possible cells
            prob = new_player_board[1,row,col]
            possible_cells = new_player_board[0] != 1.
            leng = len(possible_cells)+1e-8
            prob = prob / leng
            new_player_board[1][possible_cells] +=prob

            # empty current cell probabiliy
            new_player_board[1,row,col] = 0


            # increase the probabiliy of player cell on their opponent board
            # TODO: debug
            possible_cells = new_opponent_board[0] !=1.0
            prob  = 1./np.sum(possible_cells)
            new_opponent_board[1][possible_cells] += prob
            
            # switch players and their boards
            next_player = opponent
            new_player_board , new_opponent_board = new_opponent_board,new_player_board
        
        return PhantomTicTacToeState(
            board=new_board,
            player_board=new_player_board,
            opponent_board=new_opponent_board,
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
            return self._cached_game_result
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
        return self._cached_game_result
        

    def to_full_obs(self) -> np.ndarray:
        if self._cached_full_obs is not None:
            return self._cached_full_obs
        player = self._player_turn
        if player != 0:
            current_player_full_obs = self._board[::-1].copy()
        else:
            current_player_full_obs = self._board.copy()
        # current_player_full_obs = self._board[::-1].copy()
        percent = (MAX_TURNS-self._turn-1)*9 // MAX_TURNS
        turn_obs = np.zeros((N_ROWS*N_COLS),dtype=np.int32)
        turn_obs[percent] = 1
        turn_obs = turn_obs.reshape((1,N_ROWS,N_COLS))
        full_obs = np.concatenate([current_player_full_obs,turn_obs],axis=0)
        self._cached_full_obs = full_obs
        return self._cached_full_obs
    
    def to_player_obs(self) -> np.ndarray:
        if self._cached_player_obs is not None:
            return self._cached_player_obs
        percent = (MAX_TURNS-self._turn-1)*9 // MAX_TURNS
        turn_obs = np.zeros((N_ROWS*N_COLS),dtype=np.int32)
        turn_obs[percent] = 1
        turn_obs = turn_obs.reshape((1,N_ROWS,N_COLS))
        self._cached_player_obs = np.concatenate([self._player_board,turn_obs],axis=0)
        return self._cached_player_obs
    
    def to_player_short(self) -> tuple:
        if self._cached_player_short is not None:
            return self._cached_player_short
        board_cells = self._player_board.flatten()
        self._cached_player_short = (self._player_turn,self._player_turn,*board_cells)
        return self._cached_player_short
    
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
        

