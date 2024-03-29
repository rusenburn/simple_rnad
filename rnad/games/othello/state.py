from typing import List, Tuple
from rnad.games.state import State
import numpy as np

class OthelloState(State):
    directions = [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]
    def __init__(self,observation:np.ndarray,_n_consecutive_skip:int) -> None:
        super().__init__()
        self._observation = observation
        self._n_consecutive_skip = _n_consecutive_skip

        # cached
        self._legal_actions :np.ndarray|None= None
        self._cached_game_result : np.ndarray|None = None
        self._cached_obs : np.ndarray|None = None
    
    @property
    def n_actions(self) -> int:
        return 65
    
    @property
    def observation_space(self) -> tuple:
        return (2,8,8)
        # return self.shape
    
    @staticmethod
    def new_state():
        game_rows = 8
        game_cols = 8
        players =2
        obs = np.zeros((players+1,game_rows,game_cols),dtype=np.int32)
        obs[1,3,3]=1
        obs[1,4,4]=1
        obs[0,3,4]=1
        obs[0,4,3]=1
        n_consecutive_skips = 0
        return OthelloState(obs,n_consecutive_skips)
    def legal_actions_masks(self) -> np.ndarray:
        if self._legal_actions is not None:
            return self._legal_actions.copy()
        player :int = self._observation[2][0][0]
        obs:np.ndarray = self._observation[:2].copy()
        if player == 1:
            obs = obs[::-1]
        obs = obs[0]-obs[1]
        n_rows = 8
        n_cols = 8
        moves = np.zeros((n_rows * n_cols + 1),dtype=np.int32)
        for row_id in range(n_rows):
            for col_id in range(n_cols):
                if self._is_valid_move(obs,row_id,col_id):
                    moves[row_id*n_cols+col_id]=1
        if moves.sum() == 0:
            moves[-1]=1
        self._legal_actions = moves
        return moves.copy()

    def is_terminal(self) -> bool:
        return self._n_consecutive_skip == 2 or self._observation[:2].sum() == 64
    
    def game_result(self) -> np.ndarray:
        assert self.is_terminal()
        wdl = np.zeros((3,),dtype=np.int32)
        player = self._observation[2][0][0]
        other = 1-player
        player_score:int = np.sum(self._observation[player])
        other_score :int= np.sum(self._observation[other])
        # if player_score > other_score:
        #     wdl[0]+=1
        # elif player_score == other_score:
        #     wdl[1]+=1
        # else:
        #     wdl[2]+=1
        score = np.zeros((2,),dtype=np.int32)
        if player_score > other_score:
            score =  np.array([1,-1],dtype=np.int32)
        elif player_score == other_score:
            score =  np.array([0,0],dtype=np.int32)
        else:
            score = np.array([-1,1],dtype=np.int32)
        if player != 0:
            score = score[::-1]
        self._cached_game_result = score.copy()
        return self._cached_game_result.copy()
        #     # wdl[2]+=1
        # return wdl.copy()
        
    def step(self, action: int) -> 'OthelloState':
        legal_actions = self.legal_actions_masks()
        if legal_actions[action]==0:
            raise ValueError(f'action {action} is not allowed')
        player = self._observation[2][0][0]
        other = 1-player
        new_obs = self._observation.copy()
        new_obs[2] = other
        if action == 64: # skip action [0,63] are normal actions but 64 is skip action
            return OthelloState(new_obs,self._n_consecutive_skip+1)
        skips = 0
        row_id = action // 8
        col_id = action % 8
        obs:np.ndarray = self._observation[:2].copy()
        if player == 1:
            obs = obs[::-1]
        obs = obs[0]-obs[1]
        vs = self._is_valid_move(obs,row_id,col_id)
        assert len(vs) > 0
        for x,y in vs:
            new_obs[player][x,y] = 1
            new_obs[other][x,y] = 0

        new_obs[player][row_id,col_id]=1
        return OthelloState(new_obs,skips)


    def render(self,full:bool) -> None:
        obs = self._observation[0]-self._observation[1]
        legal_actions = self.legal_actions_masks()[:-1].reshape((8,8))
        print()
        print(  '   0 1 2 3 4 5 6 7')
        for x in range(8):
            print(f"{x*8:2d}", end=' ')
            for y in range(8):
                v :str = "."
                if obs[x][y] == 1:
                    v="X"
                elif obs[x][y] == -1:
                    v="O"
                if legal_actions[x][y]==1:
                    v=","
                print(v, end=' ')
            print("\n",end='')
        player = self._observation[2][0][0]
        v = "X"
        if player == 1:
            v= "O"
        print(f"\n# Player {v} Turn #")

    @property
    def player_turn(self) -> int:
        return self._observation[2,0,0]
    def to_player_short(self) -> tuple:
        player =self._observation[2,0,0]
        obs:np.ndarray = self._observation[0]-self._observation[1]
        r = (player,self._n_consecutive_skip,*obs.flatten())
        return r
    
    def to_player_obs(self) -> np.ndarray:
        if self._cached_obs is not None:
            return self._cached_obs.copy()
        player = self._observation[2][0][0]
        # take observation without last row ( player row )
        obs = self._observation[:-1].copy()
        if player == 0:
            self._cached_obs = obs.copy()
        else:
            # reverse it
            self._cached_obs = obs[::-1].copy()
        
        self._cached_obs = self._cached_obs
        return self._cached_obs.copy()
    
    def to_full_obs(self) -> np.ndarray:
        return self.to_player_obs()

    def get_symmetries(self, probs: np.ndarray) -> List[tuple['State', np.ndarray]]:
        res : List[Tuple['State',np.ndarray]] = []
        ar = self._observation[0]
        ar_2 = self._observation[1]
        ar_3 = self._observation[2]
        skip_prob = probs[-1]

        # reshape probs to match observation and remove last element which is skip_probability
        reshaped_probs = probs[:-1].reshape((8,8))
        for i in range(1,4):
            # rotate observation and reshaped_probs by i
            sym = np.array(
                [np.rot90(ar,k=i,axes=(1,0)),
                np.rot90(ar_2,k=i,axes=(1,0)),
                ar_3])
            sym_state = OthelloState(sym,self._n_consecutive_skip)
            sym_probs = np.zeros((65,),dtype=probs.dtype)
            sym_probs[:-1] = np.rot90(reshaped_probs,k=i,axes=(1,0)).flatten()
            # set skip probability value
            sym_probs[-1] = skip_prob
            res.append((sym_state,sym_probs))
        return res
        
    @property
    def shape(self):
        return (2,8,8)

    def to_full_short(self) -> tuple:
        return super().to_full_short()
    
    def _is_valid_move(self,obs:np.ndarray,row:int,col:int)->List[List[int]]:
        if not self._is_on_board(row,col) or obs[row][col]!=0:
            return []
        
        obs[row,col]=1 # temporary add it to board

        tiles_to_flip = []
        
        for row_dir , col_dir in self.directions:
            x,y =row,col
            x+=row_dir
            y+=col_dir
            if self._is_on_board(x,y) and obs[x,y]==-1:
                x+=row_dir
                y+=col_dir
                if not self._is_on_board(x,y):
                    continue
                
                while obs[x][y] == -1:
                    x+=row_dir
                    y+=col_dir
                    if not self._is_on_board(x,y):
                        break
                if not self._is_on_board(x,y):
                    continue
                if obs[x,y] == 1:
                    while True:
                        x-=row_dir
                        y-=col_dir
                        if x==row and y==col:
                            break
                        tiles_to_flip.append([x,y])
        obs[row,col]=0 # restore empty space
        return tiles_to_flip
    def _is_on_board(self,row:int,col:int)->bool:
        return row < 8 and row>=0 and col<8 and col>=0