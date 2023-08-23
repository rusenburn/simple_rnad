import numpy as np

class ReservoirBuffer():
    def __init__(self,max_len:int,observation_shape:tuple,n_actions:int) -> None:
        self.max_len = max_len
        self.observation_shape = observation_shape
        self._i = 0
        self._current_size = 0

        self._obs = np.zeros((max_len,*observation_shape),dtype=np.float32)
        self._actions = np.zeros((max_len,),dtype=np.int32)
        self._legal_actions = np.zeros((max_len,n_actions),dtype=np.int32)
        

    
    def save(self,obs:np.ndarray,action:int,legal_actions:np.ndarray):
        if self._i < self.max_len:
            self._save(obs,action,legal_actions,self._i)
        else:
            index = np.random.randint(0,self._i+1)
            if index < self.max_len:
                self._save(obs,action,legal_actions,index)
        self._i+=1
    

    def _save(self,obs:np.ndarray,action:int,legal_actions:np.ndarray,index:int):
        self._obs[index] = obs
        self._actions[index] = action
        self._legal_actions[index] = legal_actions
        
        self._current_size+=1
        if self._current_size > self.max_len:
            self._current_size = self.max_len
    
    def sample(self,sample_size:int)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        if sample_size > self._current_size:
            sample_size = self._current_size
        
        indices = np.random.choice(self._current_size,size=sample_size,replace=False)
        return (self._obs[indices].copy() , self._actions[indices].copy(),self._legal_actions[indices].copy())
