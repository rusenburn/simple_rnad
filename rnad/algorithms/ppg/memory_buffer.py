import numpy as np
from rnad.algorithms.ppo.memory_buffer import MemoryBuffer



class PolicyMemoryBuffer():
    def __init__(self,
                observation_shape:tuple,
                n_actions:int,
                n_workers:int,
                worker_steps:int,
                n_game_players:int) -> None:
        self._observation_shape = observation_shape
        self._n_actions = n_actions
        self._n_workers = n_workers
        self._worker_steps = worker_steps
        self._n_game_players = n_game_players
        self.reset()
    
    
    def reset(self):
        self.current_step = 0

        self.partial_observations = np.zeros(
            (self._worker_steps, self._n_workers, *self._observation_shape), dtype=np.float32)
        
        self.full_observations = np.zeros(
            (self._worker_steps, self._n_workers, *self._observation_shape), dtype=np.float32)
        
        self.actions = np.zeros(
            (self._worker_steps, self._n_workers), dtype=np.float32)
        
        self.legal_actions_masks = np.zeros((self._worker_steps,self._n_workers,self._n_actions),dtype=np.int32)

        self.rewards = np.zeros(
            (self._worker_steps, self._n_workers,self._n_game_players), dtype=np.float32)
        
        self.log_probs = np.zeros(
            (self._worker_steps, self._n_workers), dtype=np.float32)
        
        self.terminals = np.zeros(
            (self._worker_steps, self._n_workers), dtype=np.bool8)
        
        self.players = np.zeros(
            (self._worker_steps, self._n_workers), dtype=np.int32)
    
    def save(self,
             step_partial_observations:np.ndarray,step_full_observations:np.ndarray,
             step_actions:np.ndarray,step_legal_actions_masks:np.ndarray,
             step_log_probs:np.ndarray,step_rewards:np.ndarray,step_terminals:np.ndarray,
             step_players:np.ndarray):
        self.partial_observations[self.current_step] = np.array(
            step_partial_observations)
        self.full_observations[self.current_step] = np.array(
            step_full_observations)
        self.actions[self.current_step] = step_actions
        self.legal_actions_masks[self.current_step] = step_legal_actions_masks
        self.log_probs[self.current_step] = step_log_probs
        self.rewards[self.current_step] = step_rewards
        self.terminals[self.current_step] = step_terminals
        self.players[self.current_step] = step_players
        self.current_step += 1

    def sample(self):
        return self.partial_observations.swapaxes(0, 1).copy(), self.full_observations.swapaxes(0, 1).copy(), self.actions.swapaxes(0, 1).copy(), self.legal_actions_masks.swapaxes(0,1).copy(),self.log_probs.swapaxes(0, 1).copy(), self.rewards.swapaxes(0, 1).copy(), self.terminals.swapaxes(0, 1).copy(),self.players.swapaxes(0,1).copy()

class AuxMemoryBuffer():
    def __init__(self,observation_shape:tuple,n_actions:int,n_workers:int,worker_steps:int,n_policy_iterations:int) -> None:
        self.obsevation_shape = observation_shape
        self.n_workers = n_workers
        self.n_actions = n_actions
        self.worker_steps = worker_steps
        self.n_policy_iterations = n_policy_iterations
        self.n_examples = self.n_workers * self.worker_steps * self.n_policy_iterations
        self.reset_aux_data()
    
    def reset_aux_data(self):
        self.aux_partial_observations = np.zeros((self.n_examples,*self.obsevation_shape),dtype=np.float32)
        self.aux_legal_actions_masks = np.zeros((self.n_examples,self.n_actions))
        self.aux_full_observations = np.zeros((self.n_examples,*self.obsevation_shape),dtype=np.float32)
        self.aux_returns = np.zeros((self.n_examples,),dtype=np.float32)
        self.players = np.zeros((self.n_examples,),dtype=np.int32)
        self.aux_index = 0
    
    def append_aux_data(self,partial_observations:np.ndarray,legal_actions_masks:np.ndarray,full_observations:np.ndarray,returns:np.ndarray,players:np.ndarray):
        if self.aux_index >= self.n_examples:
            raise ValueError("Cannot append data to a full sized buffer")
        
        n_iterations_example = self.n_workers * self.worker_steps
        begin = self.aux_index * n_iterations_example
        end = begin + n_iterations_example
        self.aux_partial_observations[begin:end] = partial_observations.copy()
        self.aux_legal_actions_masks[begin:end] = legal_actions_masks.copy()
        self.aux_full_observations[begin:end] = full_observations.copy()
        self.aux_returns[begin:end] = returns.copy()
        self.players[begin:end] = players.copy()
        self.aux_index+=1
    
    def sample(self):
        '''
        return partial observations , legal action masks , full observations , returns
        '''
        return self.aux_partial_observations, self.aux_legal_actions_masks,self.aux_full_observations,self.aux_returns,self.players
        
class PPGMemoryBuffer():
    def __init__(self,observation_shape:tuple,n_actions ,n_workers:int,worker_steps:int,n_policy_iterations) -> None:
        self.memory_buffer = MemoryBuffer(
            observation_shape= observation_shape,
            n_actions=n_actions,
            n_workers=n_workers,
            worker_steps=worker_steps)
        
        self.obsevartion_shape = observation_shape
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.n_policy_iterations = n_policy_iterations
        self.n_examples = self.n_workers * self.worker_steps * self.n_policy_iterations
        self.reset_aux_data()
    

    def reset_policy_data(self):
        self.memory_buffer.reset()
    
    def save_policy_data(self,step_partial_observations:np.ndarray,step_full_observations:np.ndarray,
                         step_actions:np.ndarray,step_legal_action_masks:np.ndarray,
                         step_log_probs:np.ndarray,step_values:np.ndarray,
                         step_rewards:np.ndarray,step_terminals:np.ndarray,
                         step_players:np.ndarray):
        self.memory_buffer.save(
            step_partial_observations=step_partial_observations,
            step_full_observations=step_full_observations,
            step_actions=step_actions,
            step_legal_actions_masks=step_legal_action_masks,
            step_log_probs=step_log_probs,
            step_values=step_values,
            step_rewards=step_rewards,
            step_terminals=step_terminals,
            step_players=step_players)
    
    def sample_policy_data(self):
        return self.memory_buffer.sample()

    def reset_aux_data(self):
        self.aux_partial_observations = np.zeros((self.n_examples,*self.aux_partial_observations),dtype=np.float32)
        self.aux_full_observations = np.zeros((self.n_examples,*self.aux_partial_observations),dtype=np.float32)
        self.aux_returns = np.zeros((self.n_examples,),dtype=np.int32)
        self.aux_index = 0
    
    def append_aux_data(self,partial_observations:np.ndarray,returns:np.ndarray):
        if self.aux_index >= self.n_examples:
            raise ValueError("Cannot append data to a full sized buffer")
        
        n_iterations_example = self.n_workers * self.worker_steps
        begin = self.aux_index * n_iterations_example
        end = begin + n_iterations_example
        self.aux_partial_observations[begin:end] = partial_observations.copy()
        self.aux_returns[begin:end] = returns.copy()
        self.aux_index+=1

    def sample_aux_data(self):
        return self.aux_full_observations,self.aux_returns
    