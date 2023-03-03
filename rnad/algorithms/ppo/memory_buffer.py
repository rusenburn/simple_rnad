import numpy as np


class MemoryBuffer:
    def __init__(self, observation_shape: tuple,n_actions:int, n_workers: int, worker_steps: int) -> None:
        self.observation_shape = observation_shape
        self.n_workers = n_workers
        self.worker_steps = worker_steps
        self.n_actions = n_actions
        self.reset()

    def reset(self):
        self.current_worker = 0
        self.current_step = 0
        self.partial_observations = np.zeros(
            (self.worker_steps, self.n_workers, *self.observation_shape), dtype=np.float32)
        self.full_observations = np.zeros(
            (self.worker_steps, self.n_workers, *self.observation_shape), dtype=np.float32)
        self.actions = np.zeros(
            (self.worker_steps, self.n_workers), dtype=np.float32)
        self.legal_actions_masks = np.zeros((self.worker_steps,self.n_workers,self.n_actions),dtype=np.int32)

        self.rewards = np.zeros(
            (self.worker_steps, self.n_workers,2), dtype=np.float32)
        self.log_probs = np.zeros(
            (self.worker_steps, self.n_workers), dtype=np.float32)
        self.values = np.zeros(
            (self.worker_steps, self.n_workers), dtype=np.float32)
        self.terminals = np.zeros(
            (self.worker_steps, self.n_workers), dtype=np.bool8)
        self.players = np.zeros(
            (self.worker_steps, self.n_workers), dtype=np.int32)

    def save(self, step_partial_observations: np.ndarray, step_full_observations: np.ndarray, step_actions: np.ndarray, step_legal_actions_masks:np.ndarray,step_log_probs: np.ndarray, step_values: np.ndarray, step_rewards: np.ndarray, step_terminals: np.ndarray,step_players:np.ndarray):
        self.partial_observations[self.current_step] = np.array(
            step_partial_observations)
        self.full_observations[self.current_step] = np.array(
            step_full_observations)
        self.actions[self.current_step] = step_actions
        self.legal_actions_masks[self.current_step] = step_legal_actions_masks
        self.log_probs[self.current_step] = step_log_probs
        self.rewards[self.current_step] = step_rewards
        self.values[self.current_step] = step_values
        self.terminals[self.current_step] = step_terminals
        self.players[self.current_step] = step_players
        self.current_step += 1

    def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,np.ndarray,np.ndarray]:
        return self.partial_observations.swapaxes(0, 1), self.full_observations.swapaxes(0, 1), self.actions.swapaxes(0, 1), self.legal_actions_masks.swapaxes(0,1),self.log_probs.swapaxes(0, 1), self.values.swapaxes(0, 1), self.rewards.swapaxes(0, 1), self.terminals.swapaxes(0, 1),self.players.swapaxes(0,1)
