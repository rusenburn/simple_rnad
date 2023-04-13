
import torch as T
from typing import Callable, Sequence
from rnad.algorithms.ppo.ppo import PPO
import os
import time
import numpy as np
from collections import deque
from torch.optim import Adam
from rnad.games.game import Game, SinglePlayerGame, VecGame
from rnad.networks import ActorClippedLinearNetwork, ActorLinearNetwork, ActorResNetwork, CriticLinearNetwork, CriticResNetwork, PytorchNetwork
from rnad.algorithms.ppo.memory_buffer import MemoryBuffer
import copy

EPS = 1e-8
class PastSelf(PPO):
    def __init__(self,
        game_fns:Sequence[Callable[[],Game]],
        total_steps=1_000_000,
        step_size=128,
        n_batches=4,
        n_epochs=4,
        gamma=0.9,
        gae_lam=0.95,
        clip_ratio=0.2,
        lr=2.5e-4,
        entropy_coef=0.01,
        critic_coef=0.5,
        max_grad_norm=0.5,
        normalize_adv=True,
        decay_lr=True ,
        testing_game_fn:Callable[[],Game]|None=None,
        save_name:str|None=None) -> None:

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        g = game_fns[0]()
        self._n_game_actions = g.n_actions
        self._observation_space = g.observation_space
        if len(self._observation_space) > 2:
            self.actor : PytorchNetwork = ActorResNetwork(self._observation_space,self._n_game_actions)
            self.critic : PytorchNetwork = CriticResNetwork(self._observation_space)
        else:
            self.actor : PytorchNetwork = ActorClippedLinearNetwork(self._observation_space,self._n_game_actions,blocks=3)
            self.critic : PytorchNetwork = CriticLinearNetwork(self._observation_space)
        copies_count = len(game_fns)//4
        nn_count = len(game_fns) - copies_count
        self.copies = [copy.deepcopy(self.actor).to(self.device) for _ in range(copies_count)]
        all_nns = [*self.copies]
        [all_nns.append(self.actor) for _ in range(nn_count)]
        self.actor.to(self.device)
        game_fns = [lambda :SinglePlayerGame(game_fn,nn) for game_fn,nn in zip(game_fns,all_nns)]
        self._vec_game = VecGame(game_fns)
        self._n_workers = self._vec_game.n_games
        assert (self._n_workers*step_size) % n_batches == 0

        # Data Collection 
        self._total_steps = int(total_steps) 
        self._step_size = int(step_size)

        # Training Hyperparameters
        self._n_batches = int(n_batches)
        self._n_epochs = int(n_epochs)
        self._gamma = gamma
        self._gae_lam = gae_lam
        self._clip_ratio = clip_ratio
        self._lr = lr
        self._entropy_coef = entropy_coef
        self._critic_coef = critic_coef
        self._decay_lr = decay_lr

        # Improvements
        self._max_grad_norm = max_grad_norm
        self._normalize_adv = normalize_adv

        self.memory_buffer = MemoryBuffer(self._observation_space,self._n_game_actions,self._n_workers,self._step_size)
        
        
        self.actor_optim = Adam(self.actor.parameters(),lr=lr,eps=EPS)
        self.critic_optim = Adam(self.critic.parameters(),lr=lr,eps=EPS)

        
        self.testing_game_fn = testing_game_fn
        self.set_enemy_interval = 10000
        self.next_nn_idx = 0
        self.save_name = save_name if save_name is not None else ""
    
    def run(self)->None:
        t_start = time.perf_counter()
        partial_observations : np.ndarray
        full_observations :np.ndarray
        players :np.ndarray

        partial_observations,full_observations,legal_actions_masks,players = self._vec_game.reset()
        total_iterations = int(self._total_steps//self._n_workers)
        if self._decay_lr:
            def lr_fn(x:int) : return (total_iterations-x )/total_iterations
        else:
            def lr_fn(x:int):return 1.
        actor_scheduler = T.optim.lr_scheduler.LambdaLR(self.actor_optim,lr_fn)
        critic_scheduler = T.optim.lr_scheduler.LambdaLR(self.critic_optim,lr_fn)


        summary_time_step = []
        summary_duration = []

        value_losses = deque(maxlen=100)
        policy_losses = deque(maxlen=100)
        entropies = deque(maxlen=100)
        total_losses = deque(maxlen=100)
        explained_variances = deque(maxlen=100)

        log_interval = 10000
        next_log = log_interval 
        
        test_interval = 50000
        next_test = test_interval

        change_interval = 5000
        next_change = change_interval
        previous = copy.deepcopy(self.actor)
        test_previous_win_ratio : float|None = None
        random_win_ratio : float|None = None
        
        for iteration in range(1,int(self._total_steps//self._n_workers)+1):
            actions ,log_probs,values = self._choose_actions(partial_observations,full_observations,legal_actions_masks)

            new_partial_obs , new_full_obs ,new_legal_actions_masks,rewards , dones , new_players = self._vec_game.step(actions)

            self.memory_buffer.save(
                step_partial_observations=partial_observations,
                step_full_observations=full_observations,
                step_log_probs=log_probs,
                step_legal_actions_masks=legal_actions_masks,
                step_values=values,
                step_rewards=rewards,
                step_actions=actions,
                step_terminals=dones,
                step_players=players
            )

            partial_observations = new_partial_obs
            full_observations = new_full_obs
            players = new_players
            legal_actions_masks = new_legal_actions_masks
            if iteration % self._step_size == 0:
                with T.no_grad():
                    last_values_t :T.Tensor= self.critic(T.tensor(full_observations,dtype=T.float32,device=self.device))
                    last_values_ar = last_values_t.cpu().numpy()
                
                p_loss,v_loss,ent,total_loss,explained_variance = self._train_network(last_values_ar,new_players)

                value_losses.append(v_loss)
                policy_losses.append(p_loss)
                entropies.append(ent)
                total_losses.append(total_loss)
                explained_variances.append(explained_variance)
                self.memory_buffer.reset()
                self.actor.save_model(os.path.join("tmp", f"{self.save_name}_actor.pt"))
                self.critic.save_model(os.path.join("tmp",f"{self.save_name}_critic.pt"))
                T.save(self.actor_optim, os.path.join("tmp", f"{self.save_name}_actor_optim.pt"))
                T.save(self.critic_optim, os.path.join(
                    "tmp", f"{self.save_name}_critic_optim.pt"))
                
                # if iteration*self._n_workers % 1_000 < self._n_workers:
                if iteration*self._n_workers >= next_test:
                    next_test += test_interval
                    test_previous_win_ratio = self._pit(self.actor,previous,100)
                    random_win_ratio = self._pit(self.actor,None,100)
                    previous.load_state_dict(self.actor.state_dict())

                if iteration*self._n_workers >= next_log:
                    next_log += log_interval
                    total_duration = time.perf_counter()-t_start
                    steps_done = self._n_workers * iteration
                    fps = steps_done // total_duration
                    total_loss = np.mean(np.array(total_losses))
                    v_loss = np.mean(np.array(value_losses))
                    p_loss = np.mean(np.array(p_loss))
                    ent = np.mean(np.array(entropies))
                    explained_variance = np.mean(np.array(explained_variances))
                    print(f"**************************************************")
                    print(f"Iteration:      {iteration} of {total_iterations}")
                    print(f"Learning rate:  {actor_scheduler.get_last_lr()[0]:0.3e}")
                    print(f"FPS:            {fps}")
                    print(
                        f"Total Steps:    {steps_done} of {int(self._total_steps)}")
                    print(f"Total duration: {total_duration:0.2f} seconds")
                    print(f"Total Loss:     {total_loss:0.3f}")
                    print(f"Value Loss:     {v_loss:0.3f}")
                    print(f"Policy Loss:    {p_loss:0.3f}")
                    print(f"Entropy:        {ent:0.3f}")
                    print(f"E-Var-ratio:    {explained_variance:0.3f}")
                    if test_previous_win_ratio is not None:
                        print(f"Test-win-ratio: {test_previous_win_ratio:0.3f}")
                        test_previous_win_ratio = None
                    if random_win_ratio is not None:
                        print(f"Random-w-ratio: {random_win_ratio:0.3f}")
                        random_win_ratio = None

                    value_losses.clear()
                    policy_losses.clear()
                    entropies.clear()
                    total_losses.clear()
                    explained_variances.clear()
                if iteration * self._n_workers >= next_change:
                    next_change += change_interval
                    print(f'... Settings Current Policy as an Enemy ...')
                    state_dict = self.actor.state_dict()
                    next_index = self.next_nn_idx % len(self.copies)
                    next_copy = self.copies[next_index]
                    next_copy.load_state_dict(state_dict)
                    self.next_nn_idx+=1
                    change_interval+=1000
            actor_scheduler.step()
            critic_scheduler.step()
    
    def _calculate_advantages(self,rewards:np.ndarray,values:np.ndarray,terminals:np.ndarray,players:np.ndarray,last_values:np.ndarray,last_players:np.ndarray):
        adv_arr = np.zeros((self._n_workers,self._step_size+1),dtype=np.float32)
        next_val : float
        next_player : int
        for worker in range(self._n_workers):
            for step in  reversed(range(self._step_size)):
                current_player = players[worker][step]
                current_reward = rewards[worker][step][current_player]
                current_val = values[worker][step]
                is_terminal = int(terminals[worker][step])

                if step == self._step_size-1 : # Last Step
                    next_player = last_players[worker]
                    next_val = last_values[worker]
                    if current_player != next_player:
                        next_val *=-1
                else:
                    next_player = players[worker][step+1]
                    next_val = values[worker][step+1]
                    if current_player != next_player:
                        next_val *=-1
                
                delta = current_reward + (self._gamma * next_val * (1-is_terminal)) - current_val
                next_adv:float = adv_arr[worker][step+1]
                if current_player !=next_player and not is_terminal:
                    next_adv = -next_adv
                next_ = self._gamma* self._gae_lam * next_adv * (1-is_terminal)
                
                # next_ = (self._gamma* self._gae_lam * adv_arr[worker][step+1] * (1-is_terminal))
                # if current_player != next_player:
                #     next_ = -next_
                # next_ = 0
                adv_arr[worker][step] = delta + next_
        
        adv_arr = adv_arr[:,:-1]

        advantages_tensor = T.tensor(adv_arr.flatten(),dtype=T.float32,device=self.device)
        return advantages_tensor
    
