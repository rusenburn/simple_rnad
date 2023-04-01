import time
import os
import copy
import torch as T
from torch.distributions import Categorical
import numpy as np
from collections import deque
from torch.optim import Adam
from torch.nn.utils.clip_grad import clip_grad_norm_,clip_grad_value_
from typing import Callable,Sequence
from rnad.algorithms.ppo.memory_buffer import MemoryBuffer
from rnad.games.game import Game,VecGame
from rnad.match import Match
from rnad.networks import ActorClippedLinearNetwork, ActorLinearNetwork, ClippedActorResNetwork, CriticLinearNetwork, PytorchNetwork,ActorResNetwork,CriticResNetwork
from rnad.players import NNPlayer, RandomPlayer

EPS = 1e-8
class PPO():
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
        save_name:str|None = None
        ) -> None:
    
        self._vec_game = VecGame(game_fns)
        self._n_workers = self._vec_game.n_games
        g = game_fns[0]()
        self._n_game_actions = g.n_actions
        self._observation_space = g.observation_space

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
        if len(self._observation_space) > 2:
            self.actor : PytorchNetwork = ActorResNetwork(self._observation_space,self._n_game_actions)
            self.critic : PytorchNetwork = CriticResNetwork(self._observation_space)
        else:
            self.actor : PytorchNetwork = ActorLinearNetwork(self._observation_space,self._n_game_actions)
            self.critic : PytorchNetwork = CriticLinearNetwork(self._observation_space)
        
        self.actor_optim = Adam(self.actor.parameters(),lr=lr,eps=EPS)
        self.critic_optim = Adam(self.critic.parameters(),lr=lr,eps=EPS)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.testing_game_fn = testing_game_fn

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
        test_interval = 50000
        next_log = log_interval 
        next_test = test_interval
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
            actor_scheduler.step()
            critic_scheduler.step()
            
            # TODO(rusenburn): plot summaries
    

    def _choose_actions(self,partial_observations:np.ndarray,full_observations:np.ndarray,legal_actions_masks:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray]:
        partial_obs_t = T.tensor(partial_observations,dtype=T.float32,device=self.device)
        full_obs_t = T.tensor(full_observations,dtype=T.float32,device=self.device)
        legal_actions_masks_t = T.tensor(legal_actions_masks,dtype=T.int32,device=self.device)
        probs:T.Tensor
        with T.no_grad():
            x = self.actor(partial_obs_t)
            values:T.Tensor = self.critic(full_obs_t)
        
        if isinstance(x,tuple):
            probs,_ = x
        else:
            probs = x
        legal_probs = probs * legal_actions_masks_t
        legal_probs[(legal_probs<0.003).logical_and(legal_probs>0)] = EPS
        legal_probs /= legal_probs.sum(dim=-1,keepdim=True)
        legal_action_sample = Categorical(legal_probs)
        action_dist = Categorical(probs)
        actions = legal_action_sample.sample()
        log_probs :T.Tensor= action_dist.log_prob(actions)
        return actions.cpu().numpy(),log_probs.cpu().numpy(),values.squeeze(-1).cpu().numpy()
    
    def _train_network(self,last_values:np.ndarray,last_players:np.ndarray):
        batch_size = (self._step_size*self._n_workers) // self._n_batches

        partial_obs_samples ,full_obs_samples , actions_samples, legal_action_masks_samples, log_probs_samples , value_samples,reward_samples,terminal_samples,players_samples = self.memory_buffer.sample()

        all_advantages_tensor : T.Tensor= self._calculate_advantages(
            reward_samples,value_samples,terminal_samples,players_samples,last_values,last_players)
        
        if self._normalize_adv:
            normalized_advantages_tensor = (
                        all_advantages_tensor - all_advantages_tensor.mean())/(all_advantages_tensor.std()+EPS)
        else:
            normalized_advantages_tensor = all_advantages_tensor
        
        value_loss_info = T.zeros((self._n_epochs,self._n_batches),dtype=T.float32,device=self.device)
        policy_loss_info = T.zeros((self._n_epochs,self._n_batches,),dtype=T.float32,device=self.device)
        entropies_info = T.zeros((self._n_epochs,self._n_batches),dtype=T.float32,device=self.device)
        total_losses_info = T.zeros((self._n_epochs,self._n_batches),dtype=T.float32,device=self.device)

        partiall_obs_arr = partial_obs_samples.reshape(
                (self._n_workers*self._step_size, *self._observation_space))

        full_obs_arr = full_obs_samples.reshape(
                (self._n_workers*self._step_size, *self._observation_space))
        actions_arr = actions_samples.reshape(
            (self._n_workers*self._step_size,))
        log_probs_arr = log_probs_samples.reshape(
            (self._n_workers*self._step_size,))
        values_arr = value_samples.reshape(
            (self._n_workers*self._step_size,))
        legal_action_masks_arr = legal_action_masks_samples.reshape(
            (self._n_workers*self._step_size,self._n_game_actions))

        all_partiall_obs_tensor = T.tensor(partiall_obs_arr,dtype=T.float32,device=self.device)
        all_full_obs_tensor = T.tensor(full_obs_arr,dtype=T.float32,device=self.device)
        all_actions_tensor = T.tensor(actions_arr,dtype=T.int32,device=self.device)
        all_log_probs_tensor = T.tensor(log_probs_arr,dtype=T.float32,device=self.device)
        all_values_tensor = T.tensor(values_arr,dtype=T.float32, device=self.device)
        all_legal_actions_masks_tensor = T.tensor(legal_action_masks_arr,dtype=T.int32,device=self.device)

        # TODO :DEBUG
        all_returns_tensor = all_values_tensor + all_advantages_tensor

        explained_variance:float = self._calculate_explained_variance(all_values_tensor,all_returns_tensor)

        for epoch in range(self._n_epochs):
            batch_starts = np.arange(
                0, self._n_workers*self._step_size, batch_size)
            indices = np.arange(self._n_workers*self._step_size, dtype=np.int32)
            np.random.shuffle(indices)
            batches =[indices[i:i+batch_size] for i in batch_starts]

            for i , batch in enumerate(batches):
                partial_obs_tensor = all_partiall_obs_tensor[batch]
                full_obs_tensor = all_full_obs_tensor[batch]
                old_log_probs_tensor = all_log_probs_tensor[batch]
                actions_tensor = all_actions_tensor[batch]
                values_tensor = all_values_tensor[batch]
                returns_tensor = all_returns_tensor[batch]
                legal_action_masks_tensor = all_legal_actions_masks_tensor[batch]
                if self._normalize_adv:
                    advantages_tensor = normalized_advantages_tensor[batch].clone()
                else:
                    advantages_tensor = all_advantages_tensor[batch].clone()
                

                probs : T.Tensor
                probs  = self.actor(partial_obs_tensor)

                # nans= probs.isnan()
                # probs = probs*legal_action_masks_tensor
                # probs /= probs.sum(dim=-1,keepdim=True)
                critic_values : T.Tensor = self.critic(full_obs_tensor)
                critic_values = critic_values.squeeze()

                dist = Categorical(probs)

                entropy : T.Tensor= dist.entropy().mean()

                # Actor Loss
                new_log_probs :T.Tensor = dist.log_prob(actions_tensor)
                prob_ratio : T.Tensor = (new_log_probs-old_log_probs_tensor).exp()
                weighted_probs = advantages_tensor * prob_ratio
                clipped_prob_ratio = prob_ratio.clamp(1-self._clip_ratio,1+self._clip_ratio)
                weighted_clipped_probs = advantages_tensor * clipped_prob_ratio
                actor_loss = -T.min(weighted_probs,weighted_clipped_probs).mean()


                # Critic Loss
                old_value_predictions = values_tensor
                clipped_values_predictions = old_value_predictions + T.clamp(critic_values-old_value_predictions,-self._clip_ratio,self._clip_ratio)
                critic_loss_1 = (returns_tensor- critic_values)**2
                critic_loss_2 = (returns_tensor-clipped_values_predictions)**2
                critic_loss = 0.5 * T.max(critic_loss_1,critic_loss_2).mean()
                total_loss = actor_loss + self._critic_coef*critic_loss - self._entropy_coef * entropy

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()

                total_loss.backward()

                

                if self._max_grad_norm:
                    clip_grad_value_(self.actor.parameters(),10000)
                    clip_grad_norm_(self.actor.parameters(),
                                    max_norm=self._max_grad_norm)
                    
                    clip_grad_value_(self.critic.parameters(),10000)
                    clip_grad_norm_(self.critic.parameters(),
                                    max_norm=self._max_grad_norm)
                
                self.actor_optim.step()
                self.critic_optim.step()
                with T.no_grad():
                    value_loss_info[epoch,i] = critic_loss.clone()
                    policy_loss_info[epoch,i] = actor_loss.clone()
                    entropies_info[epoch,i] = entropy.clone()
                    total_losses_info[epoch,i] = total_loss.clone()
        with T.no_grad():
            value_loss_mean = value_loss_info.flatten().mean()
            policy_loss_mean = policy_loss_info.flatten().mean()
            entropy_mean= entropies_info.flatten().mean()
            total_loss_mean = total_losses_info.flatten().mean()
        return policy_loss_mean.cpu().item(),value_loss_mean.cpu().item(),entropy_mean.cpu().item(),total_loss_mean.cpu().item(),explained_variance

    def _calculate_explained_variance(self,predictions:T.Tensor,target:T.Tensor)->float:
        assert predictions.ndim == 1 and target.ndim ==1
        target_var = target.var() + EPS
        unexplained_var_ratio = (target-predictions).var()/ target_var
        explained_var_ratio = 1- unexplained_var_ratio
        return explained_var_ratio.cpu().item()

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
                # next_ = (self._gamma* self._gae_lam * adv_arr[worker][step+1] * (1-is_terminal))
                # if current_player != next_player:
                #     next_ = -next_
                next_ = 0
                adv_arr[worker][step] = delta + next_
        
        adv_arr = adv_arr[:,:-1]

        advantages_tensor = T.tensor(adv_arr.flatten(),dtype=T.float32,device=self.device)
        return advantages_tensor
    def _pit(self,net_1:PytorchNetwork,net_2:PytorchNetwork|None,n_sets:int)->float|None:
        if self.testing_game_fn is None:
            return None
        testing_game_fn = self.testing_game_fn
        player_2 = RandomPlayer() if net_2 is None else NNPlayer(net_2)
        match_ = Match(testing_game_fn,NNPlayer(net_1),player_2,n_sets=n_sets)
        score = match_.start()
        return (score[0]*2 + score[1]) / score.sum() / 2

def hook_pi(pi:T.Tensor,lr:float):
    def grad_fn(grad:T.Tensor):
        with T.no_grad():
            grad = (pi - T.clamp(grad*lr+pi ,-2.0,2.0))/lr
    hook = pi.register_hook(grad_fn)
    return hook