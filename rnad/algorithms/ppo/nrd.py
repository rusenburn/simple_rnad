from typing import Callable, Sequence
import numpy as np
from rnad.games.game import Game
from .ppo import PPO
import torch as T
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_,clip_grad_value_


EPS = 1e-8

class NRD(PPO):
    def __init__(self, game_fns: Sequence[Callable[[], Game]], total_steps=1000000, step_size=128, n_batches=4, n_epochs=4, gamma=0.9, gae_lam=0.95, clip_ratio=0.2, lr=0.00025, entropy_coef=0.01, critic_coef=0.5, max_grad_norm=0.5, normalize_adv=True, decay_lr=True, testing_game_fn: Callable[[], Game] | None = None, save_name: str | None = None) -> None:
        super().__init__(game_fns, total_steps, step_size, n_batches, n_epochs, gamma, gae_lam, clip_ratio, lr, entropy_coef, critic_coef, max_grad_norm, normalize_adv, decay_lr, testing_game_fn, save_name)
    
    def _train_network(self, last_values: np.ndarray, last_players: np.ndarray):
        threshold = 2
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
            (self._n_workers*self._step_size,)).astype(np.int32)
        all_one_hot_actions_ar = np.zeros(
                        (len(actions_arr), self._n_game_actions), dtype=np.int32)
        all_one_hot_actions_ar[np.arange(len(actions_arr)), actions_arr] = 1
        # all_one_hot_actions_ar[np.arange(len(actions_arr)), actions_arr] = 1
        log_probs_arr = log_probs_samples.reshape(
            (self._n_workers*self._step_size,))
        values_arr = value_samples.reshape(
            (self._n_workers*self._step_size,))
        legal_action_masks_arr = legal_action_masks_samples.reshape(
            (self._n_workers*self._step_size,self._n_game_actions))

        all_partiall_obs_tensor = T.tensor(partiall_obs_arr,dtype=T.float32,device=self.device)
        all_full_obs_tensor = T.tensor(full_obs_arr,dtype=T.float32,device=self.device)
        all_actions_tensor = T.tensor(actions_arr,dtype=T.int32,device=self.device)
        all_one_hot_actions_tensor = T.tensor(all_one_hot_actions_ar,dtype=T.int32,device=self.device)
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
                one_hot_actions_tensor = all_one_hot_actions_tensor[batch]
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
                logits = dist.logits
                logits = logits * one_hot_actions_tensor

                entropy : T.Tensor= dist.entropy().mean()

                thresh_center = T.zeros_like(logits)
                can_decrease = logits - thresh_center > -threshold
                can_increase = logits - thresh_center < threshold
                
                advantages_tensor = advantages_tensor.unsqueeze(-1).expand(-1,self._n_game_actions)
                force_negative = advantages_tensor.clamp(max=0.0)
                force_positive = advantages_tensor.clamp(min=0.0)
                clipped_force = can_decrease*force_negative + can_increase*force_positive
                o:T.Tensor = logits * clipped_force.detach()
                actor_loss = -o.sum(-1).mean()

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