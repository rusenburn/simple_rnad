
from abc import ABC , abstractmethod
from typing import Sequence
import torch as T
from torch.nn.utils.clip_grad import clip_grad_norm_,clip_grad_value_
import numpy as np
from rnad.networks import ActorNetwork, CriticNetwork


EPS = 1e-8

class PolicyTrainer(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def train_policy(self,
                     all_partial_observations:np.ndarray,
                     all_full_observations:np.ndarray,
                     all_actions:np.ndarray,
                     all_legal_action_masks:np.ndarray,
                     all_log_probs:np.ndarray,
                     all_adv:np.ndarray,
                     all_players:np.ndarray,
                     is_aux_training:bool,
                     log_dict:dict):
        raise NotImplementedError()

    @abstractmethod
    def train_critic(self,
                     full_obs:np.ndarray,
                     returns:np.ndarray,
                     players:np.ndarray,
                     is_aux_training:bool,
                     log_dict:dict):
        raise NotImplementedError()
    
    @abstractmethod
    def set_trained_network(self,actors:Sequence[ActorNetwork],critics:Sequence[CriticNetwork]):
        raise NotImplementedError()
    
class AuxTrainer(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    

    @abstractmethod
    def train_aux(self,partial_obs:np.ndarray,
                  legal_action_masks:np.ndarray,
                  old_dirtibutions:np.ndarray,
                  players:np.ndarray,
                  log_dict:dict):
        raise NotImplementedError()
    
    @abstractmethod
    def set_trained_network(self,actors:Sequence[ActorNetwork],actor_optims):
        raise NotImplementedError()


class NeurdPolicyTrainer(PolicyTrainer):
    def __init__(self, n_batches=4, n_aux_batches=8,logits_threshold=2,entropy_coef=0.01, lr=0.00025, decay_lr=True, normalize_advantages=True,max_grad_norm=0.5) -> None:
        super().__init__()
        self.n_batches = n_batches
        self.n_aux_batches = n_aux_batches
        self.entropy_coef = entropy_coef
        self.logits_threshold = logits_threshold
        self.lr = lr
        self.decay_lr = decay_lr
        self.normalize_advantages = normalize_advantages
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
        self.actors :Sequence[ActorNetwork] = []
        self.actor_optims : Sequence[T.optim.Adam] = []
        self.critics : Sequence[CriticNetwork] = []
        self.critic_optims : Sequence[T.optim.Adam] = []
        self._max_grad_norm = max_grad_norm
    

    def train_policy(self,
                     all_partial_observations: np.ndarray, 
                     all_full_observations: np.ndarray, 
                     all_actions: np.ndarray, 
                     all_legal_action_masks: np.ndarray, 
                     all_log_probs: np.ndarray, 
                     all_adv: np.ndarray,
                     players:np.ndarray,
                     is_aux_training:bool,
                     log_dict:dict):
        n_batches = self.n_aux_batches if is_aux_training else self.n_batches
        all_policy_losses = [list() for _ in self.actors]
        all_entropies = [list() for _ in self.actors]
        # all_one_hot_actions = np.zeros_like(all_legal_action_masks)
        # all_one_hot_actions[np.arange(len(all_actions)),all_actions.astype(np.int32)] = 1
        for player,(actor,optimizer) in enumerate(zip(self.actors,self.actor_optims)):
            predicate = players == player
            n_examples = len(np.argwhere(predicate))
            batch_size = n_examples // n_batches


            player_partial_observations_t = T.tensor(all_partial_observations[predicate],dtype=T.float32,device=self.device)
            player_actions_t = T.tensor(all_actions[predicate],dtype=T.int32,device=self.device)
            player_log_probs_t = T.tensor(all_log_probs[predicate],dtype=T.float32,device=self.device)
            player_legal_action_masks_t = T.tensor(all_legal_action_masks[predicate],dtype=T.int32,device=self.device)
            # player_one_hot_actions = T.tensor(all_one_hot_actions[predicate])

            player_advantages_t = T.tensor(all_adv[predicate],dtype=T.float32,device=self.device)

            if self.normalize_advantages:
                player_advantages_t = (player_advantages_t - player_advantages_t.mean()) / (player_advantages_t.std()+1e-8)
            batch_starts = np.arange(0,n_examples,batch_size)
            indices = np.arange(n_examples,dtype=np.int32)
            batches = [indices[i:i+batch_size] for i in batch_starts]

            for i , batch in enumerate(batches):
                partial_obs_t = player_partial_observations_t[batch]
                # old_log_prob_t = player_log_probs_t[batch]
                actions_t = player_actions_t[batch]
                legal_action_masks_t = player_legal_action_masks_t[batch]
                advantages_t = player_advantages_t[batch]

                probs ,logits = actor.predict(partial_obs_t)
                fixed_probs = probs * legal_action_masks_t
                fixed_probs /=fixed_probs.sum(dim=-1,keepdim=True)
                dist = T.distributions.Categorical(fixed_probs)
                entropy:T.Tensor = dist.entropy().mean()
                played_logits = logits[T.arange(len(batch),dtype=T.int64),actions_t.to(dtype=T.int64)]
                
                with T.no_grad():
                    thresh_center = T.zeros_like(played_logits)
                    can_decrease = played_logits - thresh_center > -self.logits_threshold
                    can_increase = played_logits - thresh_center < self.logits_threshold
                    negative_force = advantages_t.clamp(max=0)
                    positive_force = advantages_t.clamp(min=0)
                    clipped_force = can_decrease*negative_force + can_increase * positive_force

                weighted_logits = clipped_force.detach() * played_logits
                actor_loss = -weighted_logits.mean()


                dist = T.distributions.Categorical(fixed_probs)
                entropy:T.Tensor = dist.entropy().mean()

                # new_log_prob : T.Tensor = dist.log_prob(actions_t)
                # prob_ratio : T.Tensor = (new_log_prob - old_log_prob_t).exp()
                # weighted_probs = advantages_t*prob_ratio
                # clipped_prob_ratio = prob_ratio.clamp(1-self.clip_ratio , 1+self.clip_ratio)
                # weighted_clipped_probs = clipped_prob_ratio * advantages_t
                # actor_loss = -T.min(weighted_probs,weighted_clipped_probs).mean()
                # total_loss =  actor_loss - self.entropy_coef * entropy

                optimizer.zero_grad()

                actor_loss.backward()

                if self._max_grad_norm:
                    clip_grad_value_(actor.parameters(),10000)
                    clip_grad_norm_(actor.parameters(),self._max_grad_norm)
                optimizer.step()

                all_policy_losses[player].append(actor_loss.cpu().item())
                all_entropies[player].append(entropy.cpu().item())
        for player,_ in enumerate(self.actors):
            loss_mean = np.mean(all_policy_losses[player])
            loss_std = np.std(all_policy_losses[player]) if len(all_policy_losses[player]) else 0

            entropy_avg = np.mean(all_entropies[player])
            log_dict[f"Player {player} Policy Loss"] = f"{loss_mean:0.3f}"
            log_dict[f"Player {player} Entropy"] = f"{entropy_avg:0.3f}"

    def train_critic(self,full_obs:np.ndarray,returns:np.ndarray,players:np.ndarray,is_aux_training:bool,log_dict:dict):
        n_batches = self.n_aux_batches if is_aux_training else self.n_batches
        all_critic_losses = [list() for _ in self.critics ]
        for player,(critic,optim) in enumerate(zip(self.critics,self.critic_optims)):
            predicate =  players == player
            n_examples = len(np.argwhere(predicate))
            batch_size = n_examples // n_batches

            player_full_obs_t = T.tensor(full_obs[predicate],dtype=T.float32,device=self.device)
            # player_actions_t = T.tensor(actions[predicate],dtype=T.int32,device=self.device)
            player_returns_t = T.tensor(returns[predicate],dtype=T.float32,device=self.device)

            batch_starts = np.arange(0,n_examples,batch_size)
            indices = np.arange(n_examples,dtype=np.int32)
            batches = [indices[i:i+batch_size] for i in batch_starts]

            for i, batch in enumerate(batches):
                full_obs_t = player_full_obs_t[batch]
                returns_t = player_returns_t[batch]

                critic_values = self.critics[player].evaluate(full_obs_t)
                critic_values = critic_values.squeeze()

                critic_loss = (0.5 * (returns_t - critic_values)**2).mean()

                optim.zero_grad()

                critic_loss.backward()

                if self._max_grad_norm:
                    clip_grad_value_(critic.parameters(),10_000)
                    clip_grad_norm_(critic.parameters(),0.5)
                
                optim.step()

                all_critic_losses[player].append(critic_loss.cpu().item())
        
        for player,_ in enumerate(self.critics):
            loss_mean = np.mean(all_critic_losses[player])
            loss_std = np.std(all_critic_losses[player]) if len(all_critic_losses[player]) else 0
            log_dict[f"Player {player} Critic Loss"] = f"{loss_mean:0.3f}"
    
    
    
    def set_trained_network(self,actors:Sequence[ActorNetwork],critics:Sequence[CriticNetwork]):
        self.actors = actors
        self.actor_optims = [T.optim.Adam(actor.parameters(),lr=self.lr) for actor in actors]
        self.critics = critics
        self.critic_optims = [ T.optim.Adam(critic.parameters(),lr=self.lr) for critic in critics]
        return actors,critics,self.actor_optims,self.critic_optims

class LegalActionAuxTrainer(AuxTrainer):
    def __init__(self,n_batches:int=8) -> None:
        super().__init__()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.n_batches = n_batches
        self.actors :Sequence[ActorNetwork] = []
        self.optims :Sequence[T.optim.Adam] = []


    def train_aux(self, all_partial_obs: np.ndarray, all_legal_action_masks: np.ndarray, all_old_dirtibutions: np.ndarray, all_players: np.ndarray,log_dict:dict):
        all_kl_losses = [list() for _ in self.actors]
        for player,(actor,optim) in enumerate(zip(self.actors,self.optims)):
            predicate = all_players == player
            player_partial_obs_t = T.tensor(all_partial_obs[predicate],dtype=T.float32,device=self.device)
            player_legal_action_masks_t = T.tensor(all_legal_action_masks[predicate],dtype=T.int32,device=self.device)
            player_old_distributions = T.tensor(all_old_dirtibutions[predicate],dtype=T.float32,device=self.device)

            player_old_distributions = player_old_distributions * player_legal_action_masks_t
            player_old_distributions = player_old_distributions + EPS
            player_old_distributions /= player_old_distributions.sum(dim=-1,keepdim=True)
            player_old_distributions = player_old_distributions.detach()

            n_examples = len(np.argwhere(predicate))
            batch_size = n_examples // self.n_batches

            batch_starts = np.arange(0,n_examples,batch_size)
            indices = np.arange(n_examples,dtype=np.int32)
            batches = [indices[i:i+batch_size] for i in batch_starts]

            for i,batch in enumerate(batches):
                partial_obs = player_partial_obs_t[batch]
                old_distributions = player_old_distributions[batch]

                new_probs,_ = actor.predict(partial_obs)

                kl_loss = T.nn.functional.kl_div(new_probs.log(),old_distributions,reduction="batchmean").mean()

                optim.zero_grad()
                kl_loss.backward()
                optim.step()

                all_kl_losses[player].append(kl_loss.cpu().item())
        
        for player,_ in enumerate(self.actors):
            kl_mean = np.mean(all_kl_losses[player])
            log_dict[f"Player {player} Legal Action KLoss"] = f"{kl_mean:0.3f}"
    
    def set_trained_network(self, actors: Sequence[ActorNetwork], actor_optims:Sequence[T.optim.Adam]):
        self.actors = actors
        self.optims = actor_optims
        return self.actors,self.optims


class PPOPolicyTrainer(PolicyTrainer):
    def __init__(self, n_batches=4, n_aux_batches=8,clip_ratio=0.2,entropy_coef=0.01, lr=0.00025, decay_lr=True, normalize_advantages=True,max_grad_norm=0.5) -> None:
        super().__init__()
        self.n_batches = n_batches
        self.n_aux_batches = n_aux_batches
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.decay_lr = decay_lr
        self.normalize_advantages = normalize_advantages
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
        self.actors :Sequence[ActorNetwork] = []
        self.actor_optims : Sequence[T.optim.Adam] = []
        self.critics : Sequence[CriticNetwork] = []
        self.critic_optims : Sequence[T.optim.Adam] = []
        self._max_grad_norm = max_grad_norm
    

    def train_policy(self,
                     all_partial_observations: np.ndarray, 
                     all_full_observations: np.ndarray, 
                     all_actions: np.ndarray, 
                     all_legal_action_masks: np.ndarray, 
                     all_log_probs: np.ndarray, 
                     all_adv: np.ndarray,
                     players:np.ndarray,
                     is_aux_training:bool,
                     log_dict:dict):
        n_batches = self.n_aux_batches if is_aux_training else self.n_batches
        all_policy_losses = [list() for _ in self.actors]
        all_entropies = [list() for _ in self.actors]
        for player,(actor,optimizer) in enumerate(zip(self.actors,self.actor_optims)):
            predicate = players == player
            n_examples = len(np.argwhere(predicate))
            batch_size = n_examples // n_batches

            player_partial_observations_t = T.tensor(all_partial_observations[predicate],dtype=T.float32,device=self.device)
            player_actions_t = T.tensor(all_actions[predicate],dtype=T.int32,device=self.device)
            player_log_probs_t = T.tensor(all_log_probs[predicate],dtype=T.float32,device=self.device)
            player_legal_action_masks_t = T.tensor(all_legal_action_masks[predicate],dtype=T.int32,device=self.device)
            player_advantages_t = T.tensor(all_adv[predicate],dtype=T.float32,device=self.device)

            if self.normalize_advantages:
                player_advantages_t = (player_advantages_t - player_advantages_t.mean()) / (player_advantages_t.std()+1e-8)
            batch_starts = np.arange(0,n_examples,batch_size)
            indices = np.arange(n_examples,dtype=np.int32)
            batches = [indices[i:i+batch_size] for i in batch_starts]

            for i , batch in enumerate(batches):
                partial_obs_t = player_partial_observations_t[batch]
                old_log_prob_t = player_log_probs_t[batch]
                actions_t = player_actions_t[batch]
                legal_action_masks_t = player_legal_action_masks_t[batch]
                advantages_t = player_advantages_t[batch]


                probs ,_ = actor.predict(partial_obs_t)
                fixed_probs = probs * legal_action_masks_t
                fixed_probs /=fixed_probs.sum(dim=-1,keepdim=True)

                dist = T.distributions.Categorical(fixed_probs)
                entropy:T.Tensor = dist.entropy().mean()

                new_log_prob : T.Tensor = dist.log_prob(actions_t)
                prob_ratio : T.Tensor = (new_log_prob - old_log_prob_t).exp()
                weighted_probs = advantages_t*prob_ratio
                clipped_prob_ratio = prob_ratio.clamp(1-self.clip_ratio , 1+self.clip_ratio)
                weighted_clipped_probs = clipped_prob_ratio * advantages_t
                actor_loss = -T.min(weighted_probs,weighted_clipped_probs).mean()
                total_loss =  actor_loss - self.entropy_coef * entropy

                optimizer.zero_grad()

                total_loss.backward()

                if self._max_grad_norm:
                    clip_grad_value_(actor.parameters(),10000)
                    clip_grad_norm_(actor.parameters(),self._max_grad_norm)
                optimizer.step()

                all_policy_losses[player].append(actor_loss.cpu().item())
                all_entropies[player].append(entropy.cpu().item())
        for player,_ in enumerate(self.actors):
            loss_mean = np.mean(all_policy_losses[player])
            loss_std = np.std(all_policy_losses[player]) if len(all_policy_losses[player]) else 0

            entropy_avg = np.mean(all_entropies[player])
            log_dict[f"Player {player} Policy Loss"] = f"{loss_mean:0.3f}"
            log_dict[f"Player {player} Entropy"] = f"{entropy_avg:0.3f}"

    def train_critic(self,full_obs:np.ndarray,returns:np.ndarray,players:np.ndarray,is_aux_training:bool,log_dict:dict):
        n_batches = self.n_aux_batches if is_aux_training else self.n_batches
        all_critic_losses = [list() for _ in self.critics ]
        for player,(critic,optim) in enumerate(zip(self.critics,self.critic_optims)):
            predicate =  players == player
            n_examples = len(np.argwhere(predicate))
            batch_size = n_examples // n_batches

            player_full_obs_t = T.tensor(full_obs[predicate],dtype=T.float32,device=self.device)
            # player_actions_t = T.tensor(actions[predicate],dtype=T.int32,device=self.device)
            player_returns_t = T.tensor(returns[predicate],dtype=T.float32,device=self.device)

            batch_starts = np.arange(0,n_examples,batch_size)
            indices = np.arange(n_examples,dtype=np.int32)
            batches = [indices[i:i+batch_size] for i in batch_starts]

            for i, batch in enumerate(batches):
                full_obs_t = player_full_obs_t[batch]
                returns_t = player_returns_t[batch]

                critic_values = self.critics[player].evaluate(full_obs_t)
                critic_values = critic_values.squeeze()

                critic_loss = (0.5 * (returns_t - critic_values)**2).mean()

                optim.zero_grad()

                critic_loss.backward()

                if self._max_grad_norm:
                    clip_grad_value_(critic.parameters(),10_000)
                    clip_grad_norm_(critic.parameters(),0.5)
                
                optim.step()

                all_critic_losses[player].append(critic_loss.cpu().item())
        
        for player,_ in enumerate(self.critics):
            loss_mean = np.mean(all_critic_losses[player])
            loss_std = np.std(all_critic_losses[player]) if len(all_critic_losses[player]) else 0
            log_dict[f"Player {player} Critic Loss"] = f"{loss_mean:0.3f}"
    
    
    
    def set_trained_network(self,actors:Sequence[ActorNetwork],critics:Sequence[CriticNetwork]):
        self.actors = actors
        self.actor_optims = [T.optim.Adam(actor.parameters(),lr=self.lr) for actor in actors]
        self.critics = critics
        self.critic_optims = [ T.optim.Adam(critic.parameters(),lr=self.lr) for critic in critics]
        return actors,critics,self.actor_optims,self.critic_optims