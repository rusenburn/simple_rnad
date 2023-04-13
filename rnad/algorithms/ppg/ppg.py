from typing import Callable, Sequence
import os
import time
import numpy as np
from enum import Enum
import torch as T
import copy
from torch.distributions.categorical import Categorical
from rnad.algorithms.ppg.memory_buffer import AuxMemoryBuffer, PolicyMemoryBuffer
from rnad.algorithms.ppg.trainers import AuxTrainer, PolicyTrainer
from rnad.games.game import Game,VecGame
from rnad.games.state import State
from rnad.match import Match
from rnad.networks import ActorNetwork, CriticNetwork
from rnad.players import NNPlayer, RandomPlayer, TurnBasedNNPlayer

EPS = 1e-8

class PPG():
    def __init__(
            self,
            game_fns:Sequence[Callable[[],Game]],
            policy_trainer:PolicyTrainer,
            aux_trainer:AuxTrainer,
            max_steps=1_000_000,
            step_size=256,
            n_policy_iterations=16,
            n_policy_epochs=1,
            n_value_epochs=1,
            n_aux_iterations=6,
            gamma= 0.99,
            gae_lam=0.95,
            normalize_rewards=True,
            max_reward_norm=3,
            actors :Sequence[ActorNetwork]=[],
            critics:Sequence[CriticNetwork]=[],
            testing_game_fn:Callable[[],Game]|None=None,
            testing_intervals=50_000,
            log_intervals = 10_000,
            save_name:str = ""
            ) -> None:
        
        self.policy_trainer = policy_trainer
        self.aux_trainer = aux_trainer
        self._game_fns = game_fns
        self._max_steps = max_steps
        self._step_size = step_size
        self._n_workers = len(game_fns)
        self._n_policy_iterations = n_policy_iterations
        self._n_policy_epochs = n_policy_epochs
        self._n_value_epochs = n_value_epochs
        self._n_aux_iterations = n_aux_iterations

        self._n_players = 2
        self._game_vec = VecGame(game_fns)
        self._n_game_actions = self._game_vec._n_actions
        self._observations_shape = self._game_vec._single_obs_space

        self._device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self._actors : Sequence[ActorNetwork] = [actor.to(self._device) for actor in actors]
        
        self._critics : Sequence[CriticNetwork] = [critic.to(self._device)for critic in critics]

        self.policy_memory_buffer = PolicyMemoryBuffer(
                observation_shape=self._game_vec._single_obs_space,
                n_actions=self._n_game_actions,
                n_workers=len(game_fns),
                worker_steps=self._step_size,
                n_game_players=self._n_players)
        
        self.aux_memory_buffer = AuxMemoryBuffer(
            observation_shape=self._game_vec._single_obs_space,
            n_actions=self._n_game_actions,
            n_workers=len(game_fns),
            worker_steps=self._step_size,
            n_policy_iterations=self._n_policy_iterations
        )

        self._gamma = gamma
        self._gae_lam = gae_lam
        self._normalize_rewards = normalize_rewards
        self._max_reward_norm = max_reward_norm
        self.rewards_abs_moving_average = 1

        self._testing_game_fn = testing_game_fn
        self._testing_intervals = testing_intervals
        self._log_intervals = log_intervals
        self.log_dict = {Phases.GLOBAL:dict(),Phases.POLICY_PHASE:dict(),Phases.AUX_PHASE:dict(),Phases.TESTING_PHASE:dict()}
        self._save_name = save_name
    
    def run(self):
        t_start = time.perf_counter()
        
        total_steps = 0
        next_testing_phase = 50_000
        next_log_phase = 10_000
        previous_actors = copy.deepcopy(self._actors)
        partial_obs , full_obs, legal_actions_masks,players= self._game_vec.reset()
        while total_steps < self._max_steps:
            n_steps , partial_obs,full_obs,legal_actions_masks,players= self._run_policy_phase(
                partial_obs , full_obs, legal_actions_masks,players)
            total_steps +=n_steps

            self.log_dict[Phases.GLOBAL]["Current Step"] = f"{total_steps} of {self._max_steps}"
            self.log_dict[Phases.GLOBAL]["Completed"] = f"{total_steps*100/ self._max_steps:0.3} %"
            self.log_dict[Phases.GLOBAL]["Total Duration"] = f"{time.perf_counter() - t_start}"
            self.log_dict[Phases.GLOBAL]["FPS"] = f"{total_steps// (time.perf_counter() - t_start)}"

            self._run_aux_phase()
            next_testing_phase = self._run_testing_phase(total_steps,next_testing_phase,previous_actors)
            self._run_log_phase(total_steps,next_log_phase)

            for player,(actor,critic) in enumerate(zip(self._actors,self._critics)):
                actor.save_model(os.path.join("tmp",f"{self._save_name}_{player}_actor.pt"))
                critic.save_model(os.path.join("tmp",f"{self._save_name}_{player}_critic.pt"))

            
    
    def _run_policy_phase(self,partial_obs:np.ndarray,full_obs:np.ndarray,legal_actions_masks:np.ndarray,players:np.ndarray):
        steps = 0
        for policy_phase in range(self._n_policy_iterations):
            n_steps , partial_obs,full_obs,legal_actions_masks,players = self._collect_data(self._game_vec,self._step_size,
                        partial_observations=partial_obs,full_observations=full_obs,
                        legal_action_masks=legal_actions_masks,players=players)
            steps += n_steps

            (all_partial_obs , all_full_obs , all_actions ,
            all_legal_action_masks,all_log_probs,all_rewards ,
            all_terminals ,all_players)= self.policy_memory_buffer.sample()
            all_values = np.zeros_like(all_log_probs)
            for player in range(self._n_players):
                predicate = all_players == player
                # player_idx = np.argwhere(predicate)
                player_full_obs = all_full_obs[predicate]
                critic = self._critics[player]
                player_full_obs_t = T.tensor(player_full_obs,dtype=T.float32,device=self._device)
                with T.no_grad():
                    player_values_t:T.Tensor = critic.evaluate(player_full_obs_t)
                player_values_ar = player_values_t.squeeze().cpu().numpy()
                all_values[predicate] = player_values_ar

            last_values = np.zeros((len(players),),dtype=np.float32)
            for player in range(self._n_players):
                predicate = players == player
                player_last_full_obs = full_obs[predicate]
                critic = self._critics[player]
                player_last_full_obs_t = T.tensor(player_last_full_obs,device=self._device)
                with T.no_grad():
                    player_last_values_t:T.Tensor = critic.evaluate(player_last_full_obs_t)
                
                player_last_values_ar :np.ndarray= player_last_values_t.squeeze().cpu().numpy()
                last_values[predicate] = player_last_values_ar
            
            all_returns , all_adv = self._calculate_advantages(all_rewards,all_values,all_terminals,all_players,last_values,players)

            n_examples = self._n_workers * self._step_size
            all_rewards = all_rewards.reshape((n_examples,self._n_players))
            all_values = all_values.reshape((n_examples,))
            all_partial_obs = all_partial_obs.reshape((n_examples,*self._observations_shape))
            all_full_obs = all_full_obs.reshape((n_examples,*self._observations_shape))
            all_legal_action_masks = all_legal_action_masks.reshape((n_examples,self._n_game_actions))
            all_actions = all_actions.reshape((n_examples,))
            all_returns = all_returns.reshape((n_examples,))
            all_players = all_players.reshape((n_examples,))
            all_adv = all_adv.reshape((n_examples,))
            all_log_probs = all_log_probs.reshape((n_examples,))

            self.aux_memory_buffer.append_aux_data(
                                        partial_observations=all_partial_obs,
                                        legal_actions_masks=all_legal_action_masks,
                                        full_observations=all_full_obs,
                                        returns=all_returns,
                                        players=all_players)
            
            log_dict = self.log_dict[Phases.POLICY_PHASE]
            for player in range(self._n_players):
                predicate = all_players == player
                explained_variance = self._calculate_explained_vartiance(all_values[predicate],all_returns[predicate])
                log_dict[f"Explained Variance Player {player}"] = f"{explained_variance:0.3}"

            for epoch in range(self._n_policy_epochs):
                self.policy_trainer.train_policy(all_partial_obs,all_full_obs,all_actions,all_legal_action_masks,all_log_probs,all_adv,all_players,is_aux_training=False,log_dict = log_dict)
            for epoch in range(self._n_value_epochs):
                self.policy_trainer.train_critic(all_full_obs,all_returns,all_players,is_aux_training=False,log_dict=log_dict)

            # self.aux_trainer.train_aux(partial_obs,all_legal_action_masks,,all_players)
            self.policy_memory_buffer.reset()
        return steps , partial_obs,full_obs,legal_actions_masks,players
    
    def _run_aux_phase(self):
        partial_obs, legal_actions_masks,full_obs,returns,players = self.aux_memory_buffer.sample()
        distributions = np.zeros((len(players),self._n_game_actions),dtype=np.float32)
        for player,actor in enumerate(self._actors):
            predicate = players == player
            player_partial_obs = T.tensor(partial_obs[predicate],dtype=T.float32,device=self._device)
            with T.no_grad():
                probs , _ = actor.predict(player_partial_obs)
            probs_ar =  probs.cpu().numpy()
            distributions[predicate] = probs_ar

        log_dict = self.log_dict[Phases.AUX_PHASE]
        for aux_phase in range(self._n_aux_iterations):
            self.policy_trainer.train_critic(full_obs,returns,players,is_aux_training=True,log_dict=log_dict)
            self.aux_trainer.train_aux(partial_obs,legal_actions_masks,distributions,players,log_dict=log_dict)
        
        self.aux_memory_buffer.reset_aux_data()

    def _run_testing_phase(self,current_steps:int,next_testing_phase:int,previous_actors:Sequence[ActorNetwork]):
        log_dict = self.log_dict[Phases.TESTING_PHASE]
        if current_steps < next_testing_phase:
            log_dict.clear()
            return next_testing_phase
        
        assert len(previous_actors) == len(self._actors)
        win_percent = self._pit(tuple(self._actors),None,100)

        log_dict["win ratio vs random"] = f"{win_percent:0.3f}"
        win_percent = self._pit(tuple(self._actors),tuple(previous_actors),100)
        log_dict["win ratio vs previous"] = f"{win_percent:0.3f}"

        for actor ,previous_actor in zip(self._actors,previous_actors):
            previous_actor.load_state_dict(actor.state_dict())
        next_testing_phase = current_steps + self._testing_intervals
        return next_testing_phase

    def _run_log_phase(self,current_steps:int,next_log:int):
        if current_steps < next_log:
            return next_log
        
        print("\n")
        print("*" * 60)
        for phase_n in self.log_dict:
            for title in self.log_dict[phase_n]:
                print(f"{phase_n.name}::{title}:: {self.log_dict[phase_n][title]}")
        print("*" * 60)
        print("\n")

        next_log = current_steps + self._log_intervals
        return next_log

    def _pit(self,net_1:ActorNetwork|tuple[ActorNetwork]|None,net_2:ActorNetwork|tuple[ActorNetwork]|None,n_sets:int)->float|None:
        if self._testing_game_fn is None:
            return None
        nets = [net_1,net_2]
        players = []
        for net in nets:
            if net is None:
                player = RandomPlayer()
            elif isinstance(net,tuple):
                player = TurnBasedNNPlayer(list(net))
            else:
                player = NNPlayer(net)
            players.append(player)
        
        m = Match(self._testing_game_fn,players[0],players[1],n_sets)
        scores = m.start()
        return (scores[0] * 2  + scores[1]) / (scores.sum()*2)
    
    def _collect_data(
            self,
            vec_game:VecGame,
            step_size:int,
            partial_observations:np.ndarray,
            full_observations:np.ndarray,
            legal_action_masks:np.ndarray,
            players:np.ndarray
            ):
        n_steps = 0
        for _ in range(step_size):
            actions , log_probs = self._choose_actions(partial_observations,legal_action_masks,players)

            n_steps+= len(actions)

            new_partial_obs,new_full_obs,new_legal_actions_masks,rewards,dones,new_players = vec_game.step(actions)
            if self._normalize_rewards:
                abs_rewards = np.abs(rewards).sum()
                g = 0.001
                self.rewards_abs_moving_average = float((1-g) * self.rewards_abs_moving_average + g* abs_rewards + 1e-6)
                rewards /= self.rewards_abs_moving_average
                if self._max_reward_norm:
                    rewards = rewards.clip(-self._max_reward_norm,self._max_reward_norm)
            
            self.policy_memory_buffer.save(
                step_partial_observations=partial_observations,
                step_full_observations=full_observations,
                step_actions=actions,
                step_legal_actions_masks =legal_action_masks,
                step_log_probs=log_probs,
                step_rewards = rewards,
                step_terminals = dones,
                step_players = players)

            partial_observations = new_partial_obs
            full_observations = new_full_obs
            legal_action_masks = new_legal_actions_masks
            players = new_players
        
        return n_steps, partial_observations,full_observations,legal_action_masks,players
    
    def _choose_actions(self,observations:np.ndarray,legal_action_masks:np.ndarray,players:np.ndarray):

        actions = np.zeros((len(observations),),dtype=np.int32)
        log_probs = np.zeros((len(observations),),dtype=np.float32)
        idxs = np.array([idx for idx in range(len(observations))])
        observations_t = T.tensor(observations,dtype=T.float32,device=self._device)
        legal_actions_masks_t = T.tensor(legal_action_masks,dtype=T.int32,device=self._device)
        for player in range(self._n_players):
            predicate:np.ndarray = players == player
            player_obs:T.Tensor = observations_t[predicate]
            player_idxs:np.ndarray = idxs[predicate]
            player_action_masks = legal_actions_masks_t[predicate]

            with T.no_grad():
                probs , _ = self._actors[player].predict(player_obs)
            
            legal_probs:T.Tensor = probs * player_action_masks
            legal_probs /= legal_probs.sum(dim=-1,keepdim=True)
            legal_probs[(legal_probs<0.003).logical_and(legal_probs>0)] = EPS
            legal_probs_1 =  legal_probs/ legal_probs.sum(dim=-1,keepdim=True)
            legal_actions_dist = Categorical(legal_probs_1)
            player_actions_t = legal_actions_dist.sample()
            log_probs_t:T.Tensor = legal_actions_dist.log_prob(player_actions_t)
            
            player_actions = player_actions_t.cpu().numpy()
            player_log_probs = log_probs_t.cpu().numpy()
            assert not np.any(player_log_probs==0)    
            actions[predicate] = player_actions
            log_probs[predicate] = player_log_probs

        assert not np.any(log_probs==0)
        return actions,log_probs

    def _calculate_advantages(self,rewards:np.ndarray,values:np.ndarray,terminals:np.ndarray,
                              players:np.ndarray,last_values:np.ndarray,last_players:np.ndarray):
        

        adv_arr = np.zeros((self._n_workers,self._step_size),dtype=np.float32)

        l_values = np.zeros((self._n_workers,self._n_players),dtype=np.float32)
        l_values[np.arange(len(last_players)),last_players] = last_values
        l_values[np.arange(len(last_players)),1-last_players] = -last_values

        for player in range(self._n_players):
            player_adv = np.zeros((self._n_workers,self._step_size),dtype=np.float32)
            for worker in range(self._n_workers):
                next_val = l_values[worker][player]
                next_adv = 0
                agg_rewards = 0
                for step in reversed(range(self._step_size)):
                    if players[worker][step] == player:
                        current_reward = agg_rewards +  rewards[worker][step][player]
                        agg_rewards = 0
                        current_val = values[worker][step]
                        is_terminal = int(terminals[worker][step])
                        delta = current_reward + (self._gamma * next_val * (1-is_terminal)) - current_val
                        next_ = self._gamma * self._gae_lam * next_adv * (1-is_terminal)
                        current_adv = delta + next_
                        next_adv = current_adv
                        next_val = current_val
                        player_adv[worker][step] = current_adv
                    else:
                        is_terminal = int(terminals[worker][step])
                        if is_terminal:
                            agg_rewards = rewards[worker][step][player]
                            next_val = 0
                            next_adv = 0
                        else:
                            agg_rewards += rewards[worker][step][player]
                            next_val = next_val
                            next_adv = next_adv
            predicates = players == player
            adv_arr[predicates] = player_adv[predicates]
        
        returns = adv_arr + values
        return returns , adv_arr
    
    @staticmethod
    def _calculate_explained_vartiance(predictions:np.ndarray,target:np.ndarray)->float:
        assert predictions.ndim == 1 and target.ndim ==1
        target_var = target.var() + EPS
        unexplained_var_ratio = (target-predictions).var()/ target_var
        explained_var_ratio = 1- unexplained_var_ratio
        return explained_var_ratio
                        
                        
class Phases(Enum):
    GLOBAL=1,
    POLICY_PHASE=2,
    AUX_PHASE=3,
    TESTING_PHASE=4,
    
