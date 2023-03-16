import numpy as np
import torch as T
from typing import Any, Callable, Sequence, Tuple
from rnad.games.game import Game
from rnad.games.state import State
from rnad.networks import PytorchNetwork
from collections import deque


class Rnad():
    def __init__(self,game_fn:Callable[[],Game]) -> None:
        self.reg_networks : PytorchNetwork
        self.prev_reg_networks: PytorchNetwork
        self.network : PytorchNetwork
        self.target_network : PytorchNetwork
        self._game_fn = game_fn
        self.learner_steps = 0
        self.actor_steps = 0
        self.n_actors = 8
        self._trajectory_max = 20
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"

    
    # def step(self)->None:
    #     data = self._collect_data()
    #     alpha,update_network = self._entropy_schedule(self.learner_steps)
    #     self._update_params(data,alpha,self.learner_steps,update_network)
    #     self.learner_steps+=1


    def _collect_data(self)->Sequence[Any]:
        examples = [[] for _ in range(self.n_actors)]
        games = [self._game_fn() for _ in range(self.n_actors)]
        states = [game.reset() for game in games]
        ids = [id_ for id_ in range(self.n_actors)]
        episode_step = 0
        histories = [deque(maxlen=20) for _ in range(self.n_actors)]
        while episode_step < self._trajectory_max and len(states) > 0:
            actions  , policies , actions_h = self._choose_actions(states,histories)
            new_states,new_ids , rewards = self._apply_actions(states,ids,actions)
            for state,id_ in zip(states,ids):
                examples[id_].append((state,actions_h,policies,rewards))
            states , id_ =  new_states , new_ids
        return examples


    def _choose_actions(self,states:Sequence[State],histories:Sequence[deque[int]])->Tuple[Sequence[int],Sequence[float],Sequence[deque[int]]]:
        observations = np.stack([state.to_player_obs() for state in states],axis=0)
        action_masks = np.stack([state.legal_actions_masks() for state in states])
        observations_tensor = T.tensor(observations,dtype=T.float32,device=self.device)
        probs : T.Tensor
        with T.no_grad():
            probs ,_,_,_= self.network(observations_tensor)
        probs_ar:np.ndarray = probs.cpu().numpy()
        probs_ar = probs_ar * action_masks
        assert probs_ar.dtype == np.float32
        probs_ar /= probs_ar.sum(axis=-1,keepdims=True)

        actions = [np.random.choice(len(p),p=p) for p in probs_ar]
        for hist,action in zip(histories,actions):
            hist.append(action)
        policies = [p[a] for p,a in zip(probs_ar,actions)]
        return actions,policies,histories


    def _apply_actions(self,states:Sequence[State],ids:Sequence[int],actions:Sequence[int]):
        new_states = []
        rewards = []
        new_ids = []
        for state,id_,action in zip(states,ids,actions):
            new_state = state.step(action)
            if state.is_terminal():
                reward = state.game_result().astype(np.float32)
            else:
                reward = np.zeros((2,),dtype=np.float32)
                new_states.append(new_state)
                new_ids.append(id_)
            rewards.append(reward)
        
        return new_states,new_ids,rewards

    # def _update_params(self,data,alpha:float,learner_steps:int,update_network:bool):
    #     obs = data
    #     obs_t = obs 
    #     probs : T.Tensor
    #     probs,v, log_prob, logit =  self.network(obs_t)

    #     legal_actions = np.array([])
    #     probs = probs * legal_actions
    #     probs /= probs.sum(dim=-1,keepdim=True)

    #     _,v_target,_,_ = self.target_network(obs_t)
    #     _,_,log_prob_prev,_ = self.reg_networks(obs_t)
    #     _,_,log_prob_prev_,_ = self.prev_reg_networks(obs_t)

    #     log_prob_reg = log_prob - (alpha*log_prob_prev+ (1-alpha)*log_prob_prev_)
    #     v_target_list ,has_played_list , v_trace_prob_target_list = [],[],[]
    #     for player in range(2):
    #         reward = data.rewards[:,:,player]
    #         states = data.states
    #         v_target_,has_played,policy_target_ = v_trace(
    #             v_target,
    #             states,
    #             probs,
    #             log_prob_reg,
    #             reward,
    #             player)
    
    # def _calculate_v_trace():
    #     ...
