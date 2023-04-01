import numpy as np
import torch as T
from typing import Any, Callable, Sequence, Tuple
from rnad.games.game import Game
from rnad.games.state import State
from rnad.match import Match
from rnad.networks import PytorchNetwork, RnadNetwork
from collections import deque
from dataclasses import dataclass
from torch.nn.utils.clip_grad import clip_grad_value_
import copy
import math

from rnad.players import NNPlayer, RandomPlayer


class Rnad():
    def __init__(self, game_fn: Callable[[], Game]) -> None:

        self.reg_networks: PytorchNetwork
        self.prev_reg_networks: PytorchNetwork
        self.network: PytorchNetwork
        self.target_network: PytorchNetwork
        self.network, self.target_network, self.reg_networks, self.prev_reg_networks = self._initialize_networks(
            game_fn)
        self.optim = T.optim.Adam(self.network.parameters(), lr=5e-4)
        self._game_fn = game_fn
        self.learner_steps = 0
        self.actor_steps = 0
        self.n_actors = 8
        self._trajectory_max = 20
        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
        self.n_game_actions = self._game_fn().n_actions
        self.gamma_avg = 0.001
        self.delta_m = 200
        self.last_step = 0
        self.test_intervals = 10000
        self.last_test = 0

        self.network.to(self.device)
        self.target_network.to(self.device)
        self.reg_networks.to(self.device)
        self.prev_reg_networks.to(self.device)

    def _initialize_networks(self, game_fn: Callable[[], Game]):
        game = game_fn()
        if len(game.observation_space) < 2:
            network = RnadNetwork(game.observation_space, game.n_actions)
            target_network = copy.deepcopy(network)
            reg_network = copy.deepcopy(network)
            prev_preg_network = copy.deepcopy(network)
        else:
            network = RnadNetwork(game.observation_space, game.n_actions)
            target_network = copy.deepcopy(network)
            reg_network = copy.deepcopy(network)
            prev_preg_network = copy.deepcopy(network)
        return network, target_network, reg_network, prev_preg_network

    def run(self) -> None:
        while self.actor_steps < 1_000_000:
            self.step()

    def step(self) -> None:
        data = self._collect_data()
        alpha, update_network = self._entropy_schedule(self.learner_steps)
        all_observations: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_action_mask: list[np.ndarray] = []
        all_v_t: list[np.ndarray] = []
        all_q_t: list[np.ndarray] = []
        for episode_examples in data:
            observations, actions, action_mask, v_t, q_t = self._extract_training_examples(
                episode_examples, alpha)
            all_observations.append(observations)
            all_actions.append(actions)
            all_action_mask.append(action_mask)
            all_v_t.append(v_t)
            all_q_t.append(q_t)
        self._update_params(all_observations, all_actions,
                            all_action_mask, all_v_t, all_q_t)
        if update_network:
            self.prev_reg_networks.load_state_dict(
                self.reg_networks.state_dict())
            self.reg_networks.load_state_dict(self.target_network.state_dict())
            self.last_step = self.learner_steps
            print("Updating networks")
        self.learner_steps += 1

        if self.actor_steps - self.last_test > self.test_intervals:
            self.last_test = self.actor_steps
            # Test
            p1 = NNPlayer(self.network)
            p2 = RandomPlayer()
            m = Match(self._game_fn, player_1=p1, player_2=p2, n_sets=50)
            s = m.start()
            print(s)
            print(f"Actor Steps: {self.actor_steps}")

    def _entropy_schedule(self, learner_steps: int):
        delta = learner_steps - self.last_step
        alpha = delta / self.delta_m
        alpha = min(1, alpha)
        update_network = delta >= self.delta_m
        return alpha, update_network

    def _extract_training_examples(self, episode_examples: Sequence["Example"], alpha: float):
        observations: np.ndarray = np.array(
            [ex.state.to_player_obs() for ex in episode_examples], dtype=np.float32)
        players: np.ndarray = np.array(
            [ex.state.player_turn for ex in episode_examples], dtype=np.int32)
        actions: np.ndarray = np.array(
            [ex.action for ex in episode_examples], dtype=np.int32)
        actions_prob: np.ndarray = np.array(
            [ex.prob for ex in episode_examples], dtype=np.float32)
        rewards: np.ndarray = np.array(
            [ex.rewards for ex in episode_examples], dtype=np.float32)
        action_mask: np.ndarray = np.array(
            [ex.state.legal_actions_masks() for ex in episode_examples])
        states = [ex.state for ex in episode_examples]
        observations_t: T.Tensor = T.tensor(
            observations, dtype=T.float32, device=self.device)
        with T.no_grad():
            probs, v, log_probs, logit = self.network(observations_t)
            _, v_target, _, _ = self.target_network(observations_t)
            _, _, log_prob_reg, _ = self.reg_networks(observations_t)
            _, _, log_preb_prev_reg, _ = self.prev_reg_networks(observations_t)
        probs_ar: np.ndarray = probs.cpu().numpy()
        probs_ar = probs_ar * action_mask
        probs_ar /= probs_ar.sum(axis=-1, keepdims=True)
        # probs_ar = probs_ar[np.arange(len(probs_ar)),actions]
        log_prob_reg = log_probs - \
            (alpha*log_prob_reg + (1-alpha)*log_preb_prev_reg)
        probs_ar = np.array([probs_ar[i, action]
                                for i, action in enumerate(actions)])
        log_prob_reg_ar = log_prob_reg.cpu().numpy()
        log_prob_reg_ar = np.array([log_prob_reg_ar[i, action]
                                for i, action in enumerate(actions)])
        v_target = v_target.squeeze()
        v_target_ar = v_target.cpu().numpy()
        v_t, q_t = self._calculate_v_trace_(
            v_target_ar, states, probs_ar, actions_prob, log_prob_reg_ar, rewards, players)
        return observations, actions, action_mask, v_t, q_t

        ...

    def _collect_data(self) -> Sequence[Sequence['Example']]:
        examples: list[list[Example]] = [[] for _ in range(self.n_actors)]
        games = [self._game_fn() for _ in range(self.n_actors)]
        states = [game.reset() for game in games]
        ids = [id_ for id_ in range(self.n_actors)]
        episode_step = 0
        # histories = [deque(maxlen=20) for _ in range(self.n_actors)]
        while episode_step < self._trajectory_max and len(states) > 0:
            actions, policies = self._choose_actions(states)
            new_states, new_ids, rewards = self._apply_actions(
                states, ids, actions)
            id_: int
            state: State
            action: int
            action_h: np.ndarray
            prob: float
            reward: np.ndarray
            for state, id_, action, prob, reward in zip(states, ids, actions, policies, rewards):
                examples[id_].append(
                    Example(
                        state=state,
                        action=action,
                        action_h=np.empty((1,)),
                        prob=prob,
                        rewards=reward))
            states, ids = new_states, new_ids
        return examples

    def _choose_actions(self, states: Sequence[State]) -> Tuple[Sequence[int], Sequence[float]]:
        observations = np.stack([state.to_player_obs()
                                for state in states], axis=0)
        action_masks = np.stack([state.legal_actions_masks()
                                for state in states])
        observations_tensor = T.tensor(
            observations, dtype=T.float32, device=self.device)
        probs: T.Tensor
        with T.no_grad():
            probs, _, _, _ = self.network(observations_tensor)
        probs_ar: np.ndarray = probs.cpu().numpy()
        probs_ar = probs_ar * action_masks
        assert probs_ar.dtype == np.float32
        probs_ar /= probs_ar.sum(axis=-1, keepdims=True)

        actions = [np.random.choice(len(p), p=p) for p in probs_ar]
        # for hist, action in zip(histories, actions):
        #     hist.append(action)
        policies = [p[a] for p, a in zip(probs_ar, actions)]
        return actions, policies

    def _apply_actions(self, states: Sequence[State], ids: Sequence[int], actions: Sequence[int]):
        new_states: Sequence[State] = []
        rewards: list[np.ndarray] = []
        new_ids: list[int] = []
        for state, id_, action in zip(states, ids, actions):
            new_state = state.step(action)
            self.actor_steps += 1
            if new_state.is_terminal():
                reward = new_state.game_result().astype(np.float32)
            else:
                reward = np.zeros((2,), dtype=np.float32)
                new_states.append(new_state)
                new_ids.append(id_)
            rewards.append(reward)
        return new_states, new_ids, rewards

    def _update_params(self, observations: list[np.ndarray], actions: list[np.ndarray], action_masks: list[np.ndarray], v_t: list[np.ndarray], q_t: list[np.ndarray]):
        observations_ar = np.concatenate(observations, axis=0)
        actions_ar = np.concatenate(actions, axis=0)
        action_masks_ar = np.concatenate(action_masks, axis=0)
        v_t_ar = np.concatenate(v_t, axis=0)
        q_t_ar = np.concatenate(q_t, axis=0)

        # TODO CHECK
        one_hot_actions_ar = np.zeros(
            (len(actions_ar), self.n_game_actions), dtype=np.int32)
        one_hot_actions_ar[np.arange(len(actions_ar)), actions_ar] = 1
        observations_t = T.tensor(
            observations_ar, dtype=T.float32, device=self.device)
        actions_t = T.tensor(actions_ar, dtype=T.int32, device=self.device)
        action_masks_t = T.tensor(
            action_masks_ar, dtype=T.int32, device=self.device)
        one_hot_actions_t = T.tensor(
            one_hot_actions_ar, dtype=T.int32, device=self.device)
        v_t_t = T.tensor(v_t_ar, dtype=T.float32, device=self.device)
        q_t_t = T.tensor(q_t_ar, dtype=T.float32, device=self.device)
        # self.network(observations_t)
        batch_size = 32
        n_examples = len(observations_ar)
        n_batches = n_examples//batch_size + 1

        self.optim.zero_grad()
        for i in range(n_batches):
            batch_idx = np.random.choice(n_examples, batch_size, replace=True)
            np.random.shuffle(batch_idx)
            observations_batch = observations_t[batch_idx]
            actions_batch = actions_t[batch_idx]
            one_hot_actions_batch = one_hot_actions_t[batch_idx]
            action_mask_batch = action_masks_t[batch_idx]
            v_t_batch = v_t_t[batch_idx]
            q_t_batch = q_t_t[batch_idx]

            probs, v, _, logits = self.network(observations_batch)
            v: T.Tensor = v.squeeze()

            critic_loss: T.Tensor = ((v - v_t_batch)**2).mean()

            probs:T.Tensor = probs * action_mask_batch
            probs = probs/probs.sum(dim=-1,keepdim=True)
            dist = T.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions_batch)
            # q_t_batch = (q_t_batch - q_t_batch.mean() )/(1e-8 + q_t_batch.std())
            actor_loss = (-log_probs * q_t_batch).mean()
            # actor_loss: T.Tensor = (
            #     (-logits * one_hot_actions_batch).sum(dim=-1) * q_t_batch).mean()
            total_loss = (critic_loss + actor_loss)
            total_loss.backward()
        clip_grad_value_(self.network.parameters(), 10000)
        self.optim.step()
        with T.no_grad():
            for target_p, network_p in zip(self.target_network.parameters(), self.network.parameters()):
                target_p.copy_(self.gamma_avg * network_p +
                               (1-self.gamma_avg)*target_p)

    def _calculate_v_trace_(self, v_target: np.ndarray, states: Sequence[State], probs: np.ndarray, actor_probs: np.ndarray, log_prob_reg: np.ndarray,  rewards: np.ndarray, players: np.ndarray):
        c_ = 1
        p_ = 1
        # states: Sequence[State] = []
        # players: np.ndarray = np.array([])
        # log_prob_reg:np.ndarray = np.array([])
        values = v_target
        # values: np.ndarray = np.array([])
        n_players = 2
        # probs_theta: np.ndarray = np.array([])
        # probs_reg: np.ndarray = np.array([])
        v_t_i_hat = np.zeros((len(v_target)+1,n_players), dtype=np.float32)
        V_next_i = np.zeros((len(v_target)+1,n_players), dtype=np.float32)
        r_t_i_hat = np.zeros((len(v_target)+1,n_players), dtype=np.float32)
        eta_t_i = np.ones((len(v_target)+1,n_players), dtype=np.float32)
        # r_t_i = np.zeros((n_players, len(v_target)), dtype=np.float32)
        r_t_i = rewards
        Q_t_i_hat = np.zeros((len(v_target),n_players), dtype=np.float32)
        ga = 0.2
        # t_r = log_prob_reg* ga
        t_r = np.zeros_like(rewards)
        t_r[players==0,0] = ga*log_prob_reg[players==0]
        t_r[players==1,0] = -ga*log_prob_reg[players==1]
        t_r[players==0,1] = -ga*log_prob_reg[players==0]
        t_r[players==1,1] = ga*log_prob_reg[players==1]
        t_r = t_r + rewards
        rewards = t_r
        mu_inv = 1/probs
        for i in range(n_players):
            for t in reversed(range(len(v_target))):
                state = states[t]
                current_player = players[t]
                assert current_player == state.player_turn
                if state.player_turn != i:
                    ratio = actor_probs[t]/probs[t]
                    v_t_i_hat[t, i] = v_t_i_hat[t+1, i]
                    V_next_i[t, i] = V_next_i[t+1, i]
                    r_t_i_hat[t, i] = r_t_i[t, i] + ratio * r_t_i_hat[t+1, i]
                    eta_t_i[t, i] = ratio * eta_t_i[t+1, i]
                else:  # equals
                    ratio = actor_probs[t]/probs[t]
                    pt = min(p_, ratio * eta_t_i[t+1, i])
                    ct = min(c_, ratio * eta_t_i[t+1, i])
                    delta_v_i = pt * \
                        (r_t_i[t, i] + ratio * r_t_i_hat[t+1, i] +
                         V_next_i[t+1, i] - values[t])
                    v_t_i_hat[t, i] = values[t] + delta_v_i + \
                        ct * (v_t_i_hat[t+1, i] - V_next_i[t+1, i])
                    V_next_i[t, i] = values[t]
                    # TODO check
                    Q_t_i_hat[t, i] = (-ga*log_prob_reg[t] 
                        + values[t] 
                    + mu_inv[t] * 
                    (r_t_i[t, i] + ga*log_prob_reg[t] + ratio * (r_t_i_hat[t+1, i] + v_t_i_hat[t+1, i]) - values[t]))

        player_0_rounds = np.argwhere(players == 0)
        player_1_rounds = np.argwhere(players == 1)
        # v_0:np.ndarray = v_t_i_hat[player_0_rounds, 0]
        # v_1:np.ndarray = v_t_i_hat[player_1_rounds, 1]
        # q_0:np.ndarray = Q_t_i_hat[player_0_rounds, 0]
        # q_1:np.ndarray = Q_t_i_hat[player_1_rounds, 1]
        v_t = np.zeros((len(v_target),), dtype=np.float32)
        q_t = np.zeros((len(v_target),), dtype=np.float32)
        v_t[player_0_rounds] = v_t_i_hat[player_0_rounds, 0]
        v_t[player_1_rounds] = v_t_i_hat[player_1_rounds, 1]
        q_t[player_0_rounds] = Q_t_i_hat[player_0_rounds, 0]
        q_t[player_1_rounds] = Q_t_i_hat[player_1_rounds, 1]
        # return v_0, v_1, q_0, q_1
        return v_t, q_t


@dataclass
class Example():
    state: State
    action_h: np.ndarray
    prob: float
    rewards: np.ndarray
    action: int
