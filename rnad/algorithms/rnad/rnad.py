import time
import os
import enum
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
    def __init__(self,
                 game_fn: Callable[[], Game],
                 #
                 n_steps=1000_000,
                 # reward_transformation
                 delta_m=75_000,
                 eta=0.2,
                 eta_end=None,
                 # target network
                 gamma_avg=0.001,
                 # data collecting
                 trajectory_max=50,
                 n_episodes=128,
                 n_replays=4,
                 # training
                 lr=2.5e-4,
                 n_batches=4,
                 n_epochs=4,
                 grad_clipping=1_000,
                 decay_lr=False,
                 # neurd
                 neurd_clip=1_000,
                 neurd_beta=2,
                 # v_trace
                 v_trace_p=1.0,
                 v_trace_c=1.0,
                 # testing
                 test_intervals=20000,
                 save_name="",
                 ) -> None:

        # general
        self.n_steps_min = n_steps
        self._game_fn = game_fn
        # reward_transformation
        self.delta_m = delta_m
        self.eta_initial = eta
        self.eta_end = eta if eta_end is None else eta_end
        # target network
        self.gamma_avg = gamma_avg
        # data collection phase
        self.n_actors = n_episodes
        self._trajectory_max = trajectory_max
        self.n_replays = n_replays
        # training
        self.lr = lr
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.grad_clipping = grad_clipping

        # neurd
        self.neurd_clip = neurd_clip
        self.neurd_beta = neurd_beta

        # v_trace
        self.v_trace_p = v_trace_p
        self.v_trace_c = v_trace_c

        # testing
        self.test_intervals = test_intervals

        # appendix
        self.network, self.target_network, self.reg_networks, self.prev_reg_networks = self._initialize_networks(
            game_fn)
        self.optim = T.optim.Adam(self.network.parameters(), lr=self.lr)
        # self.optim_scheduler = T.optim.lr_scheduler.LinearLR(,)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        print(f"Device in using is {self.device}")
        self.n_game_actions = self._game_fn().n_actions
        self.log_dict = {Logs.PolicyLoss: [],
                         Logs.CriticLoss: [], Logs.ExplainedVariance: []}
        self.replay_buffer: deque[Sequence[Example]] = deque(
            maxlen=self.n_replays * self.n_actors)
        self.save_name = save_name

        self.network.to(self.device)
        self.target_network.to(self.device)
        self.reg_networks.to(self.device)
        self.prev_reg_networks.to(self.device)

        self.learner_steps = 0
        self.actor_steps = 0
        self.last_step = 0
        self.m = 0
        self.next_test = self.test_intervals
        self.t_start = time.perf_counter()

    def _initialize_networks(self, game_fn: Callable[[], Game]):
        game = game_fn()
        if len(game.observation_space) < 2:
            network = RnadNetwork(game.observation_space, game.n_actions)
            target_network = copy.deepcopy(network)
            reg_network = copy.deepcopy(network)
            prev_preg_network = copy.deepcopy(network)
        else:
            network = RnadNetwork(game.observation_space,
                                  game.n_actions, blocks=5)
            target_network = copy.deepcopy(network)
            reg_network = copy.deepcopy(network)
            prev_preg_network = copy.deepcopy(network)
        return network, target_network, reg_network, prev_preg_network

    def run(self) -> None:
        self.t_start = time.perf_counter()
        while self.actor_steps < self.n_steps_min:
            self.step()

    def step(self) -> None:
        data = self._collect_data()
        for episode_examples in data:
            self.replay_buffer.append(episode_examples)
        alpha, update_network = self._entropy_schedule(self.learner_steps)
        if update_network:
            self.prev_reg_networks.load_state_dict(
                self.reg_networks.state_dict())
            self.reg_networks.load_state_dict(self.target_network.state_dict())
            self.m += 1
            print(f"[INFO] Updating networks m:{self.m}")
        all_observations,all_actions,all_action_mask,all_v_t,all_adv_pi = self._extract_all_training_examples(self.replay_buffer,alpha)
        n_examples, = all_actions.shape
        actor_loss, critic_loss, explained_variance = self._update_params(all_observations, all_actions,
                                                                          all_action_mask, all_v_t, all_adv_pi)
        self.log_dict[Logs.PolicyLoss].append(actor_loss)
        self.log_dict[Logs.CriticLoss].append(critic_loss)
        self.log_dict[Logs.ExplainedVariance].append(explained_variance)
        self.learner_steps += 1
        if self.actor_steps > self.next_test:
            self.next_test = self.actor_steps + self.test_intervals
            # Test
            p1 = NNPlayer(self.network)
            p2 = RandomPlayer()
            m = Match(self._game_fn, player_1=p1, player_2=p2, n_sets=50)
            s = m.start()
            win_ratio = (s[0]*2 + s[1]) / (s.sum()*2)
            
            p1 = NNPlayer(self.reg_networks)
            m_reg = Match(self._game_fn, player_1=p1, player_2=p2, n_sets=50)
            s_reg = m_reg.start()
            win_ratio_reg = (s_reg[0]*2 + s_reg[1]) / (s_reg.sum()*2)
            T.save(self.network.state_dict(), os.path.join(
                "tmp", f"{self.save_name}_rnad.pt"))

            duration = time.perf_counter() - self.t_start
            fps = self.actor_steps // duration
            actor_loss_mean = np.mean(self.log_dict[Logs.PolicyLoss])
            critic_loss_mean = np.mean(self.log_dict[Logs.CriticLoss])
            explained_variance_mean = np.mean(
                self.log_dict[Logs.ExplainedVariance])
            self.log_dict[Logs.PolicyLoss].clear()
            self.log_dict[Logs.CriticLoss].clear()
            self.log_dict[Logs.ExplainedVariance].clear()
            print("*********************************************")
            print(f"[INFO] Policy Loss : {actor_loss_mean:0.3f}")
            print(f"[INFO] Critic Loss : {critic_loss_mean:0.3f}")
            print(
                f"[INFO] Explained Variance : {explained_variance_mean:0.3f}")
            print(f"[INFO] Actor Steps : {self.actor_steps} of {self.n_steps_min}")
            print(f"[INFO] Learner Steps : {self.learner_steps}")
            print(f"[INFO] FPS : {fps}")
            print(f"[INFO] Duration : {int(duration)} seconds")
            print(f"[INFO] Examples Count : {n_examples}")
            print(f"[INFO] Current m : {self.m}")
            print(f"[INFO] Alpha : {alpha:0.3f}")
            print(f"[INFO] Eta : {self.current_eta:0.3f}")
            print(f"[INFO] Win Ratio : {win_ratio:0.3f} {s}")
            print(f"[INFO] Win Ratio Reg : {win_ratio_reg:0.3f} {s_reg}")
            print("*********************************************\n")

    def _entropy_schedule(self, learner_steps: int):
        update_network = False
        while self.actor_steps- self.last_step > self.delta_m:
            update_network = True
            self.last_step = self.last_step + self.delta_m

        delta = self.actor_steps - self.last_step
        alpha = delta / self.delta_m
        alpha = min(1,alpha)
        return alpha, update_network

    def _extract_training_examples(self, episode_examples: Sequence["Example"], alpha: float):
        observations: np.ndarray = np.array(
            [ex.state.to_player_obs() for ex in episode_examples], dtype=np.float32)
        players: np.ndarray = np.array(
            [ex.state.player_turn for ex in episode_examples], dtype=np.int32)
        actions: np.ndarray = np.array(
            [ex.action for ex in episode_examples], dtype=np.int32)
        actor_probs: np.ndarray = np.array(
            [ex.probs.copy() for ex in episode_examples], dtype=np.float32)
        rewards: np.ndarray = np.array(
            [ex.rewards for ex in episode_examples], dtype=np.float32)
        action_mask: np.ndarray = np.array(
            [ex.state.legal_actions_masks() for ex in episode_examples])
        states = [ex.state for ex in episode_examples]
        observations_t: T.Tensor = T.tensor(
            observations, dtype=T.float32, device=self.device)
        with T.no_grad():
            probs, v, _, logit = self.network(observations_t)
            probs_target, v_target, _, _ = self.target_network(observations_t)
            prob_reg, _, _, _ = self.reg_networks(observations_t)
            prob_prev_reg, _, _, _ = self.prev_reg_networks(observations_t)
        probs_ar: np.ndarray = probs.cpu().numpy()
        prob_reg_ar: np.ndarray = prob_reg.cpu().numpy()
        prob_reg_prev_reg_ar: np.ndarray = prob_prev_reg.cpu().numpy()
        probs_target_ar = probs_target.cpu().numpy()

        actor_probs, _ = self._to_legal_actions_probs(actor_probs, action_mask)
        probs_ar, log_probs = self._to_legal_actions_probs(
            probs_ar, action_mask)
        probs_target_ar, log_probs = self._to_legal_actions_probs(
            probs_target_ar, action_mask)
        prob_reg_ar, log_prob_reg = self._to_legal_actions_probs(
            prob_reg_ar, action_mask)
        prob_reg_prev_reg_ar, log_prob_prev_reg = self._to_legal_actions_probs(
            prob_reg_prev_reg_ar, action_mask)

        log_prob_reg_ar = log_probs - \
            (alpha*log_prob_reg + (1-alpha)*log_prob_prev_reg)

        v_target = v_target.squeeze()
        v_target_ar = v_target.cpu().numpy()
        v_ar = v.squeeze().cpu().numpy()
        # v_t, q_t = self._calculate_v_trace_2(
        #     v_target_ar, states, probs_ar, actor_probs, log_prob_reg_ar, rewards, actions,players)
        # v_t, q_t = self._calculate_v_trace_(
        #     v_target_ar, states, probs_ar, actor_probs, log_prob_reg_ar, rewards, actions,players)
        v_t, q_t = self._calculate_v_trace_(
            v_ar, states, probs_ar, actor_probs, log_prob_reg_ar, rewards, actions, players)
        # v_t, q_t = self._calculate_v_trace_2(
        #     v_ar, states, probs_ar, actor_probs, log_prob_reg_ar, rewards, actions,players)

        adv_pi = self._calculate_adv_pi(q_t, probs_ar, action_mask)
        return observations, actions, action_mask, v_t, adv_pi

    def _extract_all_training_examples(self,all_examples:Sequence[Sequence['Example']],alpha:float):
        observations = np.array([ex.state.to_player_obs() for exs in all_examples  for ex in exs])
        players: np.ndarray = np.array(
            [ex.state.player_turn for episode_examples in all_examples for ex in episode_examples], dtype=np.int32)
        actions: np.ndarray = np.array(
            [ex.action for episode_examples in all_examples for ex in episode_examples], dtype=np.int32)
        actor_probs: np.ndarray = np.array(
            [ex.probs.copy() for episode_examples in all_examples for ex in episode_examples], dtype=np.float32)
        rewards: np.ndarray = np.array(
            [ex.rewards for episode_examples in all_examples for ex in episode_examples], dtype=np.float32)
        action_mask: np.ndarray = np.array(
            [ex.state.legal_actions_masks() for episode_examples in all_examples for ex in episode_examples])
        states = [ex.state for episode_examples in all_examples for ex in episode_examples]
        terminals = np.array([ex.terminal for episode_examples in all_examples for ex in episode_examples],dtype=np.bool8)

        observations_t : T.Tensor = T.tensor(observations,dtype=T.float32,device=self.device)

        with T.no_grad():
            probs, v, _, logit = self.network(observations_t)
            probs_target, v_target, _, _ = self.target_network(observations_t)
            prob_reg, _, _, _ = self.reg_networks(observations_t)
            prob_prev_reg, _, _, _ = self.prev_reg_networks(observations_t)

        probs_ar: np.ndarray = probs.cpu().numpy()
        prob_reg_ar: np.ndarray = prob_reg.cpu().numpy()
        prob_reg_prev_reg_ar: np.ndarray = prob_prev_reg.cpu().numpy()
        probs_target_ar = probs_target.cpu().numpy()

        actor_probs, _ = self._to_legal_actions_probs(actor_probs, action_mask)
        probs_ar, log_probs = self._to_legal_actions_probs(
            probs_ar, action_mask)
        probs_target_ar, log_probs = self._to_legal_actions_probs(
            probs_target_ar, action_mask)
        prob_reg_ar, log_prob_reg = self._to_legal_actions_probs(
            prob_reg_ar, action_mask)
        prob_reg_prev_reg_ar, log_prob_prev_reg = self._to_legal_actions_probs(
            prob_reg_prev_reg_ar, action_mask)  

        log_prob_reg_ar = log_probs - \
            (alpha*log_prob_reg + (1-alpha)*log_prob_prev_reg)

        v_target = v_target.squeeze()
        v_target_ar = v_target.cpu().numpy()
        v_ar = v.squeeze().cpu().numpy()

        v_t, q_t = self._calculate_v_trace_all(
            v_target_ar, states, probs_ar, actor_probs, log_prob_reg_ar, rewards, actions, players,terminals)
        
        adv_pi = self._calculate_adv_pi(q_t, probs_ar, action_mask)
        return observations, actions, action_mask, v_t, adv_pi
        
    def _collect_data(self) -> Sequence[Sequence['Example']]:
        examples: list[list[Example]] = [[] for _ in range(self.n_actors)]
        games = [self._game_fn() for _ in range(self.n_actors)]
        states = [game.reset() for game in games]
        ids = [id_ for id_ in range(self.n_actors)]
        episode_step = 0
        # histories = [deque(maxlen=20) for _ in range(self.n_actors)]
        while episode_step < self._trajectory_max and len(states) > 0:
            actions, policies = self._choose_actions(states)
            new_states, new_ids, rewards ,terminals= self._apply_actions(
                states, ids, actions)
            id_: int
            state: State
            action: int
            action_h: np.ndarray
            probs: np.ndarray
            reward: np.ndarray
            for state, id_, action, probs, reward , terminal in zip(states, ids, actions, policies, rewards ,terminals):
                examples[id_].append(
                    Example(
                        state=state,
                        action=action,
                        action_h=np.empty((1,)),
                        probs=probs,
                        rewards=reward,
                        terminal=terminal
                        ))
            states, ids = new_states, new_ids
        return examples

    def _choose_actions(self, states: Sequence[State]) -> Tuple[Sequence[int], np.ndarray]:
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
        probs_ar = (probs_ar * action_masks)
        assert probs_ar.dtype == np.float32 or probs_ar.dtype == np.float64
        probs_ar /= probs_ar.sum(axis=-1, keepdims=True)

        actions = [np.random.choice(len(p), p=p) for p in probs_ar]
        return actions, probs_ar

    def _apply_actions(self, states: Sequence[State], ids: Sequence[int], actions: Sequence[int]):
        new_states: Sequence[State] = []
        rewards: list[np.ndarray] = []
        new_ids: list[int] = []
        terminals:list[int]=[]
        for state, id_, action in zip(states, ids, actions):
            new_state = state.step(action)
            self.actor_steps += 1
            if new_state.is_terminal():
                reward = new_state.game_result().astype(np.float32)
                terminals.append(True)
            else:
                reward = np.zeros((2,), dtype=np.float32)
                new_states.append(new_state)
                new_ids.append(id_)
                terminals.append(False)
            rewards.append(reward)
        return new_states, new_ids, rewards , terminals

    def _update_params(self, observations: np.ndarray, actions: np.ndarray, action_masks: np.ndarray, v_t: np.ndarray, adv_pi: np.ndarray):
        
        observations_ar = observations
        actions_ar = actions
        action_masks_ar = action_masks
        v_t_ar = v_t
        adv_pi_ar = adv_pi

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
        adv_pi_t = T.tensor(adv_pi_ar, dtype=T.float32, device=self.device)

        n_examples = len(observations_ar)
        n_batches = self.n_batches
        batch_size = math.ceil(n_examples / n_batches)
        n_epochs = self.n_epochs
        explained_variances: list[T.Tensor] = []
        actor_losses: list[T.Tensor] = []
        critic_losses: list[T.Tensor] = []
        for epoch in range(n_epochs):
            self.optim.zero_grad()
            t = []
            for i in range(n_batches):
                batch_idx = np.random.choice(
                    n_examples, batch_size, replace=True)
                np.random.shuffle(batch_idx)
                observations_batch = observations_t[batch_idx]
                actions_batch = actions_t[batch_idx]
                one_hot_actions_batch = one_hot_actions_t[batch_idx]
                action_mask_batch = action_masks_t[batch_idx]
                v_t_batch = v_t_t[batch_idx]
                adv_pi_batch = adv_pi_t[batch_idx]

                probs, v, _, logits = self.network(observations_batch)
                v: T.Tensor = v.squeeze()

                critic_loss: T.Tensor = ((v - v_t_batch)**2).mean()
                with T.no_grad():
                    explained_variance = self._calculate_explained_variance(
                        v, v_t_batch)
                    explained_variances.append(explained_variance)
                probs: T.Tensor = probs * action_mask_batch
                probs = probs/probs.sum(dim=-1, keepdim=True)
                dist = T.distributions.Categorical(probs)
                played_logits = logits[:]
                with T.no_grad():
                    thresh_center = T.zeros_like(played_logits)
                    can_decrease = played_logits - thresh_center > -self.neurd_beta
                    can_increase = played_logits - thresh_center < self.neurd_beta
                    negative_force = adv_pi_batch.clamp(max=0)
                    positive_force = adv_pi_batch.clamp(min=0)
                    clipped_force = can_decrease*negative_force + can_increase * positive_force

                weighted_logits: T.Tensor = clipped_force.detach() * played_logits * \
                    action_mask_batch
                actor_loss = -weighted_logits.sum(dim=-1).mean()

                total_loss = (critic_loss + actor_loss)
                t.append(total_loss)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            t_loss = T.stack(t,dim=0).sum()
            t_loss.backward()
            clip_grad_value_(self.network.parameters(), self.grad_clipping)
            self.optim.step()

        with T.no_grad():
            for target_p, network_p in zip(self.target_network.parameters(), self.network.parameters()):
                target_p.data.copy_(self.gamma_avg * network_p.data +
                               (1-self.gamma_avg)*target_p.data)
        mean_actor_loss = T.mean(T.stack(actor_losses, dim=0))
        mean_critic_loss = T.mean(T.stack(critic_losses, dim=0))
        mean_explained_variances = T.mean(T.stack(explained_variances, dim=0))
        return mean_actor_loss.cpu().item(), mean_critic_loss.cpu().item(), mean_explained_variances.cpu().item()

    def _calculate_v_trace_(self, v_target: np.ndarray, states: Sequence[State], probs: np.ndarray, actor_probs: np.ndarray, log_prob_reg: np.ndarray,  rewards: np.ndarray, actions: np.ndarray, players: np.ndarray):
        c_ = self.v_trace_c
        p_ = self.v_trace_p
        assert actor_probs.ndim == 2
        n_examples, n_actions = actor_probs.shape
        assert len(v_target) == n_examples
        values = v_target
        n_players = 2
        v_t_i_hat = np.zeros((len(v_target)+1, n_players), dtype=np.float32)
        V_next_i = np.zeros((len(v_target)+1, n_players), dtype=np.float32)
        r_t_i_hat = np.zeros((len(v_target)+1, n_players), dtype=np.float32)
        eta_t_i = np.ones((len(v_target)+1, n_players), dtype=np.float32)
        Q_t_i_aaaa_hat = np.zeros(
            (len(v_target), n_players, n_actions), dtype=np.float32)

        eta = self.current_eta

        t_r = np.zeros_like(rewards)
        for t, (action, lg_p_reg, player) in enumerate(zip(actions, log_prob_reg, players)):
            if player == 0:
                t_r[t, 0] = -eta * lg_p_reg[action]
                t_r[t, 1] = eta * lg_p_reg[action]
            else:
                t_r[t, 0] = eta * lg_p_reg[action]
                t_r[t, 1] = -eta * lg_p_reg[action]
        t_r = t_r + rewards
        rewards = t_r
        r_t_i = rewards
        mu_inv = 1/probs
        for i in range(n_players):
            for t in reversed(range(len(v_target))):
                state = states[t]
                current_player = players[t]
                at = actions[t]
                assert current_player == state.player_turn
                if state.player_turn != i:
                    ratio = probs[t, at]/actor_probs[t, at]
                    v_t_i_hat[t, i] = v_t_i_hat[t+1, i]
                    V_next_i[t, i] = V_next_i[t+1, i]
                    r_t_i_hat[t, i] = r_t_i[t, i] + ratio * r_t_i_hat[t+1, i]
                    eta_t_i[t, i] = ratio * eta_t_i[t+1, i]
                else:  # equals
                    ratio = probs[t, at]/actor_probs[t, at]
                    pt = min(p_, ratio * eta_t_i[t+1, i])
                    ct = min(c_, ratio * eta_t_i[t+1, i])
                    delta_v_i = pt * \
                        (r_t_i[t, i] + ratio * r_t_i_hat[t+1, i] +
                         V_next_i[t+1, i] - values[t])
                    v_t_i_hat[t, i] = values[t] + delta_v_i + \
                        ct * (v_t_i_hat[t+1, i] - V_next_i[t+1, i])
                    V_next_i[t, i] = values[t]
                    extra = (mu_inv[t, at] *
                             (r_t_i[t, i] + eta*log_prob_reg[t, at] + ratio * (r_t_i_hat[t+1, i] + v_t_i_hat[t+1, i]) - values[t]))
                    for a in range(n_actions):
                        Q_t_i_aaaa_hat[t, i, a] = values[t] - \
                            eta*log_prob_reg[t, a]
                    Q_t_i_aaaa_hat[t, i, at] += extra

        player_0_rounds = np.argwhere(players == 0)
        player_1_rounds = np.argwhere(players == 1)

        v_t = np.zeros((len(v_target),), dtype=np.float32)
        q_t = np.zeros((len(v_target), n_actions), dtype=np.float32)
        v_t[player_0_rounds] = v_t_i_hat[player_0_rounds, 0]
        v_t[player_1_rounds] = v_t_i_hat[player_1_rounds, 1]
        q_t[player_0_rounds] = Q_t_i_aaaa_hat[player_0_rounds, 0]
        q_t[player_1_rounds] = Q_t_i_aaaa_hat[player_1_rounds, 1]

        return v_t, q_t

    def _calculate_v_trace_2(self,
                             v_target: np.ndarray,
                             states: Sequence[State],
                             probs: np.ndarray,
                             actor_probs: np.ndarray,
                             log_prob_reg: np.ndarray,
                             rewards: np.ndarray,
                             actions: np.ndarray,
                             players: np.ndarray):
        '''
        calculate values using gae lambda without v_trace , needs to be online
        '''
        n_exmaples, n_actions = actor_probs.shape
        eta = self.current_eta
        t_r = np.zeros_like(rewards)
        for t, (action, lg_p_reg, player) in enumerate(zip(actions, log_prob_reg, players)):
            if player == 0:
                t_r[t, 0] = -eta * lg_p_reg[action]
                t_r[t, 1] = eta * lg_p_reg[action]
            else:
                t_r[t, 0] = eta * lg_p_reg[action]
                t_r[t, 1] = -eta * lg_p_reg[action]
        t_r = t_r + rewards
        rewards = t_r

        adv_arr = np.zeros((n_exmaples,), dtype=np.float32)
        returns_arr = np.zeros((n_exmaples,), dtype=np.float32)
        q_t_a_arr = np.zeros((n_exmaples, n_actions), dtype=np.float32)
        gamma = 0.99
        gae_lam = 0.95

        for player in range(2):
            player_advs = np.zeros((n_exmaples,), dtype=np.float32)
            player_returns = np.zeros((n_exmaples,), dtype=np.float32)
            player_q_t_aaa = np.zeros(
                (n_exmaples, n_actions), dtype=np.float32)
            next_val = 0
            next_adv = 0
            agg_rewards = 0
            for step in reversed(range(n_exmaples)):
                current_player = players[step]
                if player == current_player:
                    performed_action = actions[step]
                    current_reward = agg_rewards + rewards[step][player]
                    agg_rewards = 0
                    current_value = v_target[step]
                    delta = current_reward + (gamma * next_val) - current_value
                    next_ = gamma * gae_lam * next_adv
                    current_adv = delta + next_
                    player_advs[step] = current_adv
                    player_returns[step] = current_adv + current_value
                    for a in range(n_actions):
                        player_q_t_aaa[step][a] = -eta * \
                            log_prob_reg[step][a] + current_value

                    player_q_t_aaa[step, performed_action] += 1/actor_probs[step, performed_action] * (
                        current_adv + eta * log_prob_reg[step][[performed_action]])
                    next_adv = current_adv
                    next_val = current_value
                else:
                    agg_rewards += rewards[step][player]
                    next_val = next_val
                    next_adv = next_adv
            predicates = players == player
            adv_arr[predicates] = player_advs[predicates]
            returns_arr[predicates] = player_returns[predicates]
            q_t_a_arr[predicates] = player_q_t_aaa[predicates]
        return returns_arr, q_t_a_arr
    
    def _calculate_v_trace_all(self, v_target: np.ndarray, states: Sequence[State], probs: np.ndarray, actor_probs: np.ndarray, log_prob_reg: np.ndarray,  rewards: np.ndarray, actions: np.ndarray, players: np.ndarray,terminals:np.ndarray):
        c_ = self.v_trace_c
        p_ = self.v_trace_p
        assert actor_probs.ndim == 2
        n_examples ,n_actions = actor_probs.shape
        assert len(v_target) == n_examples
        values = v_target
        n_players = 2

        v_t_i_hat = np.zeros((n_examples+1, n_players), dtype=np.float32)
        V_next_i = np.zeros((n_examples+1, n_players), dtype=np.float32)
        r_t_i_hat = np.zeros((n_examples+1, n_players), dtype=np.float32)
        eta_t_i = np.ones((n_examples+1, n_players), dtype=np.float32)
        Q_t_i_aaaa_hat = np.zeros(
            (n_examples, n_players, n_actions), dtype=np.float32)

        eta = self.current_eta

        t_r = np.zeros_like(rewards)

        # TODO preformance
        for t, (action, lg_p_reg, player) in enumerate(zip(actions, log_prob_reg, players)):
            if player == 0:
                t_r[t, 0] = -eta * lg_p_reg[action]
                t_r[t, 1] = eta * lg_p_reg[action]
            else:
                t_r[t, 0] = eta * lg_p_reg[action]
                t_r[t, 1] = -eta * lg_p_reg[action]

        t_r = t_r + rewards
        rewards = t_r
        r_t_i = rewards
        mu_inv = 1/probs
        for i in range(n_players):
            for t in reversed(range(n_examples)):
                state = states[t]
                current_player = players[t]
                at= actions[t]
                terminal = int(terminals[t])
                assert current_player == state.player_turn
                if state.player_turn != i:
                    ratio = probs[t,at]/actor_probs[t,at]
                    v_t_i_hat[t,i] = v_t_i_hat[t+1,i] * (1-terminal)
                    V_next_i[t,i] = V_next_i[t+1,i] * (1-terminal)
                    r_t_i_hat[t,i] = r_t_i[t,i] + ratio * r_t_i_hat[t+1,i] * (1-terminal)
                    eta_t_i[t,i] = ratio * eta_t_i[t+1,i] if not terminal else 1
                else: # current player
                    ratio = probs[t,at]/actor_probs[t,at]
                    pt = min(p_,ratio * eta_t_i[t+1,i] if not terminal else ratio)
                    ct = min(c_,ratio * eta_t_i[t+1,i] if not terminal else ratio)

                    delta_v_i = pt * (r_t_i[t,i] + ratio * r_t_i_hat[t+1,i] * (1-terminal) +
                                      V_next_i[t+1,i]*(1-terminal) - values[t])
                    v_t_i_hat[t,i] = values[t] + delta_v_i + ct*(v_t_i_hat[t+1,i] - V_next_i[t+1,i]) * (1-terminal)
                    V_next_i[t,i] = values[t]
                    extra = (mu_inv[t,at]* 
                             (r_t_i[t,i] + eta*log_prob_reg[t,at] + ratio*(r_t_i_hat[t+1,i]+v_t_i_hat[t+1,i])*(1-terminal) - values[t]))
                    for a in range(n_actions):
                        Q_t_i_aaaa_hat[t,i,a]=values[t] - eta*log_prob_reg[t,a]
                    Q_t_i_aaaa_hat[t,i,at] +=extra
        
        player_0_predicates = players == 0
        player_1_predicates = players == 1
        v_t = np.zeros((n_examples,), dtype=np.float32)
        q_t = np.zeros((n_examples, n_actions), dtype=np.float32)
        v_t[player_0_predicates] = v_t_i_hat[:n_examples][player_0_predicates, 0]
        v_t[player_1_predicates] = v_t_i_hat[:n_examples][player_1_predicates, 1]
        q_t[player_0_predicates] = Q_t_i_aaaa_hat[:n_examples][player_0_predicates, 0]
        q_t[player_1_predicates] = Q_t_i_aaaa_hat[:n_examples][player_1_predicates, 1]
        # TODO check
        return v_t, q_t
    @staticmethod
    def _to_legal_actions_probs(probs: np.ndarray, legal_actions_masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert probs.shape == legal_actions_masks.shape
        legal_probs: np.ndarray = probs * legal_actions_masks + 1e-8
        legal_probs /= legal_probs.sum(axis=-1, keepdims=True)
        return legal_probs, np.log(legal_probs)

    def _calculate_adv_pi(self, q_t_ar: np.ndarray, prob_ar: np.ndarray, is_c_ar: np.ndarray):
        adv_pi_ar = np.zeros_like(q_t_ar)
        for t, (q_t, prob, is_c) in enumerate(zip(q_t_ar, prob_ar, is_c_ar)):
            s = np.sum(q_t * prob, axis=-1, keepdims=True)
            adv_pi = q_t - s
            adv_pi = is_c * adv_pi
            adv_pi = np.clip(adv_pi, a_min=-self.neurd_clip,
                             a_max=self.neurd_clip)
            adv_pi_ar[t] = adv_pi.copy()
        return adv_pi_ar

    @staticmethod
    def _calculate_explained_variance(predictions: T.Tensor, target: T.Tensor) -> T.Tensor:
        assert predictions.ndim == 1 and target.ndim == 1
        target_var = target.var() + 1e-8
        unexplained_var_ratio = (target-predictions).var() / target_var
        explained_var_ratio = 1 - unexplained_var_ratio
        return explained_var_ratio
    
    @property
    def current_eta(self):
        eta = self.eta_initial * \
            (self.eta_end/self.eta_initial)**(self.actor_steps / self.n_steps_min)
        return eta


@dataclass
class Example():
    state: State
    action_h: np.ndarray
    probs: np.ndarray
    rewards: np.ndarray
    action: int
    terminal:bool


class Logs(enum.IntEnum):
    CriticLoss = 0,
    PolicyLoss = 1,
    ExplainedVariance = 2,
