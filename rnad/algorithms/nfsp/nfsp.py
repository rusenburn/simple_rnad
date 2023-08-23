import torch
import os
import random
import time
import copy
import numpy as np
from typing import Callable
from rnad.algorithms.nfsp.circular_buffer import CircularBuffer
from rnad.algorithms.nfsp.reservoir_buffer import ReservoirBuffer
from rnad.games.game import Game
from rnad.games.state import State
from rnad.networks import ActorLinearNetwork, StateActionLinearNetwork, StateActionResNetwork, ActorResNetwork
from rnad.match import Match
from rnad.players import NNPlayer, RandomPlayer


class NFSP():
    def __init__(self,
                 game_fn: Callable[[], Game],
                 n_steps=1_000_000,
                 lr=2.5e-4,
                 batch_size=128,
                 update_intervals=128,
                 batches=2,
                 eta=0.1,
                 greedy_exploration=0.1,
                 target_update_interval=1000,
                 rl_buffer_size=30_000,
                 sl_buffer_size=100_000,
                 save_name=""
                 ) -> None:
        self.game_fn = game_fn
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.save_name = save_name
        self.greedy_exploration = greedy_exploration
        self.eta = eta
        game = game_fn()
        self.n_actions = game.n_actions
        self.observation_shape = game.observation_space
        self.rl_buffer = CircularBuffer(
            max_len=rl_buffer_size, observation_shape=game.observation_space, n_actions=self.n_actions)
        self.sl_buffer = ReservoirBuffer(
            max_len=sl_buffer_size, observation_shape=game.observation_space, n_actions=self.n_actions)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        if len(self.observation_shape)>2:
            self.policy = ActorResNetwork(
                shape=self.observation_shape, n_actions=self.n_actions).to(self.device)
            self.network = StateActionResNetwork(
            shape=self.observation_shape, n_actions=self.n_actions).to(self.device)
        else:
            self.policy = ActorLinearNetwork(
                shape=self.observation_shape, n_actions=self.n_actions).to(self.device)
            self.network = StateActionLinearNetwork(
            shape=self.observation_shape, n_actions=self.n_actions).to(self.device)

        self.target = copy.deepcopy(self.network).to(self.device)
        
        self.network_optim = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.policy_optim = torch.optim.Adam(
            self.policy.parameters(), lr=lr*0.5)
        self.update_interval = target_update_interval
        self.next_update = target_update_interval
        self.log_interval = 500
        self.next_log = self.log_interval
        self.current_step = 0
        self.training_intervals = update_intervals
        self.batches = batches
        self.next_training = self.training_intervals
        self.n_updates = 0

    def run(self):
        self.target.load_state_dict(self.network.state_dict())
        self.fill_buffers()
        while self.current_step < self.n_steps:
            self.step()

    def step(self):
        env = self.game_fn()
        state = env.reset()
        terminal = False
        t = 0
        while not terminal and t < 100:
            player = state.player_turn
            obs = state.to_player_obs()
            action, is_best_response = self.choose_action(state)
            new_state = state.step(action)
            t += 1
            new_player = new_state.player_turn
            terminal = new_state.is_terminal()
            if terminal:
                rewards = new_state.game_result()
                reward = rewards[player]
            else:
                reward = 0
            new_state_legal_actions = new_state.legal_actions_masks() if not terminal else np.zeros_like(state.legal_actions_masks())
            self.rl_buffer.save(obs, new_state.to_player_obs(
            ), action, reward, player, new_player, state.legal_actions_masks(), new_state_legal_actions, terminal)
            if is_best_response:
                self.sl_buffer.save(obs, action, state.legal_actions_masks())
            state = new_state
            if self.current_step+t > self.next_training:
                self.next_training += self.training_intervals
                policy_loss = None
                q_loss = None
                for _ in range(self.batches):
                    policy_loss = self.update_policy_network()
                    q_loss = self.update_network()
                # if policy_loss is not None:
                #     print(f"policy_loss {policy_loss}")
                # if q_loss is not None:
                #     print(f"q loss {q_loss}")
                if self.n_updates > self.next_update:
                    self.next_update += self.update_interval
                    self.target.load_state_dict(self.network.state_dict())
        self.current_step += t
        if self.current_step > self.next_log:
            self.next_log += self.log_interval
            m = Match(self.game_fn, NNPlayer(self.policy),
                      RandomPlayer(), n_sets=100)
            s = m.start()
            win_ratio = (s[0]*2 + s[1]) / (s.sum()*2)
            print(f"[INFO] Step {self.current_step} of {self.n_steps}")
            print(f"[INFO] Win Ratio : {win_ratio:0.3f} {s}")

            torch.save(self.policy.state_dict(), os.path.join(
                "tmp", f"{self.save_name}_nfsp.pt"))

    def choose_action(self, state: State) -> tuple[int, bool]:
        best_action = False
        legal_actions = state.legal_actions_masks()
        observation_t = torch.tensor(
            np.array([state.to_player_obs()]), dtype=torch.float32, device=self.device)
        greedy_or_avg, exploration_exploitation = np.random.rand(2)
        with torch.no_grad():
            if greedy_or_avg < self.eta:
                best_action = True
                if exploration_exploitation < self.greedy_exploration:
                    probs = np.ones((self.n_actions,), dtype=np.float32)
                    legal_probs = probs*legal_actions
                    legal_probs /= legal_probs.sum()
                    assert np.any(legal_probs > 0)
                    assert np.all(np.isnan(legal_probs) == False)
                    action = np.random.choice(self.n_actions, p=legal_probs)
                else:
                    qsa_t = self.network.evaluate(observation_t)
                    qsa = qsa_t.cpu().numpy()[0]
                    qsa -= (1-legal_actions)*640000
                    assert np.any(qsa > -64000)
                    action = int(np.argmax(qsa))
            else:
                probs = self.policy.predict(observation_t).cpu().numpy()[0]
                legal_probs = probs*legal_actions
                legal_probs /= legal_probs.sum()
                action = np.random.choice(self.n_actions, p=legal_probs)
        return action, best_action

    def update_policy_network(self):
        obs, actions, legal_actions = self.sl_buffer.sample(self.batch_size)
        if len(actions) < 10:
            return

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(
            actions, dtype=torch.int32, device=self.device)
        legal_actions_t = torch.tensor(
            legal_actions, dtype=torch.float32, device=self.device)
        probs = self.policy.predict(obs_t)
        probs = probs * legal_actions_t
        probs = probs/probs.sum(dim=-1, keepdim=True)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        import math
        can_increase = log_probs < math.log(0.99)
        can_increase = can_increase.detach().to(torch.float32)
        
        log_probs = log_probs *can_increase
        loss = -torch.mean(log_probs)

        self.policy_optim.zero_grad()
        loss.backward()
        self.policy_optim.step()
        return loss.detach().cpu().item()

    def update_network(self):
        obs, new_obs, actions, rewards, players, new_players, legal_actions, new_legal_actions, terminals = self.rl_buffer.sample(
            self.batch_size)
        one_hot_actions = np.zeros(
            (len(actions), self.n_actions), dtype=np.int32)
        one_hot_actions[np.arange(len(actions)), actions] = 1

        if len(actions) < 10:
            return

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        new_obs_t = torch.tensor(
            new_obs, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(
            actions, dtype=torch.int64, device=self.device)
        one_hot_actions_t = torch.tensor(
            one_hot_actions, dtype=torch.int, device=self.device
        )
        rewards_t = torch.tensor(
            rewards, dtype=torch.float32, device=self.device)
        players_t = torch.tensor(
            players, dtype=torch.int32, device=self.device)
        new_players_t = torch.tensor(
            new_players, dtype=torch.int32, device=self.device)
        legal_actions_t = torch.tensor(
            legal_actions, dtype=torch.int32, device=self.device)
        new_legal_actions_t = torch.tensor(
            new_legal_actions, dtype=torch.int32, device=self.device)
        terminals_t = torch.tensor(
            terminals, dtype=torch.bool, device=self.device)
        same_player = players_t == new_players_t

        qsa = self.network.evaluate(obs_t)
        qsa_ = self.target.evaluate(new_obs_t).detach()
        qsa_[new_legal_actions_t == False] = -float("inf")
        qsa_[terminals_t] = 0
        new_max_actions = torch.argmax(qsa_, dim=-1)
        max_new_qsa = qsa_[torch.arange(len(new_max_actions)), new_max_actions]
        max_new_qsa[same_player == False] = -max_new_qsa[same_player == False]
        chosen_actions_qsa = qsa[torch.arange(len(actions)), actions_t]
        loss = torch.mean((rewards_t + max_new_qsa - chosen_actions_qsa)**2)
        self.network_optim.zero_grad()
        loss.backward()
        self.network_optim.step()
        l = loss.detach().cpu().item()
        self.n_updates += 1
        assert l < 76320
        return l

    def fill_buffers(self):
        print("[INFO] Filling buffers")
        current_size = 0
        env = self.game_fn()
        state = env.reset()
        while current_size < 10000:
            player = state.player_turn
            obs = state.to_player_obs()
            legal_actions = state.legal_actions_masks()
            probs = np.ones((self.n_actions,), dtype=np.float32)
            legal_probs = probs*legal_actions
            legal_probs /= legal_probs.sum()
            assert np.any(legal_probs > 0)
            assert np.all(np.isnan(legal_probs) == False)
            action = np.random.choice(self.n_actions, p=legal_probs)
            new_state = env.step(action)
            new_player = new_state.player_turn
            terminal = new_state.is_terminal()
            if terminal:
                rewards = new_state.game_result()
                reward = rewards[player]
            else:
                reward = 0
            
            new_state_legal_actions = new_state.legal_actions_masks() if not terminal else np.zeros_like(state.legal_actions_masks())
            self.rl_buffer.save(obs, new_state.to_player_obs(
            ), action, reward, player, new_player, state.legal_actions_masks(), new_state_legal_actions, terminal)
            # self.sl_buffer.save(obs, action, state.legal_actions_masks())
            current_size+=1
            state = new_state
            if terminal:
                state = env.reset()
