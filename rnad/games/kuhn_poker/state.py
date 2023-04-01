import numpy as np
import collections
import random
from enum import IntEnum
from rnad.games.state import State


class CARDS(IntEnum):
    J = 0
    Q = 1
    K = 2


class ACTIONS(IntEnum):
    Fold = 0
    CHECK = 1
    CALL = 2
    RAISE = 3


N_ACTIONS = 4

OBSERVATION_SPACE = (4, 3)


# board consists of 3 lists , player 0 , player 1  the deck
# 0:J 1:Q 2:K

# OBSERVATION consists of two dimensional array (3X3) the first dimension represent a card J,Q,K
# and the second dimension represent the probability of its existance in the hands of current_player , opponent , deck

class KuhnPokerState(State):
    def __init__(self,
                 board: tuple[list[CARDS], list[CARDS], list[CARDS]],
                 money: tuple[int, int, int],
                 player_turn: int,
                 turn: int,
                 last_action: ACTIONS | None
                 ) -> None:
        super().__init__()
        self._board = board
        self._money = money
        self._player_turn = player_turn
        self._turn = turn
        self._last_action = last_action

        # cache
        self._cached_legal_actions_masks: None | np.ndarray = None
        self._cached_is_terminal: None | bool = None
        self._cached_game_result: None | np.ndarray = None

        self._cached_player_obs: None | np.ndarray = None
        self._cached_full_obs: None | np.ndarray = None

    @property
    def player_turn(self) -> int:
        return self._player_turn

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def observation_space(self) -> tuple:
        return OBSERVATION_SPACE

    @staticmethod
    def new_state() -> "KuhnPokerState":
        player = 0
        turn = 0
        deck = [CARDS.J, CARDS.Q, CARDS.K]
        random.shuffle(deck)
        board: tuple[list[CARDS], list[CARDS], list[CARDS]] = ([], [], deck)
        board[0].append(deck.pop())
        board[1].append(deck.pop())
        money = (-1, -1, 2)
        return KuhnPokerState(board=board, money=money, player_turn=player, turn=turn, last_action=None)

    def legal_actions_masks(self) -> np.ndarray:
        if self._cached_legal_actions_masks is not None:
            return self._cached_legal_actions_masks.copy()
        actions_masks = np.zeros((N_ACTIONS,), dtype=np.int32)

        if self._turn == 0:
            actions_masks[[ACTIONS.RAISE.value, ACTIONS.CHECK.value]] = 1
        elif self._turn == 1:
            if self._last_action == ACTIONS.CHECK:
                actions_masks[[ACTIONS.RAISE.value, ACTIONS.CHECK.value]] = 1
            elif self._last_action == ACTIONS.RAISE:
                actions_masks[[ACTIONS.CALL.value, ACTIONS.Fold.value]] = 1
            else:
                raise AssertionError("not reachable")
        elif self._turn == 2:
            assert self._last_action == ACTIONS.RAISE
            actions_masks[[ACTIONS.CALL.value, ACTIONS.Fold.value]] = 1

        self._cached_legal_actions_masks = actions_masks
        return self._cached_legal_actions_masks.copy()

    def step(self, action: int) -> 'KuhnPokerState':
        assert self.legal_actions_masks()[action] == 1
        if self._turn == 0:
            new_turn = 1
            new_player_turn = 1-self._player_turn
            if action == ACTIONS.CHECK:
                new_board = ([*self._board[0]],
                             [*self._board[1]], [*self._board[2]])
                new_money = (*self._money,)
                last_action = ACTIONS.CHECK
                return KuhnPokerState(board=new_board, money=new_money, player_turn=new_player_turn, turn=new_turn, last_action=last_action)
            elif action == ACTIONS.RAISE:
                new_board = ([*self._board[0]],
                             [*self._board[1]], [*self._board[2]])
                p0, p1, d = self._bet(self._player_turn, self._money)
                new_money = p0, p1, d
                last_action = ACTIONS.RAISE
                return KuhnPokerState(
                    board=new_board, money=new_money,
                    player_turn=new_player_turn,
                    turn=new_turn, last_action=last_action)
            else:
                raise AssertionError("Not Reachable")
        elif self._turn == 1:
            new_player_turn = 1-self._player_turn
            if action == ACTIONS.CHECK:
                assert self._last_action == ACTIONS.CHECK
                new_money = self._showdown(self._money)
                new_turn = -1
                new_board = ([*self._board[0]],
                             [*self._board[1]], [*self._board[2]])

                return KuhnPokerState(
                    board=new_board,
                    money=new_money,
                    player_turn=new_player_turn,
                    turn=new_turn,
                    last_action=ACTIONS.CHECK)
            elif action == ACTIONS.CALL:
                assert self._last_action == ACTIONS.RAISE
                new_money = self._bet(self._player_turn, self._money)
                new_money = self._showdown(new_money)
                new_turn = -1
                new_board = ([*self._board[0]],
                             [*self._board[1]], [*self._board[2]])
                return KuhnPokerState(
                    board=new_board,
                    money=new_money,
                    player_turn=new_player_turn,
                    turn=new_turn,
                    last_action=ACTIONS.CHECK)
            elif action == ACTIONS.Fold:
                assert self._last_action == ACTIONS.RAISE
                p0, p1, d = self._fold(self._player_turn, self._money)
                new_turn = -1
                new_board = ([*self._board[0]],
                             [*self._board[1]], [*self._board[2]])
                new_money = p0, p1, d
                return KuhnPokerState(
                    board=new_board,
                    money=new_money,
                    player_turn=new_player_turn,
                    turn=new_turn,
                    last_action=ACTIONS.Fold)
            elif action == ACTIONS.RAISE:
                assert self._last_action == ACTIONS.CHECK
                p0, p1, d = self._bet(self._player_turn, self._money)
                new_board = ([*self._board[0]],
                             [*self._board[1]], [*self._board[2]])
                new_money = p0, p1, d
                new_turn = self._turn+1
                return KuhnPokerState(
                    board=new_board,
                    money=new_money,
                    turn=new_turn,
                    player_turn=new_player_turn,
                    last_action=ACTIONS.RAISE)
        elif self._turn == 2:
            assert self._last_action == ACTIONS.RAISE
            new_player_turn = 1-self._player_turn
            if action == ACTIONS.CALL:
                new_money = self._bet(self._player_turn, self._money)
                new_money = self._showdown(new_money)
                new_board = ([*self._board[0]],
                             [*self._board[1]], [*self._board[2]])
                new_turn = -1
                return KuhnPokerState(
                    board=new_board,
                    money=new_money,
                    player_turn=new_player_turn,
                    turn=new_turn,
                    last_action=ACTIONS.CALL)
            elif action == ACTIONS.Fold:
                assert self._last_action == ACTIONS.RAISE
                p0, p1, d = self._fold(self._player_turn, self._money)
                new_turn = -1
                new_board = ([*self._board[0]],
                             [*self._board[1]], [*self._board[2]])
                new_money = p0, p1, d
                return KuhnPokerState(
                    board=new_board,
                    money=new_money,
                    player_turn=new_player_turn,
                    turn=new_turn,
                    last_action=ACTIONS.Fold)
            else:
                raise AssertionError("Not Reachable")
        raise AssertionError("NOT REACHABLE")

    def game_result(self) -> np.ndarray:
        assert self.is_terminal()
        p0, p1, d = self._money
        assert d == 0
        rewards = np.array([p0, p1], dtype=np.int32)
        return rewards

    def to_player_obs(self) -> np.ndarray:
        if self._cached_player_obs is not None:
            return self._cached_player_obs.copy()
        player = self._player_turn
        opponent = 1-player
        deck = 2
        player_card = self._board[player][0]
        other_cards = [self._board[opponent][0], self._board[deck][0]]
        player_obs = np.zeros(OBSERVATION_SPACE, dtype=np.float32)
        player_obs[player_card][0] = 1
        for card in other_cards:
            player_obs[card][1] = 1/2
            player_obs[card][2] = 1/2
        p0, p1, d = self._money
        player_obs[3, :] = d
        self._cached_player_obs = player_obs
        return self._cached_player_obs.copy()

    def to_full_obs(self) -> np.ndarray:
        if self._cached_full_obs is not None:
            return self._cached_full_obs.copy()
        player = self._player_turn
        opponent = 1-player
        deck = 2
        player_card = self._board[player][0]
        opponent_card = self._board[opponent][0]
        deck_card = self._board[deck][0]
        player_full_obs = np.zeros(OBSERVATION_SPACE, dtype=np.float32)
        player_full_obs[player_card][0] = 1
        player_full_obs[opponent_card][1] = 1
        player_full_obs[deck_card][2] = 1

        p0,p1,d = self._money
        player_full_obs[3,:] = d
        self._cached_full_obs = player_full_obs
        return self._cached_full_obs.copy()

    def to_full_short(self) -> tuple:
        raise NotImplementedError()

    def to_player_short(self) -> tuple:
        raise NotImplementedError()
    def is_terminal(self) -> bool:
        return self._turn < 0 or self._turn > 2

    def render(self, full: bool) -> None:
        raise NotImplementedError()

    def _showdown(self, money: tuple[int, int, int]):
        p0, p1, d = money
        p0_hand = self._board[0][0]
        p1_hand = self._board[1][0]
        if p0_hand > p1_hand:
            p0 += d
        elif p0_hand < p1_hand:
            p1 += d
        else:
            assert d % 2 == 0
            split = d//2
            p0 += split
            p1 += split
        d = 0
        return p0,p1,d

    def _bet(self, player: int, money: tuple[int, int, int]):
        p0, p1, d = money
        if player == 0:
            p0 -= 1
        else:
            p1 -= 1
        d += 1
        return p0, p1, d

    def _fold(self, player, money: tuple[int, int, int]):
        p0, p1, d = money
        if player == 0:
            p1 += d
        else:
            p0 += d
        d = 0
        return p0, p1, d
