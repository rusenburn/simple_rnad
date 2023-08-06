import random
import copy
import numpy as np
from typing import TypeAlias
from enum import Enum, IntEnum
from rnad.games.state import State


class CARDS(IntEnum):
    J = 0
    Q = 1
    K = 2


class ACTIONS(IntEnum):
    CHECKORFOLD = 0
    CALL = 1
    RAISE = 2


OBSERVATION_SPACE = (5, 4)


Board: TypeAlias = tuple[list[CARDS], list[CARDS], list[CARDS], list[CARDS]]


class LeducPokerState(State):
    def __init__(self, board: Board, money: tuple[int, int, int], round_n_turn: tuple[int, int],
                 player_turn: int, last_action: ACTIONS | None, n_raises: int) -> None:
        super().__init__()
        self._board = board
        self._player_turn = player_turn
        self._money = money
        self._round, self._turn = round_n_turn
        self._last_action = last_action
        self._n_raises = n_raises

        if self._round > 0:
            assert len( self._board[2]) == 1
        # cache
        self._cached_legal_actions_masks: np.ndarray | None = None
        self._cached_result: np.ndarray | None = None
        self._cached_is_terminal: bool | None = None
        self._cached_player_obs: np.ndarray | None = None
        self._cached_full_obs: np.ndarray | None = None

    @staticmethod
    def new_state() -> 'LeducPokerState':
        p0_hand = []
        p1_hand = []
        floor = []
        deck = [CARDS.J, CARDS.J, CARDS.Q, CARDS.Q, CARDS.K, CARDS.K]
        random.shuffle(deck)
        p0_hand.append(deck.pop())
        p1_hand.append(deck.pop())
        money = (-1, -1, 2)
        board: Board = (p0_hand, p1_hand, floor, deck)
        return LeducPokerState(board=board,
                               money=money,
                               round_n_turn=(0, 0),
                               player_turn=0,
                               last_action=None,
                               n_raises=0)

    def legal_actions_masks(self) -> np.ndarray:
        assert not self.is_terminal()
        assert self._turn >= 0, "Game should be done"
        if self._cached_legal_actions_masks is not None:
            return self._cached_legal_actions_masks.copy()
        action_masks = np.zeros((len(ACTIONS),), dtype=np.int32)
        if self._turn == 0:
            action_masks[[ACTIONS.CHECKORFOLD, ACTIONS.RAISE]] = 1
        else:
            action_masks[ACTIONS.RAISE] = 1
            if self._last_action == ACTIONS.RAISE:
                action_masks[[ACTIONS.CALL, ACTIONS.CHECKORFOLD]] = 1
            if self._last_action == ACTIONS.CHECKORFOLD:
                action_masks[[ACTIONS.CHECKORFOLD, ACTIONS.RAISE]] = 1
            assert self._n_raises <= 2
            if self._n_raises == 2:
                action_masks[ACTIONS.RAISE] = 0

        self._cached_legal_actions_masks = action_masks
        return self._cached_legal_actions_masks.copy()

    def step(self, action: int) -> 'LeducPokerState':
        assert self.legal_actions_masks()[action] == 1
        return self._play_turns(action)

    def game_result(self) -> np.ndarray:
        if self._cached_result is not None:
            return self._cached_result.copy()
        assert self.is_terminal()

        p0_money, p1_money, d = self._money
        assert d == 0
        self._cached_result = np.array((p0_money, p1_money), dtype=np.float32)
        return self._cached_result.copy()

    def _play_turns(self, action: int) -> 'LeducPokerState':
        bet_amount = 2 if self._round == 0 else 4
        if self._turn == 0:
            new_turn = 1
            new_player_turn = 1-self._player_turn
            new_board: Board = copy.deepcopy(self._board)
            if action == ACTIONS.CHECKORFOLD:
                new_money = (*self._money,)
                last_action = ACTIONS.CHECKORFOLD
                return LeducPokerState(board=new_board, money=new_money,
                                       round_n_turn=(self._round, new_turn), player_turn=new_player_turn, last_action=last_action, n_raises=0)
            elif action == ACTIONS.RAISE:
                n_raises = self._n_raises + 1
                new_money = self._bet(
                    self._player_turn, self._money, bet_amount=bet_amount)
                last_action = ACTIONS.RAISE
                return LeducPokerState(board=new_board, money=new_money,
                                       round_n_turn=(self._round, new_turn),
                                       player_turn=new_player_turn,
                                       last_action=ACTIONS.RAISE,
                                       n_raises=n_raises)
            else:
                raise AssertionError("Not Reachable")
        elif self._turn == 1:
            new_player_turn = 1-self._player_turn
            new_board = copy.deepcopy(self._board)
            if action == ACTIONS.CHECKORFOLD:
                last_action = ACTIONS.CHECKORFOLD
                if self._last_action == ACTIONS.CHECKORFOLD:
                    if self._round == 0:
                        # SHOWCARD
                        new_board = self._show_card(new_board)
                        new_money = (*self._money,)
                        new_round = self._round + 1
                        return LeducPokerState(board=new_board, money=new_money,
                                               round_n_turn=(new_round, 0), player_turn=new_player_turn,
                                               last_action=last_action, n_raises=0)
                    elif self._round == 1:
                        # SHOWDOWN
                        new_money = (*self._money,)
                        new_money = self._showdown(new_money)
                        new_round = -1
                        new_turn = -1
                        return LeducPokerState(board=new_board, money=new_money,
                                               round_n_turn=(
                                                   new_round, new_turn), player_turn=new_player_turn,
                                               last_action=last_action, n_raises=0)
                    else:
                        raise AssertionError("Not Reachable")
                elif self._last_action == ACTIONS.RAISE:
                    # FOLD
                    new_money = self._fold(self._player_turn, self._money)
                    new_round = -1
                    new_turn = -1
                    return LeducPokerState(board=new_board, money=new_money, round_n_turn=(new_round, new_turn),
                                           player_turn=new_player_turn, last_action=last_action, n_raises=0)
                else:
                    raise AssertionError("Not Reachable")
            elif action == ACTIONS.CALL:
                assert self._last_action == ACTIONS.RAISE
                last_action = ACTIONS.CALL
                new_money = self._call(self._player_turn, self._money)

                if self._round == 1:
                    # SHOWDOWN
                    new_money = self._showdown(new_money)
                    new_round = -1
                    new_turn = -1
                    return LeducPokerState(board=new_board, money=new_money,
                                           round_n_turn=(new_round, new_turn),
                                           player_turn=new_player_turn,
                                           last_action=last_action, n_raises=0)
                elif self._round == 0:
                    # SHOWCARD
                    new_board = self._show_card(new_board)
                    new_round = 1
                    new_turn = 0
                    return LeducPokerState(board=new_board, money=new_money,
                                           round_n_turn=(new_round, new_turn),
                                           player_turn=new_player_turn,
                                           last_action=last_action, n_raises=0)
                else:
                    raise AssertionError("Not Reachable")
            elif action == ACTIONS.RAISE:
                last_action = ACTIONS.RAISE
                n_raises = self._n_raises+1
                if self._last_action != ACTIONS.RAISE:
                    new_money = (*self._money,)
                    new_money = self._bet(
                        self._player_turn, money=new_money, bet_amount=bet_amount)
                    new_round = self._round

                    return LeducPokerState(board=new_board, money=new_money, round_n_turn=(new_round, 2),
                                           player_turn=new_player_turn, last_action=last_action, n_raises=n_raises)
                elif self._last_action == ACTIONS.RAISE:
                    new_money = (*self._money,)
                    new_money = self._call(self.player_turn, money=new_money)
                    new_money = self._bet(
                        self._player_turn, money=new_money, bet_amount=bet_amount)
                    new_round = self._round
                    return LeducPokerState(board=new_board, money=new_money, round_n_turn=(new_round, 2),
                                           player_turn=new_player_turn, last_action=last_action, n_raises=n_raises)
                else:
                    raise AssertionError("Not Reachable")
            else:
                raise AssertionError("Not Reachable")
        elif self._turn >= 2:
            if action == ACTIONS.CALL:
                new_board = copy.deepcopy(self._board)
                new_player_turn = 1 - self._player_turn
                assert self._last_action == ACTIONS.RAISE
                last_action = ACTIONS.CALL
                new_money = self._call(self._player_turn, self._money)
                if self._round == 1:
                    # SHOWDOWN
                    new_money = self._showdown(new_money)
                    new_round = -1
                    new_turn = -1
                    return LeducPokerState(board=new_board, money=new_money,
                                           round_n_turn=(new_round, new_turn),
                                           player_turn=new_player_turn,
                                           last_action=last_action, n_raises=0)
                elif self._round == 0:
                    # SHOWCARD
                    new_board = self._show_card(new_board)
                    new_round = 1
                    new_turn = 0
                    return LeducPokerState(board=new_board, money=new_money,
                                           round_n_turn=(new_round, new_turn),
                                           player_turn=new_player_turn,
                                           last_action=last_action, n_raises=0)
                else:
                    raise AssertionError("Not reachable")

            elif action == ACTIONS.RAISE:
                new_board = copy.deepcopy(self._board)
                n_raises = self._n_raises+1
                new_round = self._round
                new_turn = self._turn + 1
                new_money = (*self._money,)
                last_action = ACTIONS.RAISE
                new_player_turn = 1 - self._player_turn
                if self._last_action == ACTIONS.RAISE:
                    new_money = self._call(self._player_turn, new_money)

                new_money = self._bet(
                    self._player_turn, money=new_money, bet_amount=bet_amount)
                return LeducPokerState(board=new_board, money=new_money, round_n_turn=(new_round, new_turn),
                                       player_turn=new_player_turn, last_action=last_action, n_raises=n_raises)

            elif action == ACTIONS.CHECKORFOLD:
                assert self._last_action == ACTIONS.RAISE
                new_player_turn = 1-self._player_turn
                new_board = copy.deepcopy(self._board)
                # FOLD
                new_money = self._fold(self._player_turn, self._money)
                new_round = -1
                new_turn = -1
                last_action = ACTIONS.CHECKORFOLD
                return LeducPokerState(board=new_board, money=new_money, round_n_turn=(new_round, new_turn),
                                       player_turn=new_player_turn, last_action=last_action, n_raises=0)
            else:
                raise AssertionError("Not Reachable")
        else:
            raise AssertionError("Not Reachable")

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    @property
    def observation_space(self) -> tuple:
        return OBSERVATION_SPACE

    @property
    def player_turn(self) -> int:
        return self._player_turn

    def is_terminal(self) -> bool:
        assert self._round < 2
        assert self._turn < 4
        return self._round < 0

    def to_player_obs(self) -> np.ndarray:
        if self._cached_player_obs is not None:
            return self._cached_player_obs.copy()

        PLAYER_CELL = 0
        OPPONENT_CELL = 1
        FLOOR_CELL = 2
        DECK_CELL = 3

        BET_ROW = len(CARDS)
        ROUND_ROW = BET_ROW+1
        player = self._player_turn
        opponent = 1-player
        deck = 3
        player_card = self._board[player][0]
        private_cards = [self._board[opponent][0], *self._board[deck]]
        player_obs = np.zeros(OBSERVATION_SPACE, dtype=np.float32)
        player_obs[player_card][PLAYER_CELL] = 1
        for card in private_cards:
            player_obs[card][OPPONENT_CELL] = 1/2
            player_obs[card][DECK_CELL] = 1/2
        floor_cards = self._board[2]
        if len(floor_cards) > 0:
            card = floor_cards[0]
            player_obs[card][FLOOR_CELL] = 1
        p0, p1, d = self._money
        # maximum_bet is 1 + 2 + 2 + 4 + 4 = 13 , normalizing it by dividing it by 5
        player_obs[BET_ROW, :] = d/5
        player_obs[ROUND_ROW,0] = self._round
        player_obs[ROUND_ROW,1] = self._turn
        self._cached_player_obs = player_obs
        return self._cached_player_obs.copy()

    def to_full_obs(self) -> np.ndarray:
        if self._cached_full_obs is not None:
            return self._cached_full_obs.copy()

        PLAYER_CELL = 0
        OPPONENT_CELL = 1
        FLOOR_CELL = 2
        DECK_CELL = 3

        BET_ROW = len(CARDS)
        ROUND_ROW = BET_ROW + 1
        player = self._player_turn
        opponent = 1-player
        deck = 3
        player_card = self._board[player][0]
        opponent_card = self._board[opponent][0]
        deck_cards = self._board[deck]
        player_full_obs = np.zeros(OBSERVATION_SPACE, dtype=np.float32)
        player_full_obs[player_card][PLAYER_CELL] += 1
        player_full_obs[opponent_card][OPPONENT_CELL] += 1
        for card in deck_cards:
            player_full_obs[card][DECK_CELL] += 1
        floor_cards = self._board[2]
        if len(floor_cards) > 0:
            card = floor_cards[0]
            player_full_obs[card][FLOOR_CELL] += 1
        assert abs(np.sum(player_full_obs[:BET_ROW]) - 6) < 1e-3
        p0, p1, d = self._money
        # maximum_bet is 1 + 2 + 2 + 4 + 4 = 13 , normalizing it by dividing it by 5
        player_full_obs[BET_ROW, :] = d/5
        player_full_obs[ROUND_ROW, 0] = self._round
        player_full_obs[ROUND_ROW, 1] = self._turn
        self._cached_full_obs = player_full_obs
        return self._cached_full_obs.copy()

    def to_player_short(self) -> tuple:
        raise NotImplementedError()

    def to_full_short(self) -> tuple:
        raise NotImplementedError()

    def render(self, full: bool):
        player = self.player_turn
        
        p0 ,p1,dd= 15,15,0
        
        if player == 0:
            card = self._board[0][0]
            player_0 = " " * 5 + f"{card.name}"
            player_1 = " " *5 + "?"
            player_1_action = self._last_action
            player_0_action = None
        else:
            card = self._board[1][0]
            player_0 = " " * 5 + "?"
            player_1 = " " * 5 + f"{card.name}"
            player_1_action = None
            player_0_action =  self._last_action
        
        rp0 , rp1 , d = self._money
        player_0_money = p0+rp0
        player_1_money = p1+rp1
        deck_money = d
        floor = self._board[2]
        floor_card = " "
        if len(floor) > 0:
            floor_card = floor[0].name
        
        print("*"*10)
        print(player_1)
        print(" "*5 + f"{player_1_money}$" + " "*5 +  (player_1_action.name if player_1_action is not None else ""))
        print(" "*5 + f"{floor_card}" + " "*2 + f"{deck_money}$")
        print(" "*5 + f"{player_0_money}$" + " "*5 +  (player_0_action.name if player_0_action is not None else ""))
        print(player_0)

        legal_actions = self.legal_actions_masks() if not self.is_terminal() else np.zeros((len(ACTIONS),),dtype=np.int32)
        pr = ""
        for i,a in enumerate(ACTIONS):
            if legal_actions[i] == 1:
                pr += f"{a.name}:{a.value}\t"
        print(pr)
        

    def _bet(self, player: int, money: tuple[int, int, int], bet_amount: int):
        p0, p1, d = money
        if player == 0:
            p0 -= bet_amount
        else:
            p1 -= bet_amount
        d += bet_amount
        return p0, p1, d

    def _call(self, player: int, money: tuple[int, int, int]):
        p0, p1, d = money
        if player == 0:
            bet_amount = p0 - p1
            p0 -= bet_amount
        else:
            bet_amount = p1 - p0
            p1 -= bet_amount
        d += bet_amount
        assert bet_amount > 0, f"caller betting amount should be bigger than 0 , got '{bet_amount}'"
        return p0, p1, d

    def _show_card(self, board: Board) -> Board:
        p0_hand, p1_hand, floor, deck = board
        card = deck.pop()
        floor.append(card)
        return p0_hand, p1_hand, floor, deck

    def _fold(self, player, money: tuple[int, int, int]):
        p0, p1, d = money
        if player == 0:
            p1 += d
        else:
            p0 += d
        d = 0
        return p0, p1, d

    def _showdown(self, money: tuple[int, int, int]):
        p0, p1, d = money
        p0_hand, p1_hand, floor, _ = self._board
        p0_card = p0_hand[0]
        p1_card = p1_hand[0]
        floor_card = floor[0]
        if floor_card == p0_card:
            p0 += d
        elif floor_card == p1_card:
            p1 += d
        elif p0_hand > p1_hand:
            p0 += d
        elif p0_hand < p1_hand:
            p1 += d
        else:
            assert d % 2 == 0
            split = d//2
            p0 += split
            p1 += split
        d = 0
        return p0, p1, d
