import os
import torch as T
from rnad.games.phantom_tictactoe.game import PhantomTicTacToeGame
from rnad.games.goofspiel.game import GoofSpielGame
from rnad.algorithms.ppo.ppo import PPO
from rnad.algorithms.past_self.past_self import PastSelf
from rnad.match import Match
from rnad.players import CheaterPlayer, HumanPlayer, RandomPlayer,NNPlayer
from rnad.networks import ActorLinearNetwork, ActorResNetwork, ClippedActorResNetwork


def train_pttc_using_ppo():
    ppo = PPO(
        game_fns=[lambda:PhantomTicTacToeGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=0.02,
        testing_game_fn=lambda:PhantomTicTacToeGame())
    ppo.run()



def train_pttc_using_past_ppo():
    ppo = PastSelf(
        game_fns=[lambda:PhantomTicTacToeGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=0.02,
        testing_game_fn=lambda:PhantomTicTacToeGame())
    ppo.run()

def train_goof_using_ppo():
    ppo = PPO(
        game_fns=[lambda:GoofSpielGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=0.02,
        testing_game_fn=lambda:GoofSpielGame())
    ppo.run()


def train_goof_using_past_ppo():
    ppo = PastSelf(
        game_fns=[lambda:GoofSpielGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=0.01,
        testing_game_fn=lambda:GoofSpielGame())
    ppo.run()
def match_pttc_players():
    game_fn = lambda:PhantomTicTacToeGame()
    game = game_fn()
    device = "cuda:0" if T.cuda.is_available() else "cpu"
    nnet_1 = ActorResNetwork(game.observation_space,game.n_actions).to(device)
    nnet_1.load_model(os.path.join("tmp","actor.pt"))
    player_1 = NNPlayer(nnet_1)
    # player_1 = RandomPlayer()
    # player_2 = NNPlayer(nnet_1)
    # player_1 = HumanPlayer(False)

    nnet_2 = ActorResNetwork(game.observation_space,game.n_actions).to(device)
    nnet_2.load_model(os.path.join("tmp","actor_3.pt"))
    player_2 = CheaterPlayer(nnet_2)
    # player_2 = HumanPlayer(False)

    match_ = Match(game_fn,player_1,player_2,100,False)
    scores = match_.start()
    print(scores)

def match_goof_players():
    game_fn = lambda:GoofSpielGame()
    game = game_fn()
    device = "cuda:0" if T.cuda.is_available() else "cpu"
    nnet_1 = ActorLinearNetwork(game.observation_space,game.n_actions).to(device)
    nnet_1.load_model(os.path.join("tmp","goof_actor.pt"))
    player_1 = NNPlayer(nnet_1)
    # player_1 = RandomPlayer()
    # player_2 = NNPlayer(nnet_1)
    # player_1 = HumanPlayer(False)

    nnet_2 = ActorLinearNetwork(game.observation_space,game.n_actions).to(device)
    nnet_2.load_model(os.path.join("tmp","goof_actor_2.pt"))
    player_2 = NNPlayer(nnet_2)
    # player_2 = HumanPlayer(False)

    match_ = Match(game_fn,player_1,player_2,100,False)
    scores = match_.start()
    print(scores)

def main():
    # train_pttc_using_past_ppo()
    # train_pttc_using_ppo()
    # match_pttc_players()
    # train_goof_using_ppo()
    # train_goof_using_past_ppo()
    match_goof_players()



if __name__ == "__main__":
    main()