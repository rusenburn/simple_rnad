import os
import torch as T
from rnad.games.phantom_tictactoe.game import PhantomTicTacToeGame
from rnad.algorithms.ppo.ppo import PPO
from rnad.match import Match
from rnad.players import CheaterPlayer, RandomPlayer,NNPlayer
from rnad.networks import ActorResNetwork, ClippedActorResNetwork


def train_pttc_using_ppo():
    ppo = PPO(
        game_fns=[lambda:PhantomTicTacToeGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=1.0)
    ppo.run()
    
def match_pttc_players():
    game_fn = lambda:PhantomTicTacToeGame()
    game = game_fn()
    device = "cuda:0" if T.cuda.is_available() else "cpu"
    nnet_1 = ActorResNetwork(game.observation_space,game.n_actions).to(device)
    nnet_1.load_model(os.path.join("tmp","actor_4.pt"))
    player_1 = CheaterPlayer(nnet_1)
    # player_1 = RandomPlayer()
    # player_2 = NNPlayer(nnet_1)

    nnet_2 = ActorResNetwork(game.observation_space,game.n_actions).to(device)
    nnet_2.load_model(os.path.join("tmp","actor.pt"))
    player_2 = NNPlayer(nnet_2)

    match_ = Match(game_fn,player_1,player_2,100)
    scores = match_.start()
    print(scores)

def main():
    # train_pttc_using_ppo()
    match_pttc_players()



if __name__ == "__main__":
    main()