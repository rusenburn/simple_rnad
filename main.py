import os
import torch as T
from rnad.algorithms.nfsp.nfsp import NFSP
from rnad.algorithms.past_self.two_nets import TwoNets
from rnad.algorithms.ppg.ppg import PPG
from rnad.algorithms.ppg.trainers import LegalActionAuxTrainer, NeurdPolicyTrainer, PPOPolicyTrainer
from rnad.algorithms.ppo.nrd import NRD
from rnad.algorithms.rnad.rnad import Rnad
from rnad.games.kuhn_poker.game import KuhnPokerGame
from rnad.games.leduc_poker.game import LeducPokerGame
from rnad.games.othello.game import OthelloGame
from rnad.games.phantom_tictactoe.game import PhantomTicTacToeGame
from rnad.games.goofspiel.game import GoofSpielGame
from rnad.games.trix.game import TrixGame
from rnad.algorithms.ppo.ppo import PPO
from rnad.algorithms.past_self.past_self import PastSelf
from rnad.match import Match
from rnad.players import CheaterPlayer, HumanPlayer, RandomPlayer,NNPlayer, TurnBasedNNPlayer
from rnad.networks import ActorClippedLinearNetwork, ActorLinNetwork, ActorLinearNetwork, ActorRNetwork, ActorResNetwork, ClippedActorResNetwork, CriticLinNetwork, CriticResNetwork, RnadNetwork, SmallActorNetwork


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
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=0.01,
        testing_game_fn=lambda:GoofSpielGame())
    ppo.run()


def train_goof_using_past_ppo():
    ppo = PastSelf(
        game_fns=[lambda:GoofSpielGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=0.01,
        testing_game_fn=lambda:GoofSpielGame())
    ppo.run()



def train_goof_using_two_ppo():
    ppo = TwoNets(
        game_fns=[lambda:GoofSpielGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=0.01,
        testing_game_fn=lambda:GoofSpielGame())
    ppo.run()
def train_goof_using_rnad():
    rnad = Rnad(lambda:GoofSpielGame(),
                save_name="goof",
                n_steps=2*10**7,
                eta=0.2,
                # lr=5e-5,
                n_epochs=1,
                delta_m=300_000,
                gamma_avg=0.001,
                test_intervals=50_000,
                load_name="goof")
    rnad.run()


def train_pttc_using_rnad():
    rnad = Rnad(lambda:PhantomTicTacToeGame(),n_epochs=1,
                n_steps=10_000_000,
                delta_m=1_000_000,
                eta=0.2,
                gamma_avg=0.01,
                test_intervals=50_000,
                save_name="pttt_target_probs",
                )
    rnad.run()

def train_kuhn_using_rnad():
    rnad = Rnad(lambda:KuhnPokerGame(),
                n_steps=2*10**6,
                eta=0.1,
                save_name="kuhn_poker_rnad",
                lr=5e-5)
    rnad.run()

def train_othello_using_rnad():
    rnad = Rnad(lambda:OthelloGame(),
                n_steps=2*10**6,
                eta=0.1,
                save_name="othello",
                n_epochs=1,
                n_episodes=8,
                lr=5e-5)
    rnad.run()
def train_kuhn_poker_using_ppo():
    ppo = PPO(
        game_fns=[lambda:KuhnPokerGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=0.01,
        testing_game_fn=lambda:KuhnPokerGame(),
        save_name="kuhn_poker_ppo"
    )
    ppo.run()
def train_kuhn_poker_using_nrd():
    ppo = NRD(
        game_fns=[lambda:KuhnPokerGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=0.01,
        testing_game_fn=lambda:KuhnPokerGame(),
        save_name="kuhn_poker_nrd"
    )
    ppo.run()
def train_kuhn_poker_using_past_ppo():
    ppo = PastSelf(
        game_fns=[lambda:KuhnPokerGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=.01,
        testing_game_fn=lambda:KuhnPokerGame(),
        save_name="kuhn_poker_past_ppo")
    ppo.run()

def train_trix_using_ppo():
    ppo = PPO(
    game_fns=[lambda:TrixGame() for _ in range(8)],
        total_steps=1_000_000,critic_coef=0.5,entropy_coef=0.00,step_size=156,normalize_adv=True,
        testing_game_fn=lambda:TrixGame(),
        save_name="trix_ppo")
    ppo.run()

def train_othello_using_ppo():
    ppo = PPO(
        game_fns=[lambda:OthelloGame() for _ in range(8)],
        total_steps=500_000,critic_coef=0.5,entropy_coef=0.00,step_size=128,normalize_adv=True,
        decay_lr=False,
        testing_game_fn=lambda:OthelloGame(),
        save_name="othello_ppo")
    ppo.run()

def train_othello_using_ppg():
    game_fns=[lambda:OthelloGame() for _ in range(8)]
    game = game_fns[0]()
    actors = [ActorRNetwork(game.observation_space,game.n_actions,n_blocks=5) for _ in range(2)]
    critics = [CriticResNetwork(game.observation_space,n_blocks=5) for _ in range(2)]
    policy_trainer = PPOPolicyTrainer()
    actors,critics,actors_optims,critics_optims = policy_trainer.set_trained_network(actors,critics)
    aux_trainer = LegalActionAuxTrainer()
    aux_trainer.set_trained_network(actors,actors_optims)
    ppg = PPG(
        max_steps=2*10**6,
        policy_trainer=policy_trainer,
        aux_trainer=aux_trainer,
        critics=critics,
        actors=actors,
        game_fns=game_fns,
        testing_game_fn=game_fns[0],
        save_name="othello_ppg")
    ppg.run()


def train_pttc_using_ppg():
    game_fns=[lambda:PhantomTicTacToeGame() for _ in range(8)]
    game = game_fns[0]()
    actors = [ActorRNetwork(game.observation_space,game.n_actions,n_blocks=5) for _ in range(2)]
    critics = [CriticResNetwork(game.observation_space,n_blocks=5) for _ in range(2)]
    policy_trainer = PPOPolicyTrainer()
    actors,critics,actors_optims,critics_optims = policy_trainer.set_trained_network(actors,critics)
    aux_trainer = LegalActionAuxTrainer()
    aux_trainer.set_trained_network(actors,actors_optims)
    ppg = PPG(
        max_steps=1_000_000,
        policy_trainer=policy_trainer,
        aux_trainer=aux_trainer,
        critics=critics,
        actors=actors,
        game_fns=game_fns,
        testing_game_fn=game_fns[0],
        save_name="pttt_ppg")
    ppg.run()

def  train_kuhn_using_neurd_ppg():
    game_fns=[lambda:KuhnPokerGame() for _ in range(8)]
    game = game_fns[0]()
    actors = [ActorLinNetwork(game.observation_space,game.n_actions,n_blocks=3) for _ in range(2)]
    critics = [CriticLinNetwork(game.observation_space,n_blocks=3) for _ in range(2)]
    policy_trainer = NeurdPolicyTrainer()
    actors,critics,actors_optims,critics_optims = policy_trainer.set_trained_network(actors,critics)
    aux_trainer = LegalActionAuxTrainer()
    aux_trainer.set_trained_network(actors,actors_optims)
    ppg = PPG(
        max_steps=500_000,
        policy_trainer=policy_trainer,
        aux_trainer=aux_trainer,
        critics=critics,
        actors=actors,
        game_fns=game_fns,
        testing_game_fn=game_fns[0],
        save_name="kuhn_neurd_ppg")
    ppg.run()

def train_using_ppg():
    game_fns=[lambda:GoofSpielGame() for _ in range(8)]
    game = game_fns[0]()
    actors = [ActorLinNetwork(game.observation_space,game.n_actions,n_blocks=3) for _ in range(2)]
    critics = [CriticLinNetwork(game.observation_space,n_blocks=3) for _ in range(2)]
    policy_trainer = PPOPolicyTrainer()
    actors,critics,actors_optims,critics_optims = policy_trainer.set_trained_network(actors,critics)
    aux_trainer = LegalActionAuxTrainer()
    aux_trainer.set_trained_network(actors,actors_optims)
    ppg = PPG(
        policy_trainer=policy_trainer,
        aux_trainer=aux_trainer,
        critics=critics,
        actors=actors,
        game_fns=game_fns,
        testing_game_fn=game_fns[0],
        save_name="goof_ppg")
    ppg.run()

def train_leduc_poker_using_ppg():
    game_fns = [lambda:LeducPokerGame() for _ in range(8)]
    game = game_fns[0]()
    actors = [SmallActorNetwork(game.observation_space,game.n_actions,n_layers=2) for _ in range(2)]
    critics = [CriticLinNetwork(game.observation_space,n_blocks=3) for _ in range(2)]
    policy_trainer = NeurdPolicyTrainer()
    actors,critics,actors_optims,critics_optims = policy_trainer.set_trained_network(actors,critics)
    aux_trainer = LegalActionAuxTrainer()
    aux_trainer.set_trained_network(actors,actors_optims)
    ppg = PPG(
        policy_trainer=policy_trainer,
        aux_trainer=aux_trainer,
        critics=critics,
        actors=actors,
        game_fns=game_fns,
        testing_game_fn=game_fns[0],
        save_name="leduc_ppg")
    ppg.run()
def train_leduc_using_nfsp():
    game_fn = lambda:LeducPokerGame()
    algo = NFSP(game_fn,batches=1,update_intervals=1,save_name="leduc1")
    algo.run()

def match_leduc_players():
    game_fn = lambda:LeducPokerGame()
    game = game_fn()
    device = "cuda:0" if T.cuda.is_available() else "cpu"
    net_0s = [SmallActorNetwork(game.observation_space,game.n_actions,n_layers=2).to(device) for _ in range(2)]
    net_0s[0].load_model(os.path.join("tmp","leduc_ppg_0_actor.pt"))
    net_0s[1].load_model(os.path.join("tmp","leduc_ppg_1_actor.pt"))

    p0 = TurnBasedNNPlayer(net_0s)

    
    net_1 = RnadNetwork(game.observation_space,game.n_actions,blocks=5).to(device)
    net_1.load_model(os.path.join("tmp","leduc_rnad_rnad.pt"))

    p1 = NNPlayer(net_1)

    m = Match(game_fn,p0,p1,5000,False)
    score = m.start()
    print(score)
    
def train_leduc_poker_using_rnad():
    game_fns = [lambda:LeducPokerGame() for _ in range(32)]
    rnad = Rnad(game_fn=game_fns[0],lr=5e-5,
                save_name="leduc_rnad",
                n_epochs=4,
                delta_m=150_000,
                test_intervals=50000)
    rnad.run()

def train_othello_using_nfsp():
    game_fn = lambda:OthelloGame()
    algo = NFSP(game_fn,batches=1,update_intervals=1)
    algo.run()



def train_pttt_using_nfsp():
    game_fn = lambda:PhantomTicTacToeGame()
    algo = NFSP(game_fn,batches=1,update_intervals=1,save_name="pttc")
    algo.run()
def match_trix_players():
    game_fn = lambda : OthelloGame()
    game = game_fn()
    device = "cuda:0" if T.cuda.is_available() else "cpu"
    nnet_1 =  ActorResNetwork(game.observation_space,game.n_actions)
    # nnet_1 =   ActorLinearNetwork(game.observation_space,game.n_actions)
    nnet_1.load_model(os.path.join("tmp","trix_past_actor.pt"))
    # nnet_1.load_model(os.path.join("tmp","kuhn_poker_ppo_actor.pt"))
    nnet_1.to(device)
    player_1 = NNPlayer(nnet_1)
    # player_1 = RandomPlayer()

    # player_2 = RandomPlayer()
    player_2 = HumanPlayer(False)
    # nnet_2 = ActorLinearNetwork(game.observation_space,game.n_actions)
    # nnet_2.load_model(os.path.join("tmp","kuhn_poker_nrd_actor.pt"))
    # nnet_2.to(device)
    # player_2 = NNPlayer(nnet_2)
    m = Match(game_fn,player_1,player_2,1000)
    score = m.start()
    print(score)

def match_kuhn_poker_players():
    game_fn = lambda : KuhnPokerGame()
    game = game_fn()
    device = "cuda:0" if T.cuda.is_available() else "cpu"
    # nnet_1 =  ActorClippedLinearNetwork(game.observation_space,game.n_actions)
    nnet_0 = ActorLinNetwork(game.observation_space,game.n_actions)
    nnet_1 = ActorLinNetwork(game.observation_space,game.n_actions)
    # nnet_1 =   ActorLinearNetwork(game.observation_space,game.n_actions)
    nnet_0.load_model(os.path.join("tmp","kuhn_poker_ppg_0_actor.pt"))
    nnet_1.load_model(os.path.join("tmp","kuhn_poker_ppg_1_actor.pt"))
    # nnet_1.load_model(os.path.join("tmp","kuhn_poker_ppo_actor.pt"))
    nnet_0.to(device)
    nnet_1.to(device)
    player_1 = TurnBasedNNPlayer([nnet_0,nnet_1])
    # player_1 = RandomPlayer()

    # player_2 = RandomPlayer()
    # nnet_2 = ActorLinearNetwork(game.observation_space,game.n_actions)
    nnet_2 = RnadNetwork(game.observation_space,game.n_actions,blocks=5)
    nnet_2.load_state_dict(T.load(os.path.join("tmp","kuhn_poker_rnad_rnad.pt")))
    nnet_2.to(device)
    player_2 = NNPlayer(nnet_2)
    m = Match(game_fn,player_1,player_2,1000)
    score = m.start()
    print(score)

def match_pttc_players():
    game_fn = lambda:PhantomTicTacToeGame()
    game = game_fn()
    device = "cuda:0" if T.cuda.is_available() else "cpu"
    # nnets = tuple((ActorRNetwork(game.observation_space,game.n_actions).to(device) for _ in range(2)))
    # nnets[0].load_model(os.path.join("tmp","pttc_ppg_0_actor.pt"))
    # nnets[1].load_model(os.path.join("tmp","pttc_ppg_1_actor.pt"))
    nnet_1 = ActorResNetwork(game.observation_space,game.n_actions).to(device)
    nnet_1.load_model(os.path.join("tmp","actor_3.pt"))
    # nnet_1.load_model(os.path.join("tmp","pttc_nfsp.pt"))
    player_1 = NNPlayer(nnet_1)
    # player_1 = TurnBasedNNPlayer(nnets)
    # player_1 = RandomPlayer()
    # player_2 = NNPlayer(nnet_1)
    # player_2 = HumanPlayer(True)

    # nnet_2 = RnadNetwork(game.observation_space,game.n_actions,blocks=5).to(device)
    nnet_2 = ActorResNetwork(game.observation_space,game.n_actions).to(device)
    nnet_2.load_model(os.path.join("tmp","pttc_nfsp.pt"))
    # nnet_2.load_model(os.path.join("tmp","pttt_rnad.pt"))
    player_2 = NNPlayer(nnet_2)
    # player_2 = RandomPlayer()

    match_ = Match(game_fn,player_1,player_2,100,False)
    scores = match_.start()
    print(scores)

def match_goof_players():
    game_fn = lambda:GoofSpielGame()
    game = game_fn()
    device = "cuda:0" if T.cuda.is_available() else "cpu"

    nnets = tuple(ActorLinNetwork(game.observation_space,game.n_actions) for _ in range(2))
    nnets[0].load_model(os.path.join("tmp","goof_spiel_ppg_0_actor.pt"))
    nnets[1].load_model(os.path.join("tmp","goof_spiel_ppg_1_actor.pt"))
    [net.to(device) for net in nnets]
    # player_1 = MultiNNplayer
    # nnet_1 = ActorLinearNetwork(game.observation_space,game.n_actions).to(device)
    # nnet_1.load_model(os.path.join("tmp","goof_actor.pt"))
    # nnet_1 = ActorClippedLinearNetwork(game.observation_space,game.n_actions).to(device)
    # nnet_1.load_model(os.path.join("tmp","twin_actor_1.pt"))
    player_1 = TurnBasedNNPlayer(nnets)
    # player_1 = RandomPlayer()
    # player_2 = NNPlayer(nnet_1)
    # player_1 = HumanPlayer(False)

    # nnet_2 = ActorClippedLinearNetwork(game.observation_space,game.n_actions).to(device)
    nnet_2 = RnadNetwork(game.observation_space,game.n_actions,blocks=5).to(device)
    state_dictt = T.load(os.path.join("tmp","goof_rnad.pt"))
    nnet_2.load_state_dict(state_dictt)
    # nnet_2.load_model(os.path.join("tmp","actor.pt"))
    # nnet_2.load_model(os.path.join("tmp","twin_actor_1.pt"))
    player_2 = NNPlayer(nnet_2)
    # player_2 = HumanPlayer(False)

    match_ = Match(game_fn,player_1,player_2,500,False)
    scores = match_.start()
    print(scores)

def match_othello_players():
    game_fn = lambda : OthelloGame()
    game = game_fn()
    device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

    net_0 = ActorResNetwork(game.observation_space,game.n_actions).to(device=device)
    net_0.load_model(os.path.join("tmp","_nfsp.pt"))
    # net0_0 = ActorRNetwork(game.observation_space,game.n_actions)
    # net0_1 = ActorRNetwork(game.observation_space,game.n_actions)
    # net0_0.load_model(os.path.join("tmp","othello_ppg_0_actor.pt"))
    # net0_1.load_model(os.path.join("tmp","othello_ppg_1_actor.pt"))
    # net0_0.to(device)
    # net0_1.to(device)
    # player_0 = TurnBasedNNPlayer([net0_0,net0_1])
    player_0 = NNPlayer(net_0)
    net_1 = RnadNetwork(game.observation_space,game.n_actions,blocks=5)
    net_1.load_state_dict(T.load(os.path.join("tmp","othello_rnad.pt")))
    net_1.to(device)
    player_1 = NNPlayer(net_1)

    m = Match(game_fn,player_1=player_0,player_2=player_1,n_sets=500)
    score = m.start()
    print(score)
def main():
    # train_pttc_using_past_ppo()
    # train_pttc_using_ppo()
    # train_pttc_using_ppo()
    # match_pttc_players()
    # train_goof_using_ppo()
    # train_goof_using_past_ppo()
    # match_goof_players()
    # train_goof_using_rnad()
    # train_pttc_using_rnad()
    # train_kuhn_using_rnad()
    # train_othello_using_rnad()
    # train_goof_using_two_ppo()
    # train_kuhn_poker_using_past_ppo()
    # match_kuhn_poker_players()
    # train_kuhn_poker_using_ppo()
    # train_kuhn_poker_using_nrd()
    # train_trix_using_ppo()
    # train_othello_using_ppo()
    # train_othello_using_ppg()
    # match_othello_players()
    # train_pttc_using_ppg()
    # match_trix_players()
    # train_using_ppg()
    # train_kuhn_using_neurd_ppg()
    # train_leduc_poker_using_ppg()
    # train_othello_using_nfsp()
    # match_leduc_players()
    # train_leduc_poker_using_rnad()
    # train_pttt_using_nfsp()
    train_leduc_using_nfsp()

if __name__ == "__main__":
    main()