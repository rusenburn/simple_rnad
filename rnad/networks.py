from abc import ABC, abstractmethod
import torch as T
import torch.nn as nn


class NetworkBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def save_model(self, path: str | None) -> None:
        '''
        Saves the model into path
        '''
        raise NotImplementedError("Calling abstract save_model")

    @abstractmethod
    def load_model(self, path: str) -> None:
        '''
        Loads model from path
        '''
        raise NotImplementedError("Calling abstract load_model")


class PytorchNetwork(nn.Module, NetworkBase):
    def __init__(self) -> None:
        super().__init__()

    def save_model(self, path: str) -> None:
        try:
            T.save(self.state_dict(), path)
        except:
            print(f'could not save nn to {path}')

    def load_model(self, path: str) -> None:
        try:
            self.load_state_dict(T.load(path))
            print(f'The nn was loaded from {path}')
        except:
            print(f'could not load nn from {path}')

class ActorNetwork(PytorchNetwork):
    def __init__(self) -> None:
        super().__init__()

    def predict(self,observation:T.Tensor)->tuple[T.Tensor,T.Tensor]:
        '''
        returns a tuple containing action probabilities and logits
        '''
        raise NotImplementedError()

class CriticNetwork(PytorchNetwork):
    def __init__(self) -> None:
        super().__init__()
    
    def evaluate(self,observation:T.Tensor)->T.Tensor:
        '''
        return the value of the observation
        '''
        raise NotImplementedError()


class StateActionNetwork(PytorchNetwork):
    def __init__(self) -> None:
        super().__init__()
    
    def evaluate(self,observation:T.Tensor)->T.Tensor:
        '''
        return state-action values
        '''
        raise NotImplementedError()

class SmallActorNetwork(ActorNetwork):
    def __init__(self,shape:tuple,n_actions:int,fc_dims=128,n_layers=2) -> None:
        super().__init__()
        assert n_layers >= 1
        self.pi_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(shape[0]*shape[1],fc_dims),
            nn.ReLU()
        )
        for _ in range(n_layers-1):
            self.pi_head.append(nn.Linear(fc_dims,fc_dims))
            self.pi_head.append(nn.ReLU())

        self.pi_head.append(nn.Linear(fc_dims,n_actions))
        self.pi_head.to("cuda:0" if T.cuda.is_available() else "cpu")
    
    def forward(self,observations: T.Tensor)->tuple[T.Tensor,T.Tensor]:
        logits : T.Tensor= self.pi_head(observations)
        probs = logits.softmax(dim=-1)
        return probs,logits
    
    def predict(self, observation: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        return self(observation)

        
class ActorLinNetwork(ActorNetwork):
    def __init__(self,shape:tuple,n_actions:int,fc_dims=512,n_blocks=5) -> None:
        super().__init__()
        self._blocks = nn.ModuleList(
            LinearBlock(fc_dims) for _ in range(n_blocks)
        )
        self._pi_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(shape[0]*shape[1], fc_dims),
            nn.ReLU(),
            *self._blocks,
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions))
        self._pi_head.to("cuda:0" if T.cuda.is_available() else "cpu")
    
    def forward(self,observations:T.Tensor)->tuple[T.Tensor,T.Tensor]:
        logits:T.Tensor = self._pi_head(observations)
        probs = logits.softmax(-1)
        return probs,logits

    def predict(self, observation: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        return self(observation)
    
class CriticLinNetwork(CriticNetwork):
    def __init__(self,shape:tuple,fc_dims=512,n_blocks=5) -> None:
        super().__init__()
        self._blocks = nn.ModuleList(
            [LinearBlock(fc_dims) for _ in range(n_blocks)])

        self._v_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(shape[0]*shape[1], fc_dims),
            nn.ReLU(),
            *self._blocks,
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1))
    
    def forward(self,observation:T.Tensor)->T.Tensor:
        v = self._v_head(observation)
        return v

    def evaluate(self, observation: T.Tensor) -> T.Tensor:
        return self(observation)

class ActorLinearNetwork(PytorchNetwork):
    def __init__(self, shape: tuple, n_actions: int, fc_dims=512, blocks=3) -> None:
        super().__init__()

        self._blocks = nn.ModuleList(
            [LinearBlock(fc_dims) for _ in range(blocks)])

        self._pi_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(shape[0]*shape[1], fc_dims),
            nn.ReLU(),
            *self._blocks,
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions)
        )
        self._pi_head.to("cuda:0" if T.cuda.is_available() else "cpu")

    def forward(self, state: T.Tensor) -> T.Tensor:
        logits: T.Tensor = self._pi_head(state)
        probs = logits.softmax(dim=-1)
        return probs

    def predict(self,state:T.Tensor)->T.Tensor:
        return self(state)


class ActorClippedLinearNetwork(PytorchNetwork):
    def __init__(self, shape: tuple, n_actions: int, fc_dims=512, blocks=3) -> None:
        super().__init__()

        self._blocks = nn.ModuleList(
            [LinearBlock(fc_dims) for _ in range(blocks)])

        self._pi_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(shape[0]*shape[1], fc_dims),
            nn.ReLU(),
            *self._blocks,
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions),
            nn.Tanh()
        )
        self._pi_head.to("cuda:0" if T.cuda.is_available() else "cpu")

    def forward(self, state: T.Tensor) -> T.Tensor:
        logits: T.Tensor = self._pi_head(state)
        logits = logits * 3
        # logits = logits.clamp(-3,3)
        probs = logits.softmax(dim=-1)
        return probs


class CriticLinearNetwork(PytorchNetwork):
    def __init__(self, shape: tuple, fc_dims=512, blocks=3) -> None:
        super().__init__()

        self._blocks = nn.ModuleList(
            [LinearBlock(fc_dims) for _ in range(blocks)])

        self._v_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(shape[0]*shape[1], fc_dims),
            nn.ReLU(),
            *self._blocks,
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1),
            # nn.Tanh()
        )

        self._v_head.to("cuda:0" if T.cuda.is_available() else "cpu")

    def forward(self, state: T.Tensor) -> T.Tensor:
        v: T.Tensor = self._v_head(state)
        return v

class ActorRNetwork(ActorNetwork):
    def __init__(self,
                 shape: tuple,
                 n_actions: int,
                 filters=128,
                 fc_dims=512,
                 n_blocks=5) -> None:
        super().__init__()
    
        self._blocks = nn.ModuleList(
                [ResBlock(filters) for _ in range(n_blocks)])

        self._shared = nn.Sequential(
                nn.Conv2d(shape[0], filters, 3, 1, 1),
                *self._blocks)

        self._pi_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions))

        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self._blocks.to(device)
        self._shared.to(device)
        self._pi_head.to(device)

    def forward(self, state: T.Tensor) -> tuple[T.Tensor,T.Tensor]:
        shared: T.Tensor = self._shared(state)
        logits: T.Tensor = self._pi_head(shared)
        # pi = pi.clamp(-3,3)
        probs: T.Tensor = logits.softmax(dim=-1)
        return probs,logits

    def predict(self, observation: T.Tensor) -> tuple[T.Tensor, T.Tensor]:
        return self(observation)
    
class ActorResNetwork(PytorchNetwork):
    def __init__(self,
                 shape: tuple,
                 n_actions: int,
                 filters=128,
                 fc_dims=512,
                 n_blocks=3):
        super().__init__()

        self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(n_blocks)])

        if shape[1] == 64:
            self._shared = nn.Sequential(
                nn.Conv2d(shape[0], filters, 8, 4, 0),
                *self._blocks)
            shape = shape[0], 15, 13
        else:
            self._shared = nn.Sequential(
                nn.Conv2d(shape[0], filters, 3, 1, 1),
                *self._blocks)
        # self._shared = nn.Sequential(
        #     nn.Conv2d(shape[0], filters, 3, 1, 1),
        #     *self._blocks)

        self._pi_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions))

        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self._blocks.to(device)
        self._shared.to(device)
        self._pi_head.to(device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        shared: T.Tensor = self._shared(state)
        pi: T.Tensor = self._pi_head(shared)
        probs: T.Tensor = pi.softmax(dim=-1)
        return probs

    def predict(self,state:T.Tensor)->T.Tensor:
        return self(state)


class ClippedActorResNetwork(PytorchNetwork):
    def __init__(self,
                 shape: tuple,
                 n_actions: int,
                 filters=128,
                 fc_dims=512,
                 n_blocks=3):
        super().__init__()

        self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(n_blocks)])

        self._shared = nn.Sequential(
            nn.Conv2d(shape[0], filters, 3, 1, 1),
            *self._blocks)

        self._pi_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions))

        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self._blocks.to(device)
        self._shared.to(device)
        self._pi_head.to(device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        shared: T.Tensor = self._shared(state)
        pi: T.Tensor = self._pi_head(shared)
        pi = pi.clamp(-2.0, 2.0)
        probs: T.Tensor = pi.softmax(dim=-1)
        return probs


class CriticResNetwork(CriticNetwork):
    def __init__(self,
                 shape: tuple,
                 filters=128,
                 fc_dims=512,
                 n_blocks=3):

        super().__init__()
        self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(n_blocks)])

        if shape[1] == 64:
            self._shared = nn.Sequential(
                nn.Conv2d(shape[0], filters, 8, 4, 0),
                *self._blocks)
            shape = shape[0], 15, 13
        else:
            self._shared = nn.Sequential(
            nn.Conv2d(shape[0], filters, 3, 1, 1),
            *self._blocks)

        self._value_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1),
            # nn.Tanh()
        )

        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self._blocks.to(device)
        self._shared.to(device)
        self._value_head.to(device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        shared: T.Tensor = self._shared(state)
        value: T.Tensor = self._value_head(shared)
        return value

    def evaluate(self, observation: T.Tensor) -> T.Tensor:
        return self(observation)

    


class LinearBlock(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self._block = nn.Sequential(
            nn.Linear(dims, dims),
            nn.Linear(dims, dims),
        )
        self._se = LinearSE(dims, squeeze_rate=4)

    def forward(self, state: T.Tensor) -> T.Tensor:
        output: T.Tensor = self._block(state)
        output = self._se(output, state)
        output += state
        output = output.relu()
        return output


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self._se = SqueezeAndExcite(channels, squeeze_rate=4)

    def forward(self, state: T.Tensor) -> T.Tensor:
        initial = state
        output: T.Tensor = self._block(state)
        output = self._se(output, initial)
        output += initial
        output = output.relu()
        return output


class LinearSE(nn.Module):
    def __init__(self, dims: int, squeeze_rate: int) -> None:
        super().__init__()
        self.dims = dims
        self._fcs = nn.Sequential(
            nn.Linear(dims, int(dims//squeeze_rate)),
            nn.ReLU(),
            nn.Linear(int(dims//squeeze_rate), dims*2))

    def forward(self, state: T.Tensor, input_: T.Tensor) -> T.Tensor:
        prepared: T.Tensor = self._fcs(state)
        splitted = prepared.split(self.dims, dim=1)
        w: T.Tensor = splitted[0]
        b: T.Tensor = splitted[1]
        z = w.sigmoid()
        output = input_ * z + b
        return output


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_rate):
        super().__init__()
        self.channels = channels
        self.prepare = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self._fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, int(channels//squeeze_rate)),
            nn.ReLU(),
            nn.Linear(int(channels//squeeze_rate), channels*2)
        )

    def forward(self, state: T.Tensor, input_: T.Tensor) -> T.Tensor:
        shape_ = input_.shape
        prepared: T.Tensor = self.prepare(state)
        prepared = self._fcs(prepared)
        splitted = prepared.split(self.channels, dim=1)
        w: T.Tensor = splitted[0]
        b: T.Tensor = splitted[1]
        z = w.sigmoid()
        z = z.unsqueeze(-1).unsqueeze(-1).expand((-1, -
                                                  1, shape_[-2], shape_[-1]))
        b = b.unsqueeze(-1).unsqueeze(-1).expand((-1, -
                                                  1, shape_[-2], shape_[-1]))
        output = (input_*z) + b
        return output


class RnadNetwork(PytorchNetwork):
    def __init__(self, shape: tuple[int,int] | tuple[int,int,int], n_actions: int, fc_dims=512, blocks=3) -> None:
        super().__init__()
        if len(shape) > 2:
            filters = 128
            self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(blocks)])
            self._shared = nn.Sequential(
                nn.Conv2d(shape[0], filters, 3, 1, 1),
                *self._blocks)
            self._pi_head = nn.Sequential(
                nn.Conv2d(filters, filters, 3, 1, 1),
                nn.Flatten(),
                nn.Linear(shape[1]*shape[2]*filters, fc_dims),
                nn.ReLU(),
                nn.Linear(fc_dims, n_actions))
            self._v_head = nn.Sequential(
                nn.Conv2d(filters, filters, 3, 1, 1),
                nn.Flatten(),
                nn.Linear(shape[1]*shape[2]*filters, fc_dims),
                nn.ReLU(),
                nn.Linear(fc_dims, 1),)
        else:

            self._blocks = nn.ModuleList(
                [LinearBlock(fc_dims) for _ in range(blocks)])
            self._shared = nn.Sequential(
                nn.Flatten(),
                nn.Linear(shape[0]*shape[1], fc_dims),
                nn.ReLU(),
                *self._blocks)
            self._pi_head = nn.Sequential(
                nn.Linear(fc_dims, fc_dims),
                nn.ReLU(),
                nn.Linear(fc_dims, n_actions),
            )
            self._v_head = nn.Sequential(
                nn.Linear(fc_dims, fc_dims),
                nn.ReLU(),
                nn.Linear(fc_dims, 1)
            )
        self._shared.to("cuda:0" if T.cuda.is_available() else "cpu")
        self._pi_head.to("cuda:0" if T.cuda.is_available() else "cpu")
        self._v_head.to("cuda:0" if T.cuda.is_available() else "cpu")

    def forward(self, state: T.Tensor) -> tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        shared: T.Tensor = self._shared(state)
        logits: T.Tensor = self._pi_head(shared)
        v = self._v_head(shared)
        probs = logits.softmax(dim=-1)
        return probs, v, probs.log(), logits


class StateActionResNetwork(StateActionNetwork):
    def __init__(self,
                 shape: tuple,
                 n_actions:int,
                 filters=128,
                 fc_dims=512,
                 n_blocks=3):

        super().__init__()
        self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(n_blocks)])

        if shape[1] == 64:
            self._shared = nn.Sequential(
                nn.Conv2d(shape[0], filters, 8, 4, 0),
                *self._blocks)
            shape = shape[0], 15, 13
        else:
            self._shared = nn.Sequential(
            nn.Conv2d(shape[0], filters, 3, 1, 1),
            *self._blocks)

        self._advantages_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions),)
        self._value_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1),
        )

        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self._blocks.to(device)
        self._shared.to(device)
        self._advantages_head.to(device)
        self._value_head.to(device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        shared: T.Tensor = self._shared(state)
        advs :T.Tensor = self._advantages_head(shared)
        advs = advs - advs.mean()
        value: T.Tensor = self._value_head(shared)
        qsa = value + advs
        return qsa

    def evaluate(self, observation: T.Tensor) -> T.Tensor:
        return self(observation)

class StateActionLinearNetwork(StateActionNetwork):
    def __init__(self, shape: tuple, n_actions: int, fc_dims=512, blocks=3) -> None:
        super().__init__()

        self._blocks = nn.ModuleList(
            [LinearBlock(fc_dims) for _ in range(blocks)])

        self._shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(shape[0]*shape[1], fc_dims),
            nn.ReLU(),
            *self._blocks,
            nn.Linear(fc_dims, fc_dims),
            nn.ReLU())
        self._advs_head = nn.Linear(fc_dims, n_actions)
        self._v_head = nn.Sequential(nn.Linear(fc_dims,1))

        self._device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self._shared.to(self._device)
        self._advs_head.to(self._device)
        self._v_head.to(self._device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        shared: T.Tensor = self._shared(state)
        advs :T.Tensor= self._advs_head(shared)
        advs = advs - advs.mean(dim=-1,keepdim=True)
        v:T.Tensor = self._v_head(shared)
        qsa = v + advs
        return qsa

    def evaluate(self, observation: T.Tensor) -> T.Tensor:
        return self(observation)