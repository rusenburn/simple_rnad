from abc import ABC,abstractmethod
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

    def load_model(self, path: str ) -> None:
        try:
            self.load_state_dict(T.load(path))
            print(f'The nn was loaded from {path}')
        except:
            print(f'could not load nn from {path}')



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
        probs: T.Tensor = pi.softmax(dim=-1)
        return probs

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
        pi = pi.clamp(-2.0,2.0)
        probs: T.Tensor = pi.softmax(dim=-1)
        return probs

class CriticResNetwork(PytorchNetwork):
    def __init__(self,
                 shape: tuple,
                 filters=128,
                 fc_dims=512,
                 n_blocks=3):

        super().__init__()
        self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(n_blocks)])

        self._shared = nn.Sequential(
            nn.Conv2d(shape[0], filters, 3, 1, 1),
            *self._blocks)

        self._value_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 1),
            nn.Tanh()
            )
            
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self._blocks.to(device)
        self._shared.to(device)
        self._value_head.to(device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        shared: T.Tensor = self._shared(state)
        value: T.Tensor = self._value_head(shared)
        return value
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