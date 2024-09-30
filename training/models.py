import torch
from torch import softmax
from torch.nn import (
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    LeakyReLU,
    Sequential,
)
from torch.nn.init import kaiming_normal_


class CNN(Module):
    def __init__(self, n_dim, linear_dim):
        super(CNN, self).__init__()
        self.n_dim = n_dim
        self.linear_dim = linear_dim

        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=4, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(4, 8, kernel_size=4, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(),
            MaxPool2d(kernel_size=2),
            # Defining another 2D convolution layer
            Conv2d(8, 16, kernel_size=4, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(16, 32, kernel_size=4, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
        )

        self.linear_layers = Sequential(Linear(self.linear_dim, 2, bias=False))

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = softmax(self.linear_layers(x), dim=1)
        return x


class LLR(Module):
    def __init__(self, n_dim):
        super(LLR, self).__init__()
        self.linear = Linear(n_dim, 2, bias=False)

    # Defining the forward pass
    def forward(self, x):
        x = softmax(self.linear(x), dim=1)
        return x


class MLP(Module):
    def __init__(self, n_dim):
        super(MLP, self).__init__()
        self.n_dim = n_dim

        self.linear_layers = Sequential(
            Linear(self.n_dim, int(256), bias=False),
            ReLU(),
            Dropout(p=0.3),
            Linear(int(256), int(256), bias=False),
            ReLU(),
            Dropout(p=0.3),
            Linear(int(256), int(64), bias=False),
            ReLU(),
            Dropout(p=0.3),
            Linear(int(64), 2, bias=False),
        )

    # Defining the forward pass
    def forward(self, x):
        x = softmax(self.linear_layers(x), dim=1)
        return x


class CNN8by8(Module):
    def __init__(self, n_dim, linear_dim):
        super(CNN8by8, self).__init__()
        self.n_dim = n_dim
        self.linear_dim = linear_dim
        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=2, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=2, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=2, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 4, kernel_size=2, stride=1, padding=1),
            # BatchNorm2d(4),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(Linear(self.linear_dim, 2))
        self.softmax = torch.nn.Softmax(dim=-1)

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = softmax(self.linear_layers(x), dim=1)
        return x

    def get_layers(self):
        return (
            list(self.cnn_layers.children())
            + list(self.linear_layers.children())
            + [self.softmax]
        )

    def collect_activations(self, x):
        yield x, 0, "input"
        layers = self.get_layers()
        cur = x
        with torch.no_grad():
            for idx, layer in enumerate(layers):
                cur = layer(cur)
                yield cur, idx + 1, str(layer)


class MLP8by8(Module):
    def __init__(self, n_dim, layers=None):
        super(MLP8by8, self).__init__()
        self.n_dim = n_dim

        if layers is None:
            self.linear_layers = Sequential(
                Linear(self.n_dim, int(self.n_dim / 2)),
                ReLU(),
                Linear(int(self.n_dim / 2), int(self.n_dim / 4)),
                ReLU(),
                Linear(int(self.n_dim / 4), int(self.n_dim / 8)),
                ReLU(),
                Linear(int(self.n_dim / 8), 2),
            )
        else:
            l = []
            for idx, layer in enumerate(layers):
                if idx == 0:
                    l.append(Linear(self.n_dim, layer))
                else:
                    l.append(Linear(layers[idx - 1], layer))
                l.append(ReLU())
                l.append(torch.nn.BatchNorm1d(layer))

            l.append(Linear(layers[-1], 2))
            self.linear_layers = Sequential(*l)

    # Defining the forward pass
    def forward(self, x):
        x = self.linear_layers(x)
        return x

    def get_layers(self):
        return list(self.linear_layers.children()) + [torch.nn.Softmax(dim=-1)]

    def collect_activations(self, x):
        yield x, 0, "input"
        layers = self.get_layers()
        cur = x
        with torch.no_grad():
            for idx, layer in enumerate(layers):
                cur = layer(cur)
                yield cur, idx + 1, str(layer)


def init_he_normal(layer):
    if isinstance(layer, Conv2d) or isinstance(layer, Linear):
        kaiming_normal_(layer.weight)


models_dict = {
    "CNN": CNN,
    "LLR": LLR,
    "MLP": MLP,
    "CNN8by8": CNN8by8,
    "MLP8by8": MLP8by8,
}
