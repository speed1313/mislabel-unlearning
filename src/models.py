from flax import linen as nn


class CNN(nn.Module):
    """A simple CNN model."""

    class_num: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.class_num)(x)
        return x


class MLP(nn.Module):
    """A simple MLP model."""

    class_num: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.class_num)(x)
        x = nn.log_softmax(x)
        return x
