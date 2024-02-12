import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import torchvision
from flax.training import train_state
import models
import utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import os
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--noise_ratio", type=float, default=0.2)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--hidden_size", type=int, default=50)
parser.add_argument("--sample_per_class", type=int, default=1000)
parser.add_argument(
    "--mode", type=str, default="loss"
)  # about z_test: loss, one, leave_one_out, all
args = parser.parse_args()


CLASS_NUM = 2
HIDDEN_SIZE = 50

sample_per_class = args.sample_per_class
noise_ratio = args.noise_ratio
flip_sample_num = int(sample_per_class * noise_ratio)


config = ml_collections.ConfigDict()
config.learning_rate = args.learning_rate
config.momentum = args.momentum
config.batch_size = args.batch_size
config.num_epochs = args.num_epochs

# I_up_loss mode
mode = args.mode

dir = f"result/detection/{noise_ratio}/"

if not os.path.exists(dir):
    os.makedirs(dir)


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, CLASS_NUM)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds["image"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def mnist_transform(x):
    return np.expand_dims(np.array(x, dtype=np.float32), axis=2) / 255.0


def mnist_collate_fn(batch):
    batch = list(zip(*batch))
    x = np.stack(batch[0])
    y = np.array(batch[1])
    return x, y


def get_dataset_torch():
    mnist = {
        "train": torchvision.datasets.MNIST("./data", train=True, download=True),
        "test": torchvision.datasets.MNIST("./data", train=False, download=True),
    }

    ds = {}

    for split in ["train", "test"]:
        # only 0 and 1
        idx_list = []
        for class_number in range(CLASS_NUM):
            idx = np.where(mnist[split].targets == class_number)[0][:sample_per_class]
            idx_list.append(idx)
        idx = np.concatenate(idx_list)
        ds[split] = {
            "image": mnist[split].data[idx],
            "label": mnist[split].targets[idx],
        }

        ds[split]["image"] = mnist_transform(ds[split]["image"])
        ds[split]["label"] = np.array(ds[split]["label"], dtype=np.int32)
        # expand dim
        ds[split]["image"] = np.expand_dims(ds[split]["image"], axis=3)
    return ds["train"], ds["test"]


def add_noise(ds):
    true_labels = ds["label"].copy()
    idx_each_class = []
    for i in range(CLASS_NUM):
        idx = np.where(ds["label"] == i)[0]
        idx_each_class.append(idx)
    idx_each_class = np.array(idx_each_class)
    for i in range(CLASS_NUM):
        for j in range(flip_sample_num):
            target_label = np.random.randint(0, CLASS_NUM)
            while target_label == i:
                target_label = np.random.randint(0, CLASS_NUM)
            ds["label"][idx_each_class[i][j]] = target_label
    return ds, true_labels


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    model = models.MLP(class_num=CLASS_NUM, hidden_size=HIDDEN_SIZE)
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str, train_ds, test_ds
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    rng = jax.random.key(1)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, train_accuracy = train_epoch(
            state, train_ds, config.batch_size, input_rng
        )

        _, test_loss, test_accuracy = apply_model(
            state, test_ds["image"], test_ds["label"]
        )

        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f,"
            " test_accuracy: %.2f"
            % (
                epoch,
                train_loss,
                train_accuracy * 100,
                test_loss,
                test_accuracy * 100,
            )
        )
    # accuracy per class
    for i in range(CLASS_NUM):
        idx = np.where(test_ds["label"] == i)[0]
        _, test_loss, test_accuracy = apply_model(
            state, test_ds["image"][idx], test_ds["label"][idx]
        )
        print(f"test_accuracy for class {i}: {test_accuracy * 100}")

    return state


def show_accuracy_per_class(state, train_ds, test_ds):
    train_accuracy_list = []
    test_accuracy_list = []
    for i in range(CLASS_NUM):
        idx = np.where(train_ds["label"] == i)[0]
        _, train_loss, train_accuracy = apply_model(
            state, train_ds["image"][idx], train_ds["label"][idx]
        )

        print(f"train_accuracy for class {i}: {train_accuracy * 100}")
        train_accuracy_list.append(train_accuracy)
        idx = np.where(test_ds["label"] == i)[0]
        _, test_loss, test_accuracy = apply_model(
            state, test_ds["image"][idx], test_ds["label"][idx]
        )
        print(f"test_accuracy for class {i}: {test_accuracy * 100}")
        test_accuracy_list.append(test_accuracy)
    return train_accuracy_list, test_accuracy_list


def loss_fn(params, images, labels):
    logits = state.apply_fn({"params": params}, images)
    one_hot = jax.nn.one_hot(labels, CLASS_NUM)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits


# train with noisy data
train_ds, test_ds = get_dataset_torch()
noisy_train_ds, true_train_labels = add_noise(train_ds)
state = train_and_evaluate(config, "./", noisy_train_ds, test_ds)
train_num = len(train_ds["image"])
mislabeled_num = np.sum(noisy_train_ds["label"] != true_train_labels)
clean_num = train_num - mislabeled_num


# mislabel detection using Influence function(I_up_loss(z,z)
anomaly_list = []
for i, (one_train_data, one_train_label) in enumerate(
    zip(noisy_train_ds["image"], noisy_train_ds["label"])
):
    if mode == "loss":
        one_train_data = one_train_data.reshape((1, 28, 28, 1))
        one_train_label = jnp.expand_dims(one_train_label, axis=0)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params, one_train_data, one_train_label)
        anomaly_list.append(loss)
    else:
        leave_one_out_train_ds = {
            "image": np.delete(noisy_train_ds["image"], i, axis=0),
            "label": np.delete(noisy_train_ds["label"], i, axis=0),
        }
        if mode == "one":
            influence = utils.influence_up_loss(
                one_train_data,
                one_train_label,
                one_train_data,  # noisy_train_ds["image"],
                one_train_label,  # noisy_train_ds["label"],
                noisy_train_ds,
                loss_fn,
                state,
                CLASS_NUM,
            )
        elif mode == "all":
            influence = utils.influence_up_loss(
                one_train_data,
                one_train_label,
                noisy_train_ds["image"],
                noisy_train_ds["label"],
                noisy_train_ds,
                loss_fn,
                state,
                CLASS_NUM,
            )
        elif mode == "leave_one_out":
            influence = utils.influence_up_loss(
                one_train_data,
                one_train_label,
                leave_one_out_train_ds["image"],
                leave_one_out_train_ds["label"],
                noisy_train_ds,
                loss_fn,
                state,
                CLASS_NUM,
            )

        anomaly_list.append(influence)

# show top influence data
anomaly_list = np.array(anomaly_list)
anomaly_list = np.squeeze(anomaly_list)
idx = np.argsort(anomaly_list)[::-1]
# top_k_percent = noise_ratio
top_k_percent = 0.1
threshold = np.quantile(anomaly_list, 1 - top_k_percent)

noisy_sample_anomaly_list = anomaly_list[noisy_train_ds["label"] != true_train_labels]
plt.hist(
    noisy_sample_anomaly_list,
    bins=100,
    color="red",
    alpha=0.5,
    label="mislabeled samples",
)
correct_sample_anomaly_list = anomaly_list[noisy_train_ds["label"] == true_train_labels]
plt.hist(
    correct_sample_anomaly_list,
    bins=100,
    color="blue",
    alpha=0.5,
    label="clean samples",
)
plt.xlabel("anomaly score")
plt.ylabel("frequency")
plt.ylim(0, 50)
plt.title("anomaly score distribution")
plt.axvline(x=threshold, color="green", label=f"threshold (top {top_k_percent*100}%)")
plt.legend()
path = dir + f"{mode}_{args.num_epochs}.png"
plt.savefig(path)
plt.show(block=False)


y_score = sorted(anomaly_list)
y_true = (
    noisy_train_ds["label"][np.argsort(anomaly_list)]
    != true_train_labels[np.argsort(anomaly_list)]
)  # true if mislabeled


auc = roc_auc_score(y_true, y_score)
accuracy = accuracy_score(y_true, y_score >= threshold)
precision = precision_score(y_true, y_score >= threshold)
recall = recall_score(y_true, y_score >= threshold)
f1 = f1_score(y_true, y_score >= threshold)
print("result")
print("auc, accuracy, precision, recall, f1")
print(f"{auc}, {accuracy}, {precision}, {recall}, {f1}")

# write result to csv
path = dir + f"{mode}_{args.num_epochs}.csv"
with open(path, "w") as f:
    writer = csv.writer(f)
    # , to &
    writer.writerow(["auc", "accuracy", "precision", "recall", "f1"])
    writer.writerow([auc, accuracy, precision, recall, f1])
