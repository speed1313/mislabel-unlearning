# pytype: disable=wrong-keyword-args

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
parser.add_argument("--class_num", type=int, default=2)
parser.add_argument("--percentile", type=float, default=0.1)

args = parser.parse_args()


CLASS_NUM = args.class_num
HIDDEN_SIZE = args.hidden_size

sample_per_class = args.sample_per_class
noise_ratio = args.noise_ratio
flip_sample_num = int(sample_per_class * noise_ratio)


config = ml_collections.ConfigDict()
config.learning_rate = args.learning_rate
config.momentum = args.momentum
config.batch_size = args.batch_size
config.num_epochs = args.num_epochs
top_k_percent = args.percentile

FASHION = False
if FASHION:
    dir = f"result/pipeline/fashion/{noise_ratio}/{top_k_percent}/"
else:
    dir = f"result/pipeline/{noise_ratio}/{top_k_percent}/"

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
    if FASHION:
        mnist = {
            "train": torchvision.datasets.FashionMNIST(
                "./data", train=True, download=True
            ),
            "test": torchvision.datasets.FashionMNIST(
                "./data", train=False, download=True
            ),
        }
    else:
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
    config: ml_collections.ConfigDict, workdir: str, train_ds, test_ds, state=None
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
    if state is None:
        state = create_train_state(init_rng, config)
    else:
        state = state

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

# mislabel detection using loss
anomaly_list = []
for i, (one_train_data, one_train_label) in enumerate(
    zip(noisy_train_ds["image"], noisy_train_ds["label"])
):
    one_train_data = one_train_data.reshape((1, 28, 28, 1))
    one_train_label = jnp.expand_dims(one_train_label, axis=0)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params, one_train_data, one_train_label)
    anomaly_list.append(loss)
anomaly_list = np.array(anomaly_list)
anomaly_list = np.squeeze(anomaly_list)
loss_idx = np.argsort(anomaly_list)[::-1]


threshold = np.quantile(anomaly_list, 1 - top_k_percent)
detection_num = np.sum(anomaly_list > threshold)
print(f"detection_num: {detection_num}")


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
path = dir + f"{CLASS_NUM}_{HIDDEN_SIZE}_{args.num_epochs}_anomaly_score.png"
plt.savefig(path)
plt.show()


detected_noisy_data = noisy_train_ds["image"][loss_idx[:detection_num]]
detected_noisy_label = noisy_train_ds["label"][loss_idx[:detection_num]]
detected_clean_data = noisy_train_ds["image"][loss_idx[detection_num:]]
detected_clean_label = noisy_train_ds["label"][loss_idx[detection_num:]]
detected_noisy_train_ds = {"image": detected_noisy_data, "label": detected_noisy_label}
detected_clean_train_ds = {"image": detected_clean_data, "label": detected_clean_label}


noisy_data = noisy_train_ds["image"][noisy_train_ds["label"] != true_train_labels]
noisy_label = noisy_train_ds["label"][noisy_train_ds["label"] != true_train_labels]
clean_data = noisy_train_ds["image"][noisy_train_ds["label"] == true_train_labels]
clean_label = noisy_train_ds["label"][noisy_train_ds["label"] == true_train_labels]

original_params = state.params["Dense_1"]["kernel"].copy()


# original model
print("original model")
original_train_acc, original_test_acc = show_accuracy_per_class(
    state, noisy_train_ds, test_ds
)


# Unlearn using Influence function with noisy data
state.params["Dense_1"]["kernel"] = original_params
I_up_params_value = utils.I_up_params(
    noisy_train_ds["image"],
    noisy_train_ds["label"],
    noisy_data,
    noisy_label,
    loss_fn,
    state,
    CLASS_NUM,
)
state.params["Dense_1"]["kernel"] = state.params["Dense_1"]["kernel"] - (
    (I_up_params_value) * len(noisy_data)
) / len(train_ds["image"])
print("unlearning using Influence function with complete noise knowledge")
infleunce_noisy_train_acc, infleunce_noisy_test_acc = show_accuracy_per_class(
    state, noisy_train_ds, test_ds
)


state.params["Dense_1"]["kernel"] = original_params
I_up_params_value = utils.I_up_params(
    noisy_train_ds["image"],
    noisy_train_ds["label"],
    detected_noisy_data,
    detected_noisy_label,
    loss_fn,
    state,
    CLASS_NUM,
)
state.params["Dense_1"]["kernel"] = state.params["Dense_1"]["kernel"] - (
    (I_up_params_value) * len(detected_noisy_data)
) / len(train_ds["image"])
print("unlearning using Influence function with partial noise knowledge")
(
    detected_infleunce_noisy_train_acc,
    detected_infleunce_noisy_test_acc,
) = show_accuracy_per_class(state, noisy_train_ds, test_ds)


# unlearn using Newton's update
state.params["Dense_1"]["kernel"] = original_params
state.params["Dense_1"]["kernel"] = state.params["Dense_1"][
    "kernel"
] + utils.Newton_update(
    clean_data, clean_label, noisy_data, noisy_label, loss_fn, state, CLASS_NUM
)
print("unlearning using SSSE with complete noise knowledge")
ssse_noisy_train_acc, ssse_noisy_test_acc = show_accuracy_per_class(
    state, noisy_train_ds, test_ds
)


state.params["Dense_1"]["kernel"] = original_params
state.params["Dense_1"]["kernel"] = state.params["Dense_1"][
    "kernel"
] + utils.Newton_update(
    detected_clean_data,
    detected_clean_label,
    detected_noisy_data,
    detected_noisy_label,
    loss_fn,
    state,
    CLASS_NUM,
)
print("unlearning using SSSE with partial noise knowledge")
detected_ssse_noisy_train_acc, detected_ssse_noisy_test_acc = show_accuracy_per_class(
    state, noisy_train_ds, test_ds
)


clean_train_ds = {
    "image": noisy_train_ds["image"][noisy_train_ds["label"] == true_train_labels],
    "label": noisy_train_ds["label"][noisy_train_ds["label"] == true_train_labels],
}

# fine tuning the model using clean data
state.params["Dense_1"]["kernel"] = original_params
show_accuracy_per_class(state, clean_train_ds, test_ds)
config.epoch = 10
state = train_and_evaluate(config, "./", clean_train_ds, test_ds, state)
print("fine tuning the model using clean data")
fine_tune_train_acc, fine_tune_test_acc = show_accuracy_per_class(
    state, clean_train_ds, test_ds
)

state.params["Dense_1"]["kernel"] = original_params
config.epoch = 10
state = train_and_evaluate(config, "./", detected_clean_train_ds, test_ds, state)
print("fine tuning the model using partial clean data")
detected_fine_tune_train_acc, detected_fine_tune_test_acc = show_accuracy_per_class(
    state, clean_train_ds, test_ds
)


# retraining the model using data without noisy data
config.epoch = 20
state = train_and_evaluate(config, "./", clean_train_ds, test_ds)
print("retraining the model using clean data")
retrain_train_acc, retrain_test_acc = show_accuracy_per_class(
    state, clean_train_ds, test_ds
)

state = train_and_evaluate(config, "./", detected_clean_train_ds, test_ds)
print("retraining the model using partial clean data")
detected_retrain_train_acc, detected_retrain_test_acc = show_accuracy_per_class(
    state, clean_train_ds, test_ds
)


# show accuracy per class for each method
# plt.plot(original_train_acc, label="original_train_acc")
plt.plot(original_test_acc, label="original", color="black")
plt.plot(
    infleunce_noisy_test_acc,
    label="influence function",
    linestyle="dotted",
    color="red",
)
plt.plot(
    detected_fine_tune_test_acc,
    label="fine tune on $D_{detected}$",
    color="blue",
    linestyle="dotted",
    alpha=0.5,
)
plt.plot(
    detected_infleunce_noisy_test_acc,
    label="influence on $D_{detected}$",
    linestyle="dashed",
    color="red",
)
plt.plot(
    detected_ssse_noisy_test_acc,
    label="ssse on $D_{detected}$",
    linestyle="dashdot",
    color="green",
)
plt.plot(
    detected_retrain_test_acc,
    label="retrain on $D_{detected}$",
    color="orange",
    linestyle="dashed",
    alpha=0.5,
)

plt.plot(ssse_noisy_test_acc, label="ssse", linestyle="dashdot", color="green")
plt.plot(
    retrain_test_acc,
    label="retrain with clean samples",
    color="orange",
    linestyle="dashed",
    alpha=0.5,
)
plt.plot(
    fine_tune_test_acc, label="fine tune with clean samples", color="blue", alpha=0.5
)

plt.title("test accuracy")
plt.xlabel("class")
plt.xticks(np.arange(CLASS_NUM))
plt.ylabel("accuracy")

plt.legend()
path = dir + f"{CLASS_NUM}_{HIDDEN_SIZE}_{args.num_epochs}.png"
plt.savefig(path)
plt.show()

# result to csv

path = dir + f"{CLASS_NUM}_{HIDDEN_SIZE}_{args.num_epochs}.csv"
with open(path, "w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "original_test_acc",
            "infleunce_noisy_test_acc",
            "ssse_noisy_test_acc",
            "fine_tune_test_acc",
            "retrain_test_acc",
        ]
    )
    # class 0 accuracy
    writer.writerow(
        [
            original_test_acc[0],
            infleunce_noisy_test_acc[0],
            ssse_noisy_test_acc[0],
            fine_tune_test_acc[0],
            retrain_test_acc[0],
        ]
    )
    # class 1 accuracy
    writer.writerow(
        [
            original_test_acc[1],
            infleunce_noisy_test_acc[1],
            ssse_noisy_test_acc[1],
            fine_tune_test_acc[1],
            retrain_test_acc[1],
        ]
    )
    # average accuracy
    writer.writerow(
        [
            np.mean(original_test_acc),
            np.mean(infleunce_noisy_test_acc),
            np.mean(ssse_noisy_test_acc),
            np.mean(fine_tune_test_acc),
            np.mean(retrain_test_acc),
        ]
    )
    writer.writerow(
        [
            np.mean(original_test_acc),
            np.mean(detected_infleunce_noisy_test_acc),
            np.mean(detected_ssse_noisy_test_acc),
            np.mean(detected_fine_tune_test_acc),
            np.mean(detected_retrain_test_acc),
        ]
    )
