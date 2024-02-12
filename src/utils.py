import jax
import jax.numpy as jnp
import numpy as np
import optax


def loss_fn_for_hess(last_params, other_params, images, labels, class_num, state):
    last_params = last_params.reshape((-1, class_num))
    logits = state.apply_fn(
        {
            "params": {
                "Dense_0": other_params["Dense_0"],
                "Dense_1": {
                    "kernel": last_params,
                    "bias": other_params["Dense_1"]["bias"],
                },
            }
        },
        images,
    )
    one_hot = jax.nn.one_hot(labels, class_num)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss


def influence_up_loss(x, y, x_test, y_test, train_ds, loss_fn, state, class_num):
    """
    Compute I_up_loss(z, z_test)
    """
    x = x.reshape((1, 28, 28, 1))
    y = jnp.expand_dims(y, axis=0)
    if y_test.ndim == 0:
        x_test = x_test.reshape((1, 28, 28, 1))
        y_test = jnp.expand_dims(y_test, axis=0)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, _), upweight_grad = grad_fn(state.params, x, y)
    last_layer_upweight_grad = upweight_grad["Dense_1"]["kernel"].reshape((-1, 1))

    (_, _), test_grad = grad_fn(state.params, x_test, y_test)
    last_layer_test_grad = test_grad["Dense_1"]["kernel"].reshape((-1, 1))

    hessian_last_layer = jax.hessian(loss_fn_for_hess, argnums=0)(
        state.params["Dense_1"]["kernel"].reshape(-1),
        state.params,
        train_ds["image"],
        train_ds["label"],
        class_num,
        state,
    ).reshape(
        (
            state.params["Dense_1"]["kernel"].shape[0] * class_num,
            state.params["Dense_1"]["kernel"].shape[0] * class_num,
        )
    )
    last_layer_hessian_inv = np.linalg.pinv(hessian_last_layer)
    influence = jnp.einsum(
        "ij,jk,ki->",
        last_layer_test_grad.T,
        last_layer_hessian_inv,
        last_layer_upweight_grad,
    )
    return influence


def I_up_params(x, y, forget_x, forget_y, loss_fn, state, CLASS_NUM):
    """
    Compute I_up_params(z)
    """

    hessian_last_layer = jax.hessian(loss_fn_for_hess, argnums=0)(
        state.params["Dense_1"]["kernel"].reshape(-1),
        state.params,
        x,
        y,
        CLASS_NUM,
        state,
    ).reshape(
        (
            state.params["Dense_1"]["kernel"].shape[0] * CLASS_NUM,
            state.params["Dense_1"]["kernel"].shape[0] * CLASS_NUM,
        )
    )

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, _), forget_grad = grad_fn(state.params, forget_x, forget_y)
    last_layer_forget_grad = forget_grad["Dense_1"]["kernel"].reshape((-1, 1))

    last_layer_hessian_inv = np.linalg.pinv(hessian_last_layer)
    print(last_layer_hessian_inv.shape, last_layer_forget_grad.shape)
    I_up_params_value = -jnp.einsum(
        "ij, jk->ik", last_layer_hessian_inv, last_layer_forget_grad
    )

    return I_up_params_value.reshape((-1, CLASS_NUM))


def Newton_update(remain_x, remain_y, forget_x, forget_y, loss_fn, state, CLASS_NUM):
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, _), grad = grad_fn(state.params, remain_x, remain_y)

    hessian_last_layer = jax.hessian(loss_fn_for_hess, argnums=0)(
        state.params["Dense_1"]["kernel"].reshape(-1),
        state.params,
        remain_x,
        remain_y,
        CLASS_NUM,
        state,
    ).reshape(
        (
            state.params["Dense_1"]["kernel"].shape[0] * CLASS_NUM,
            state.params["Dense_1"]["kernel"].shape[0] * CLASS_NUM,
        )
    )

    last_layer_hessian_inv = np.linalg.pinv(hessian_last_layer)

    (_, _), grad = grad_fn(state.params, forget_x, forget_y)
    last_layer_forget_grad = grad["Dense_1"]["kernel"].reshape((-1, 1)) * len(forget_x)
    newton_update = jnp.einsum(
        "ij, jk->ik", last_layer_hessian_inv, last_layer_forget_grad
    ) / len(remain_x)

    return newton_update.reshape((-1, CLASS_NUM))
