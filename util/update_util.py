#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf


def clip_gradient_norms(gradients_to_variables, max_norm):
    """Clips the gradients by the given value.

    Args:
      gradients_to_variables: A list of gradient to variable pairs (tuples).
      max_norm: the maximum norm value.

    Returns:
      A list of clipped gradient to variable pairs.
    """
    clipped_grads_and_vars = []
    for grad, var in gradients_to_variables:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.clip_by_norm(grad.values, max_norm)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_norm(grad, max_norm)
        clipped_grads_and_vars.append((grad, var))
    return clipped_grads_and_vars


def multiply_gradients(grads_and_vars, gradient_multipliers):
    """Multiply specified gradients.

    Args:
      grads_and_vars: A list of gradient to variable pairs (tuples).
      gradient_multipliers: A map from either `Variables` or `Variable` op names
        to the coefficient by which the associated gradient should be scaled.

    Returns:
      The updated list of gradient to variable pairs.

    Raises:
      ValueError: If `grads_and_vars` is not a list or if `gradient_multipliers is empty or None or not a dictionary.
    """
    if not isinstance(grads_and_vars, list):
        raise ValueError('`grads_and_vars` must be a list.')
    if not gradient_multipliers:
        raise ValueError('`gradient_multipliers` is empty.')
    if not isinstance(gradient_multipliers, dict):
        raise ValueError('`gradient_multipliers` must be a dict.')

    multiplied_grads_and_vars = []
    for grad, var in grads_and_vars:
        if var in gradient_multipliers or var.op.name in gradient_multipliers:
            key = var if var in gradient_multipliers else var.op.name
            if grad is None:
                raise ValueError('Requested multiple of `None` gradient.')

            multiplier = gradient_multipliers[key]
            if not isinstance(multiplier, tf.Tensor):
                multiplier = tf.constant(multiplier, dtype=grad.dtype)

            if isinstance(grad, tf.IndexedSlices):
                tmp = grad.values * multiplier
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad *= multiplier
        multiplied_grads_and_vars.append((grad, var))
    return multiplied_grads_and_vars
