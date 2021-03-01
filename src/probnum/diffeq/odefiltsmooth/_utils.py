"""Utility functions for ODE filtering and smoothing."""

__all__ = ["convert_derivwise_to_coordwise", "convert_coordwise_to_derivwise"]


def convert_derivwise_to_coordwise(arr, ordint):
    """Utility function to change ordering of elements in stacked vector from ``(y1, y2,
    dy1, dy2, ddy1, ddy2, ...))``.

    to ``(y1, dy1, ddy1, ..., y2, dy2, ddy2, ...)``.
    """
    return arr.reshape((ordint + 1, -1)).T.flatten()


def convert_coordwise_to_derivwise(arr, ordint):
    """Utility function to change ordering of elements in stacked vector from ``(y1, y2,
    dy1, dy2, ddy1, ddy2, ...))``.

    to ``(y1, dy1, ddy1, ..., y2, dy2, ddy2, ...)``.
    """
    return arr.reshape((-1, ordint + 1)).T.flatten()
