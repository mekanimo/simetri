"""Modifiers change each new element of a transformation with repetitions. Only Group objects
can have modifiers. If we need to modify a single shape object then we can create a group
with only one element and assign modifiers to this group.

A modifier function can have zero or any number of arguments. This function is called
during the repetitions of a transformation.
Zero arguments: Nothing is passed.
One argument: The element being transformed is passed.
Two arguments: The element and the modifier object is passed.
Three or more arguments and/or kwargs: The element, modifier, *modifier.args, and **modifier.kwargs



Examples:
    >>> import simetri.graphics as sg
    >>> def rotator(element, modifier, mult=1):
    ...     element.rotate(mult * sg.pi / 2, about=element.center)
    >>> def painter(element):
    ...     element.fill_color = sg.change_lightness(
    ...         element.fill_color, -0.2
    ...     )
    >>> square = sg.reg_poly_shape(4, 40)
    >>> square.fill_color = sg.gold
    >>> group = sg.Group([square])
    >>> group.modifiers = [
    ...     sg.Modifier(rotator, life_span=1),
    ...     sg.Modifier(painter),
    ... ]
    >>> group.translate(80, 0, reps=15)
"""

from __future__ import annotations

import inspect
import random
from collections.abc import Callable, Sequence
from typing import Any

from ..base.all_enums import Control, State


class Modifier:
    """Change elements created by a transformation with repetitions.

    Put modifiers on a group (``group.modifiers``). A transform with
    ``reps`` copies each element, applies the transform, then calls
    each modifier on that copy. A one-argument function receives the
    element, as ``painter`` does. A function with further parameters
    receives the element and this modifier, then the arguments stored
    on the *modifier.args and **modifier.kwargs.

    Attributes:
        function (callable): Called on each transformed element.
        life_span (int): How many transformed elements this modifier
            still accepts.
        randomness (float or callable): Whether the modifier is applied.
        float is the percent chance that it will be applied and callable must return
        True or False, if True then the modifier is applied
        condition (bool or callable): Whether the modifier is applied.
        state (State): The current state of the modifier. 'INITIAL', 'RUNNING', or 'STOPPED'.
        count (int): How many times this modifier has run.
        args (tuple): Extra positional arguments for the function.
        kwargs (dict): Extra keyword arguments for the function.

    Examples:
        >>> import simetri.graphics as sg
        >>> def rotator(element, modifier):
        ...     element.rotate(sg.pi / 2, about=element.center)
        >>> square = sg.reg_poly_shape(4, 40)
        >>> mod = sg.Modifier(rotator, life_span=2)
        >>> mod.state
        <State.INITIAL: 'INITIAL'>
        >>> squares = sg.Group(square)
        >>> squares.modifiers = [mod]
        >>> squares.translate(square.width, 0, reps=4)
        >>> mod.count
        2
    """

    def __init__(
        self,
        function: Callable[..., Any],
        life_span: int | Callable[..., Any] = 10000,
        randomness: float | Callable[..., Any] | Sequence[Any] = 1.0,
        condition: bool | Callable[..., Any] = True,
        *args: object,
        seed: int | None = None,
        **kwargs: object,
    ) -> None:
        """
        Args:
            function (callable): Called on each transformed element.
            life_span (int or callable, optional): How many times the modifier is applied.
            randomness (float or callable, optional): Possibility of modifier being applied.
            condition (bool or callable, optional): Condition to apply the modification. Defaults to True.
            *args: Additional arguments for the function.
            seed (int, optional): Seed for a local RNG used by randomness checks.
                Defaults to None.
            **kwargs: Additional keyword arguments for the function.
        """
        self.function = function  # it can be a list of functions
        signature = inspect.signature(function)
        self.n_func_args = len(signature.parameters)
        self.life_span = life_span
        self._rng = random.Random(seed)
        self.randomness = randomness
        self.condition = condition
        self.state = State.INITIAL
        self._d_state = {
            Control.INITIAL: State.INITIAL,
            Control.STOP: State.STOPPED,
            Control.PAUSE: State.PAUSED,
            Control.RESUME: State.RUNNING,
            Control.RESTART: State.RESTARTING,
        }
        self.active = True
        self.count = 0
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:
        """Returns a string representation of the Modifier object.

        Returns:
            str: String representation of the Modifier object.

        Examples:
            >>> import simetri.graphics as sg
            >>> def rotator(element, modifier):
            ...     element.rotate(sg.pi / 2, about=element.center)
            >>> mod = sg.Modifier(rotator, life_span=2)
            >>> 'lifespan:2' in repr(mod)
            True
            >>> 'randomness:1.0' in repr(mod)
            True
            >>> repr(mod).startswith('Modifier(function:')
            True
        """
        return (
            f"Modifier(function:{self.function}, lifespan:{self.life_span},"
            f"randomness:{self.randomness})"
        )

    def __str__(self) -> str:
        """Returns a string representation of the Modifier object.

        Returns:
            str: String representation of the Modifier object.

        Examples:
            >>> import simetri.graphics as sg
            >>> def rotator(element, modifier):
            ...     element.rotate(sg.pi / 2, about=element.center)
            >>> mod = sg.Modifier(rotator, life_span=2)
            >>> str(mod) == repr(mod)
            True
        """
        return self.__repr__()

    def set_state(self, control: Control) -> None:
        """Sets the state of the modifier based on the control value.

        Args:
            control (Control): The control value to set the state.

        Examples:
            >>> import simetri.graphics as sg
            >>> def rotator(element, modifier):
            ...     element.rotate(sg.pi / 2, about=element.center)
            >>> mod = sg.Modifier(rotator)
            >>> mod.set_state(sg.Control.STOP)
            >>> mod.state
            <State.STOPPED: 'STOPPED'>
            >>> mod.set_state(sg.Control.RESUME)
            >>> mod.state
            <State.RUNNING: 'RUNNING'>
        """
        self.state = self._d_state[control]

    def get_value(
        self, obj: object, target: object, *args: object, **kwargs: object
    ) -> Any:
        """Gets the value from an object or callable.

        Args:
            obj (object or callable): The object or callable to get the value from.
            target (object): The target object.
            *args: Additional arguments for the callable.
            **kwargs: Additional keyword arguments for the callable.

        Returns:
            object: The value obtained from the object or callable.

        Examples:
            >>> import simetri.graphics as sg
            >>> def painter(element):
            ...     element.fill_color = sg.change_lightness(
            ...         element.fill_color, -0.2
            ...     )
            >>> mod = sg.Modifier(painter)
            >>> mod.get_value(3, None)
            3
            >>> def halt(target):
            ...     return sg.Control.STOP
            >>> mod.get_value(halt, None)
            <Control.STOP: 'STOP'>
            >>> mod.state
            <State.STOPPED: 'STOPPED'>
        """
        if callable(obj):
            res = obj(target, *args, **kwargs)
            if res in Control:
                self.set_state(res)
        else:
            res = obj
        return res

    def apply(self, element: object) -> Any | None:
        """Applies the modifier to an element.

        Called on each copy made by a transformation with repetitions.
        A one-argument function receives the element, as ``painter`` does.
        A function with further parameters receives the element, this
        modifier, and the arguments stored on the modifier, as
        ``rotator(element, modifier, mult=1)`` does. If the function
        returns a control value, that value is applied to this modifier.
        ``Control.STOP``, ``Control.PAUSE``, ``Control.RESUME``, and
        ``Control.RESTART`` are the control values.

        Args:
            element (object): The transformed element (mutated when the
                modifier function changes it).

        Returns:
            object | None: The function result, or ``None`` if the modifier
            does not run.

        Examples:
            >>> import simetri.graphics as sg
            >>> def painter(element):
            ...     element.fill_color = sg.change_lightness(
            ...         element.fill_color, -0.2
            ...     )
            >>> square = sg.reg_poly_shape(4, 40)
            >>> square.fill_color = sg.gold
            >>> sg.Modifier(painter).apply(square)
            >>> painted = sg.change_lightness(sg.gold, -0.2)
            >>> square.fill_color == painted
            True
            >>> def rotator(element, modifier):
            ...     element.rotate(sg.pi / 2, about=element.center)
            ...     return modifier.count + 1
            >>> square = sg.reg_poly_shape(4, 40)
            >>> mod = sg.Modifier(rotator, life_span=2)
            >>> mod.apply(square)
            1
            >>> mod.apply(square)
            2
            >>> mod.apply(square)
            >>> tuple(round(c, 6) for c in square.vertices[0][:2])
            (-40.0, 0.0)
        """
        if self.active and self.can_continue(element):
            if self.n_func_args == 1:
                res = self.function(element)
            else:
                res = self.function(element, self, *self.args, **self.kwargs)
            self._update_state()
            return res

    def can_continue(self, target: object) -> bool:
        """Checks if the modifier can continue to be applied.

        Args:
            target (object): The target object.

        Returns:
            bool: True if the modifier can continue, False otherwise.

        Examples:
            >>> import simetri.graphics as sg
            >>> def rotator(element, modifier):
            ...     element.rotate(sg.pi / 2, about=element.center)
            >>> square = sg.reg_poly_shape(4, 40)
            >>> mod = sg.Modifier(rotator, life_span=2)
            >>> mod.can_continue(square)
            True
            >>> sg.Modifier(
            ...     rotator, life_span=2, randomness=0.0
            ... ).can_continue(square)
            False
        """
        if callable(self.randomness):
            randomness = self.get_value(self.randomness, target)
        elif isinstance(self.randomness, float):
            rand_val = self._rng.random()
            randomness = self.randomness >= rand_val
        elif isinstance(self.randomness, (list, tuple)):
            randomness = self._rng.choice(self.randomness)

        if callable(self.condition):
            condition = self.get_value(self.condition, target)
        else:
            condition = self.condition

        if callable(self.life_span):
            life_span = self.get_value(self.life_span, target)
        else:
            life_span = self.life_span

        if life_span > 0 and condition and randomness:
            if self.state in (State.INITIAL, State.RUNNING, State.RESTARTING):
                res = True
            else:
                res = False
        else:
            res = False
        return res

    def _update_state(self) -> None:
        """Updates the state of the modifier based on its life span and count."""
        self.count += 1
        if self.count == 1:
            self.state = State.RUNNING
        if self.life_span > 0:
            if self.state == State.RESTARTING:
                self.state = State.RUNNING
            elif self.state == State.RUNNING:
                self.life_span -= 1
                if self.life_span == 0:
                    self.state = State.STOPPED
        else:
            self.state = State.STOPPED

    def stop(self) -> None:
        """Stops the modifier.

        Examples:
            >>> import simetri.graphics as sg
            >>> def painter(element):
            ...     element.fill_color = sg.change_lightness(
            ...         element.fill_color, -0.2
            ...     )
            >>> mod = sg.Modifier(painter)
            >>> mod.stop()
            >>> mod.state
            <State.STOPPED: 'STOPPED'>
        """
        self.state = State.STOPPED
