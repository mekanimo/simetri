"""Property modifiers applied over time to Group objects.

Examples:
    >>> import simetri.graphics as sg
    >>> def bump(element):
    ...     element['n'] += 1
    >>> target = {'n': 0}
    >>> mod = sg.Modifier(bump, life_span=2, seed=0)
    >>> mod.apply(target)
    >>> target['n']
    1
"""

from __future__ import annotations

import inspect
import random
from collections.abc import Callable, Sequence
from typing import Any

from ..base.all_enums import Control, State


class Modifier:
    """Used to modify the properties of a Group object.

    Attributes:
        function (callable): The function to modify the property.
        life_span (int): The number of times the modifier can be applied.
        randomness (float or callable): Determines the randomness of the modification.
        condition (bool or callable): Condition to apply the modification.
        state (State): The current state of the modifier.
        _d_state (dict): Mapping of control states to modifier states.
        count (int): Counter for the number of times the modifier has been applied.
        args (tuple): Additional arguments for the function.
        kwargs (dict): Additional keyword arguments for the function.

    Examples:
        >>> import simetri.graphics as sg
        >>> def bump(element):
        ...     element['n'] += 1
        >>> target = {'n': 0}
        >>> mod = sg.Modifier(bump, life_span=2, seed=0)
        >>> mod.state
        <State.INITIAL: 'INITIAL'>
        >>> mod.apply(target)
        >>> target['n']
        1
        >>> mod.state
        <State.RUNNING: 'RUNNING'>
        >>> mod.apply(target)
        >>> target['n']
        2
        >>> mod.state
        <State.STOPPED: 'STOPPED'>
        >>> mod.apply(target)
        >>> target['n']
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
            function (callable): The function to modify the property.
            life_span (int or callable, optional): The number of times the
                modifier can be applied. Defaults to 10000.
            randomness (float or callable, optional): Determines the randomness of the modification. Defaults to 1.0.
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
            >>> def bump(element):
            ...     element['n'] += 1
            >>> mod = sg.Modifier(bump, life_span=2, seed=0)
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
            >>> def bump(element):
            ...     element['n'] += 1
            >>> mod = sg.Modifier(bump, life_span=2, seed=0)
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
            >>> def bump(element):
            ...     element['n'] += 1
            >>> mod = sg.Modifier(bump, seed=0)
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
            >>> def bump(element):
            ...     element['n'] += 1
            >>> mod = sg.Modifier(bump, seed=0)
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

        If a function returns a control value, it will be applied to the modifier.
        Control.STOP, Control.PAUSE, Control.RESUME, and Control.RESTART are the only control values.
        Functions should have the following signature:
        def funct(target, modifier, *args, **kwargs):

        Args:
            element (object): The element to apply the modifier to.

        Returns:
            object | None: The function result, or ``None`` if the modifier
            does not run.

        Examples:
            >>> import simetri.graphics as sg
            >>> def bump(element):
            ...     element['n'] += 1
            >>> target = {'n': 0}
            >>> mod = sg.Modifier(bump, life_span=2, seed=0)
            >>> mod.apply(target)
            >>> target['n']
            1
            >>> def counted(element, modifier):
            ...     element['n'] += 1
            ...     return element['n']
            >>> target = {'n': 0}
            >>> mod = sg.Modifier(counted, life_span=2, seed=0)
            >>> mod.apply(target)
            1
            >>> mod.apply(target)
            2
            >>> mod.apply(target)
            >>> target['n']
            2
        """
        if self.active and self.can_continue(element):
            if self.n_func_args == 1:
                res = self.function(element)
            else:
                res = self.function(element, self, *self.args, **self.kwargs)
            self._update_state()
            return res
        else:
            self.state = State.STOPPED

    def can_continue(self, target: object) -> bool:
        """Checks if the modifier can continue to be applied.

        Args:
            target (object): The target object.

        Returns:
            bool: True if the modifier can continue, False otherwise.

        Examples:
            >>> import simetri.graphics as sg
            >>> def bump(element):
            ...     element['n'] += 1
            >>> mod = sg.Modifier(bump, life_span=2, seed=0)
            >>> mod.can_continue({'n': 0})
            True
            >>> sg.Modifier(bump, life_span=2, randomness=0.0, seed=0).can_continue({'n': 0})
            False
        """
        if callable(self.randomness):
            randomness = self.get_value(self.randomness, target)
        elif isinstance(self.randomness, float):
            randomness = self.randomness >= self._rng.random()
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
            >>> def bump(element):
            ...     element['n'] += 1
            >>> mod = sg.Modifier(bump, seed=0)
            >>> mod.stop()
            >>> mod.state
            <State.STOPPED: 'STOPPED'>
        """
        self.state = State.STOPPED
