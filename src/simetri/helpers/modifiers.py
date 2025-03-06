from random import random, choice
import inspect

from ..graphics.all_enums import Control, State


class Modifier:
    """Used to modify the properties of a Batch object.

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
    """

    def __init__(
        self, function, life_span=10000, randomness=1.0, condition=True, *args, **kwargs
    ):
        """
        Args:
            function (callable): The function to modify the property.
            life_span (int, optional): The number of times the modifier can be applied. Defaults to 10000.
            randomness (float or callable, optional): Determines the randomness of the modification. Defaults to 1.0.
            condition (bool or callable, optional): Condition to apply the modification. Defaults to True.
            *args: Additional arguments for the function.
            **kwargs: Additional keyword arguments for the function.
        """
        self.function = function  # it can be a list of functions
        signature = inspect.signature(function)
        self.n_func_args = len(signature.parameters)
        self.life_span = life_span
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
        self.count = 0
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        """Returns a string representation of the Modifier object.

        Returns:
            str: String representation of the Modifier object.
        """
        return (
            f"Modifier(function:{self.function}, lifespan:{self.life_span},"
            f"randomness:{self.randomness})"
        )

    def __str__(self):
        """Returns a string representation of the Modifier object.

        Returns:
            str: String representation of the Modifier object.
        """
        return self.__repr__()

    def set_state(self, control):
        """Sets the state of the modifier based on the control value.

        Args:
            control (Control): The control value to set the state.
        """
        self.state = self._d_state[control]

    def get_value(self, obj, target, *args, **kwargs):
        """Gets the value from an object or callable.

        Args:
            obj (object or callable): The object or callable to get the value from.
            target (object): The target object.
            *args: Additional arguments for the callable.
            **kwargs: Additional keyword arguments for the callable.

        Returns:
            object: The value obtained from the object or callable.
        """
        if callable(obj):
            res = obj(target, *args, **kwargs)
            if res in Control:
                self.set_state(res)
        else:
            res = obj
        return res

    def apply(self, element):
        """Applies the modifier to an element.

        If a function returns a control value, it will be applied to the modifier.
        Control.STOP, Control.PAUSE, Control.RESUME, and Control.RESTART are the only control values.
        Functions should have the following signature:
        def funct(target, modifier, *args, **kwargs):

        Args:
            element (object): The element to apply the modifier to.
        """
        if self.can_continue(element):
            if self.n_func_args == 1:
                self.function(element)
            else:
                self.function(element, self, *self.args, **self.kwargs)
            self._update_state()
        else:
            self.state = State.STOPPED

    def can_continue(self, target):
        """Checks if the modifier can continue to be applied.

        Args:
            target (object): The target object.

        Returns:
            bool: True if the modifier can continue, False otherwise.
        """
        if callable(self.randomness):
            randomness = self.get_value(self.randomness, target)
        elif type(self.randomness) == float:
            randomness = self.randomness >= random()
        elif type(self.randomness) in [list, tuple]:
            randomness = choice(self.randomness)

        if callable(self.condition):
            condition = self.get_value(self.condition, target)
        else:
            condition = self.condition

        if callable(self.life_span):
            life_span = self.get_value(self.life_span, target)
        else:
            life_span = self.life_span

        if life_span > 0 and condition and randomness:
            if self.state in [State.INITIAL, State.RUNNING, State.RESTARTING]:
                res = True
            else:
                res = False
        else:
            res = False
        return res

    def _update_state(self):
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

    def stop(self):
        """Stops the modifier."""
        self.state = State.STOPPED
