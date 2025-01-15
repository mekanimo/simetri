from random import random, choice
import inspect

from ..graphics.all_enums import Control, State


class Modifier:
    """Used to modify the properties of a Batch object.
    type: Types.INCREMENTER, Types. Types.MULTIPLIER, Types.TRANSFORMER
    life_span: the number of times the modifier can be applied to the property
    randomness: a value between 0 and 1 that determines the randomness of
    the modification, it can also be a function that returns a value between
    0 and 1"""

    def __init__(
        self, function, life_span=10000, randomness=1.0, condition=True, *args, **kwargs
    ):
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
        return (
            f"Modifier(function:{self.function}, lifespan:{self.life_span},"
            f"randomness:{self.randomness})"
        )

    def __str__(self):
        return self.__repr__()

    def set_state(self, control):
        self.state = self._d_state[control]

    def get_value(self, obj, target, *args, **kwargs):
        if callable(obj):
            res = obj(target, *args, **kwargs)
            if res in Control:
                self.set_state(res)
        else:
            res = obj
        return res

    def apply(self, element):
        """If a function returns a control value, it will be applied to
        the modifer.
        Control.STOP, Control.PAUSE, Control.RESUME, and Control.RESTART
        are the only control values.
        functions should have the following signature:
        def funct(target, modifier, *args, **kwargs):"""
        if self.can_continue(element):
            if self.n_func_args == 1:
                self.function(element)
            else:
                self.function(element, self, *self.args, **self.kwargs)
            self._update_state()
        else:
            self.state = State.STOPPED

    def can_continue(self, target):
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
            life_span = self.getValue(self.life_span, target)
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
        self.state = State.STOPPED
