#Non-deterministic Finite State Machine
#from mekanigame.utilities.timer import SimpleTimer
from collections import Iterable, Sequence, Callable
import random
import time
from operator import *
from collections import deque


BINARY = 1
UNARY = 2

class PubSub(object):

    def __init__(self):
        self.topics = defaultdict(list)

    def subscribe(self, topic, func):
        self.topics[topic].add(func)

    def publish(self, topic, **kwargs):
        for func in self.topics[topic]:
            func(**kwargs)
'''
def listener(data=None, **kwargs):
    print(data)

pubsub = PubSub()
pubsub.subscribe("Topic-1", listener)
pubsub.publish("Topic-1", data="Hello World")
'''

class State:
    '''Represents states in finite state machines.'''
    def __init__(self, name, enter=None, exit=None, update=None):
        #enter and exit are callable
        self.name = name
        self._enter = enter
        self._exit = exit
        self._update = update
        #self.timer = SimpleTimer()

    def start_timer(self):
        self.timer.start()

    def stop_timer(self):
        self.timer.stop()

    def reset_timer(self):
        if self.timer.running:
            self.stop_timer()
        self.timer.reset()

    def elapsed_time(self):
        return self.timer.elapsed

    def __str__(self):
        return f"State({self.name})"

    def __repr__(self):
        s = [f"'{self.name}'"]
        functions = [x.__name__ if x else 'None' for x in [self._enter, self._exit, self._update]]
        s.extend(functions)
        s = ', '.join(s)
        return f'State({s})'


class Event:
    '''
       Transitions use these events to cause state changes.

    '''
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

    def _match_state(self, state, lhs):
        def _match_slot(state, slot):
            if slot == '*':
                return True
            return (state.name in slot)

        for i, slot in enumerate(lhs):
            if not _match_slot(state[i], slot):
                return False
        return True

    def __call__(self):
        parent = self.parent
        transition_map = parent._transition_map
        for transition in transition_map:
            lhs = transition[1]
            if self._match_state(parent.state, lhs):
                if (self.name, lhs) in transition_map:
                    old_states = self.parent.state[:]
                    new_states = transition_map[(self.name, lhs)]
                    if isinstance(new_states, Callable):
                        new_states(self.parent)
                        new_states = self.parent.state
                    elif isinstance(new_states, str):
                        eval(new_states)
                        new_states = self.parent.state
                    parent.change_states(old_states, new_states)
                    return


class NDFSM:
    '''
        Non-deterministic finite state machine.
        It is a hybrid machine that works like objects and/or
        state machines.  They can be attached to other state
        machines to form an hierarchy.  Top level has the highest
        ranking and they can modify lower ranked machines state changes
        by blocking or triggering state changes (somehow similar to
        Rodney Brooks' subsumption architecture).
    '''

    def __init__(self, name, states=None, initial_state=None, slots=None, transitions=None,
                 triggers=None, max_history=500, **kwargs):
        self.name = name
        self.history = deque([], max_history)
        self._set_states(states, slots)
        if isinstance(initial_state, str):
            self.initial_state = [self.states[initial_state]]
        elif isinstance(initial_state, Iterable):
            self.initial_state = [self.states[state_name] for state_name in initial_state]
        else:
            self.initial_state = None
        self.__dict__['state'] = self.initial_state
        self.slots = slots
        self._set_slot_keys(slots)
        self._set_triggers(triggers)
        self.__dict__.update(kwargs)
        if self.initial_state:
            self.history.append([str(x) for x in self.initial_state])
        self._set_events(transitions)
        self._set_transitions(transitions)
        self._set_properties()
        self.blocked_states = set()
        self._fulfilled_changes = []

    @property
    def state(self):
        return self.__dict__['state']

    @state.setter
    def state(self, val):
        self.__dict__['state'] = val
        #we can add something for state changes here


    def _set_states(self, states, slots):
        #check if slots and states match
        self.states = { }
        if states:
            for state in states:
                name, enter, exit, update = state
                self.states[name] = State(name, enter, exit, update)
        else:
            states = [ ]
        if slots:
            for slot in slots.values():
                for state_name in slot:
                    if state_name not in self.states:
                        self.states[state_name] = State(state_name, None, None, None)

    def _set_triggers(self, triggers):
        if triggers:
            for trigger in triggers:
                attr_name, behaviors = trigger
                setattr(NDFSM, attr_name, Trigger(attr_name, behaviors, self))

    def _set_slot_keys(self, slots):
        if slots:
            self.slot_keys = {}
            for k, v in slots.items():
                for item in v:
                    self.slot_keys[item] = k
            self.slot_indexes = {}
            for i, (k, v) in enumerate(slots.items()):
                for state_name in v:
                    self.slot_indexes[state_name] = i

    def _set_events(self, transitions):
        if transitions:
            self.events = { }
            #event_names = [transition[0] for transition in transitions]
            event_names = [transition.method_name for transition in transitions]
            for event_name in event_names:
                event_instance = Event(event_name, self)
                self.events[event_name] = event_instance
                setattr(self, event_name, event_instance)

    def _set_transitions(self, transitions):
        def expand_state(slot_keys, states):
            new_states = ['*'] * len(self.slots)
            for i, (k, v) in enumerate(self.slots.items()):
                if k in slot_keys:
                    index = slot_keys.index(k)
                    state = states[index]
                    if isinstance(state, list):
                        if len(state) == 1:
                            state = state[0]
                        else:
                            state = tuple(state)
                    new_states[i] = state
            return new_states
        if transitions:
            self.transitions = transitions
            for transition in transitions:
                transition.start_states = tuple(expand_state(transition.slot_keys, transition.start_states))
                if isinstance(transition.end_states, Sequence):
                    transition.end_states = tuple(expand_state(transition.slot_keys, transition.end_states))
                else:
                    transition.end_states = transition.end_states
            self._transition_map = { }
            for transition in transitions:
                if isinstance(transition.end_states, Sequence):
                    (self._transition_map[(transition.method_name, transition.start_states)] =
                     ['*' if state is '*'\ else self.states[state] for state in transition.end_states])
                elif isinstance(transition.end_states, Callable):
                    self._transition_map[(transition.method_name, transition.start_states)] = transition.end_states

    def _get_func(self, state_name):
        def func(self):
            return state_name in [x.name for x in self.state]
        return func

    def _remove_fulfilled_state_changes(self, old_states, new_states):
        for change in self._fulfilled_changes:
            old_state, new_state_name = change
            new_state = self.states[new_state_name]
            if old_state[0] in old_states and new_state in new_states:
                index = old_states.index(old_state[0])
                if index == new_states.index(new_state):
                    old_states.pop(index)
                    new_states.pop(index)

    def _set_properties(self):
        if self.slots:
            for key, slot in self.slots.items():
                for state_name in slot:
                    func = self._get_func(state_name)
                    prop_name = f"is_{state_name[0].capitalize()}{state_name[1:]}"
                    setattr(NDFSM, prop_name, property(func))

    def _state_transitions(self, old_states, new_states):
        if old_states == new_states:
            return
        self._remove_fulfilled_state_changes(old_states, new_states)
        self._fulfilled_changes = [ ]
        for state in [x for x in old_states if x not in new_states and x._exit]:
            state._exit(self)
        for state in [x for x in new_states if x not in old_states and x != '*' and x._enter]:
            state._enter(self)
        for state in [x for x in new_states if x not in old_states and x != '*' and x._update]:
            state._update(self)

    def change_state(self, old_state, new_state_name):
        if new_state_name not in self.blocked_states:
            new_state = self.states[new_state_name]
            index = self.slot_indexes[new_state_name]
            if old_state !=  new_state:
                self.state[index] = new_state
                self._state_transitions(old_state, self.state[:])
                if self.state != self.history[-1]:
                    self.history.append(self.state[:])
                return True
            return False
        else:
            return False

    def change_states(self, old_states, new_states):
        new_states = [state for state in new_states if state not in self.blocked_states]
        if new_states:
            self.state = [old_states[i] if x == '*' else x for i, x in enumerate(new_states)]
            if old_states != new_states:
                self._state_transitions(old_states, self.state[:])
                if self.state != self.history[-1]:
                    self.history.append(self.state[:])



    def block_state(self, state_name):
        self.blocked_states.add(state_name)

    def unblock_state(state_name):
        if state_name in self.blocked_states:
            self.blocked_states.remove(state_name)

    def subscribe(self, func, topic):
        # func must have two arguments: func(arg1, arg2)
        Publisher.subscribe(func, topic)

    def send_message(self, topic, arg1, arg2=None):
        Publisher.send_message(topic, arg1, arg2)


class Trigger:
    '''
        Descriptor for handling triggers.
        A trigger is an attribute that can change another attribute
        if a predefined condition is met.  Conditions must yield boolean
        values that Python 3 understands.
        In FSMs we use them to change states.
    '''

    def __init__(self, attr_name, behaviors, fsm):
        self.attr_name = attr_name
        self.fsm = fsm
        self.behaviors = behaviors

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.attr_name]

    def __set__(self, instance, value):
        instance.__dict__[self.attr_name] = value
        for beh in self.behaviors:
            cond, target_state = beh
            cond = cond.replace('self.', 'instance.')
            if eval(cond):
                old_state = instance.state[:]
                if old_state[0].name != target_state:
                    instance.change_state(old_state, target_state)
                    instance._fulfilled_changes.append((old_state, target_state))
                return

class Transition:
    '''Defines a state transition.  They are very similar to rules.'''

    def __init__(self, method_name, precondition, slot_keys, start_states, end_states, on_success=None, on_fail=None):
        self.method_name = method_name
        self.precondition = precondition
        self.slot_keys = slot_keys
        self.start_states = start_states
        self.end_states = end_states


def into_mine(self):
    if debug:
        print('In the gold mine.')

def out_of_mine(self):
    if debug:
        print('I am out of the goldmine.')

def dig_gold(self):
    miner = self
    if debug:
        print('Digging gold, making money.')
    for i in range(10):
        if random.choice(['Gold', None]):
            miner.gold_in_pocket += 1
            if debug:
                print('Found gold.')
                print(f'Gold in pocket is {miner.gold_in_pocket}')
        else:
            if debug:
                print('No gold this time.')
        miner.fatigue_level += 1
        miner.thirst_level += 1
        if miner.is_pockets_full:
            return

def into_bank(self):
    if debug:
        print('I am at the bank.')

def out_of_bank(self):
    if debug:
        print('I am leaving the bank.')

def deposit_gold(self):
    miner = self
    miner.account_balance += miner.gold_in_pocket
    miner.gold_in_pocket = 0
    if debug:
        print(f'Account balance is {miner.account_balance}')

def into_home(self):
    if debug:
        print('I am home')

def out_of_home(self):
    if debug:
        print('Bye')


def into_bar(self):
    if debug:
        print('I am at the bar')
        print('State:', miner.state)

def out_of_bar(self):
    if debug:
        print('I am leaving the bar.')

def rest(self):
    miner = self
    if debug:
        print('I am resting.')
    miner.fatigue_level = 0
    miner.thirst_level = 0

def rich_now(self):
    if debug:
        print(25 * '#')
        print('I am rich now.')

def drink(self):
    miner = self
    if debug:
        print('I am having a beer.')
    miner.thirst_level = 0
    miner.gold_in_pocket -= 1

#(state name, enter func, exit func, update func)
states = [
          ('at_mine', into_mine, out_of_mine, dig_gold),
          ('at_home', into_home, out_of_home, rest),
          ('at_bar', into_bar, out_of_bar, drink),
          ('at_bank', into_bank, out_of_bank, deposit_gold),
          ('rested', None, None, None),
          ('tired', None, None, None ),
          ('thirsty', None, None, None),
          ('not_thirsty', None, None, None),
          ('wealthy', rich_now, None, None),
          ('not_wealthy', None, None, None),
          ('pockets_full', None, None, None),
          ('pockets_not_full', None, None, None),
          ]

states = [
          ('at_mine', into_mine, out_of_mine, dig_gold),
          ('at_home', into_home, out_of_home, rest),
          ('at_bar', into_bar, out_of_bar, drink),
          ('at_bank', into_bank, out_of_bank, deposit_gold),
          ('wealthy', rich_now, None, None),
          ]


slots = {
          'location': ['at_home', 'at_mine', 'at_bar', 'at_bank'],
          'tiredness': ['rested', 'tired'],
          'thirst': ['thirsty', 'not_thirsty'],
          'wealth': ['wealthy', 'not_wealthy'],
          'pockets': ['pockets_full', 'pockets_not_full']
        }

transitions = [
                Transition(
                           method_name='goto_mine',
                           precondition=None,
                           slot_keys=['location'],
                           start_states=[['at_home', 'at_bar', 'at_bank']],
                           end_states=[['at_mine']],
                           ),
                Transition(
                           method_name='go_home',
                           precondition=None,
                           slot_keys=['location'],
                           start_states=[['at_bar', 'at_bank']],
                           end_states=[['at_home']],
                           ),
                Transition(
                           method_name='goto_bar',
                           precondition='miner.is_thirsty and ',
                           slot_keys=['location'],
                           start_states=[['at_mine']],
                           end_states=[['at_bar']],
                           ),
                Transition(
                           method_name='goto_bank',
                           precondition=None,
                           slot_keys=['location'],
                           start_states=[['at_mine']],
                           end_states=[['at_bank']],
                           ),
                ]

# preconditions can be defined from single state to single state preconditions = ('at_home', 'at_bar', cond)
# or multi-state to multi-state ((('at_home', 'at_bar', 'at_bank'), '*', '*', '*', '*'),
#                                 ('at_mine', '*', '*', '*', '*'), cond)
# by adding the condition to the transition (method, start_states, end_states, cond)
# or both
# They are usually about externalities to the FSM
# For example, rain/snow, temperature, blocked roads etc. that are not part of the FSM
# It is easier to use these as preconditions as opposed to designing FSMs by including these
# externalities.  They may significantly reduce CPU load by limiting the number of states
# to managable numbers.

preconditions = [
                  ()
                ]


# Inhibitors are internal/external dynamic inputs preventing state transition events
# They can be activated and deactivated internally or externally
# They can be thought as temporary preconditions
# For more info see Subsumption Architecture by Brooks

inhibitors = [ ]


# Suppressors replace/block transitions
# Suppressors and Inhibirtors are used in Augmented FSMs

suppressors = [ ]

# Triggers:
# ge: greater or equal, lt: less than, gt: greater than,
# (attribute name, condition)
# if the condition is True then the state is automatically activated


# For example:
# We can read the first trigger below as
# if obj.fatigu_level >= 3:
#     appropriate slot is State('thirsty)
# else:
#     appropriate slot is State('not_thirsty)

#triggers = (
            #('thirst_level', (
                             #((ge, 3), 'thirsty'),
                             #((lt, 3), 'not_thirsty')
                             #),
             #),
            #('account_balance', (
                             #((gt, 10), 'wealthy'),
                             #((lt, 10), 'not_wealthy'))),
            #('fatigue_level', (
                             #((ge, 3), 'tired'),
                             #((lt, 3), 'rested'))),
            #('gold_in_pocket', (
                            #((ge, 3), 'pockets_full'),
                            #((lt, 3), 'pockets_not_full')))
            #)

triggers = (
            ('thirst_level', (
                             ('self.thirst_level >= 3', 'thirsty'),
                             ('self.thirst_level < 3', 'not_thirsty')
                             ),
             ),
            ('account_balance', (
                             ('self.account_balance >= 10', 'wealthy'),
                             ('self.account_balance < 10', 'not_wealthy'))),
            ('fatigue_level', (
                             ('self.fatigue_level >= 3', 'tired'),
                             ('self.fatigue_level < 3', 'rested'))),
            ('gold_in_pocket', (
                            ('self.gold_in_pocket >= 3', 'pockets_full'),
                            ('self.gold_in_pocket < 3', 'pockets_not_full')))
            )

miner = NDFSM(
              name = 'Miner',
              states=states,
              initial_state=('at_home', 'rested', 'not_thirsty', 'not_wealthy', 'pockets_not_full'),
              slots=slots,
              transitions=transitions,
              triggers=triggers,
              max_history=5000,

              gold_in_pocket=0,
              account_balance=0,
              thirst_level=0,
              fatigue_level=0
             )

debug = True

def run():
    for x in range(30):
        if debug:
            print([str(x) for x in miner.state])
        if miner.is_at_home or miner.is_at_bar:
            miner.goto_mine()
        elif miner.is_at_bank:
            if miner.is_not_wealthy and miner.is_rested and miner.is_not_thirsty:
                miner.goto_mine()
            else:
                miner.go_home()
        else:
            if miner.is_pockets_full and miner.is_thirsty:
                miner.goto_bar()
            else:
                miner.goto_bank()
        time.sleep(.1)

run()
for state in miner.history:
    print([str(x) for x in state])
print(len(miner.history))
print('Done')
