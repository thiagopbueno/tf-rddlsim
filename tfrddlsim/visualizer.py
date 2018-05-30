import abc
import numpy as np


class Visualizer(metaclass=abc.ABCMeta):

    def __init__(self, compiler, verbose=0):
        self.compiler = compiler
        self.verbose = verbose

    @abc.abstractmethod
    def render(self, trajectories, batch=None):
        return


class BasicVisualizer(Visualizer):

    def __init__(self, compiler, verbose=0):
        super().__init__(compiler, verbose)

    def render(self, trajectories, batch=None):
        non_fluents, states, actions, interms, rewards = trajectories
        if batch is None:
            states = [(s[0], s[1][0]) for s in states]
            interms = [(f[0], f[1][0]) for f in interms]
            actions = [(a[0], a[1][0]) for a in actions]
            rewards = rewards[0]
            self._render_batch(non_fluents, states, actions, interms, rewards)

    def _render_batch(self, non_fluents, states, actions, interms, rewards, horizon=None):
        if horizon is None:
            horizon = len(states[0][1])
            self._render_round_init(horizon, non_fluents)
            for t in range(horizon):
                s = [(s[0], s[1][t]) for s in states]
                f = [(f[0], f[1][t]) for f in interms]
                a = [(a[0], a[1][t]) for a in actions]
                r = rewards[t]
                self._render_timestep(t, s, a, f, r)
            self._render_round_end(rewards)

    def _render_timestep(self, t, s, a, f, r):
        print("============================")
        print("TIME = {}".format(t))
        print("============================")
        fluent_variables = self.compiler.action_fluent_variables
        self._render_fluent_timestep('action', a, fluent_variables)
        fluent_variables = self.compiler.interm_fluent_variables
        self._render_fluent_timestep('interms', f, fluent_variables)
        fluent_variables = self.compiler.state_fluent_variables
        self._render_fluent_timestep('states', s, fluent_variables)
        self._render_reward(r)

    def _render_fluent_timestep(self, fluent_type, fluents, fluent_variables):
        for fluent, fluent_variables in zip(fluents, fluent_variables):
            name, fluent = fluent
            _, variables = fluent_variables
            print(name)
            fluent = fluent.flatten()
            for variable, value in zip(variables, fluent):
                print('- {}: {} = {}'.format(fluent_type, variable, value))
        print()

    def _render_reward(self, r):
        print("reward = {:.4f}".format(float(r)))
        print()

    def _render_round_init(self, horizon, non_fluents):
        print('*********************************************************')
        print('>>> ROUND INIT, horizon = {}'.format(horizon))
        print('*********************************************************')
        fluent_variables = self.compiler.non_fluent_variables
        self._render_fluent_timestep('non-fluents', non_fluents, fluent_variables)

    def _render_round_end(self, rewards):
        print("*********************************************************")
        print(">>> ROUND END")
        print("*********************************************************")
        total_reward = np.sum(rewards)
        print("==> Objective value = {}".format(total_reward))
        print("==> rewards = {}".format(list(rewards)))
        print()
