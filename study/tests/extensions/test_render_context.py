"""Renderer ownership is essential when interleaving different LIBERO tasks."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def environments():
    from lerobot.envs.libero import LiberoEnv
    active = {'owner': None}
    events = []

    def make(name, task_id):
        def bind():
            active['owner'] = name
            events.append(('bind', name))

        def operation(kind, value):
            def call(*args):
                assert active['owner'] == name, f'{name} used another environment\'s GL resources'
                events.append((kind, name))
                return value
            return call

        raw = {'pixels': {'image': np.full((2, 2, 3), task_id, dtype=np.uint8)}}
        backend = SimpleNamespace(
            sim=SimpleNamespace(_render_context_offscreen=SimpleNamespace(gl_ctx=SimpleNamespace(make_current=bind))),
            robots=[SimpleNamespace(controller=SimpleNamespace(use_delta=True))],
            seed=lambda seed: None, reset=operation('reset', raw),
            step=operation('step', (raw, 0., False, {})), check_success=lambda: False,
            close=operation('close', None), env=SimpleNamespace(_get_observations=operation('render', raw)))
        env = LiberoEnv.__new__(LiberoEnv)
        env._env = backend
        env.task = name
        env.task_id = task_id
        env.init_states = False
        env.num_steps_wait = 0
        env.control_mode = 'relative'
        env._format_raw_obs = lambda observation: observation
        return env

    return [make('table', 0), make('stove', 2)], active, events


def test_alternating_tasks_bind_their_own_context_before_rendering(environments):
    envs, active, events = environments
    for env in [envs[0], envs[1], envs[0], envs[1]]:
        # Simulate a different simulator being the last one to reset/render.
        active['owner'] = 'foreign'
        raw, *_ = env.step(np.zeros(7))
        assert np.all(raw['pixels']['image'] == env.task_id)
        active['owner'] = 'foreign'
        assert np.all(env.render() == env.task_id)
    assert all(events[i][0] == 'bind' for i in range(0, len(events), 2))


def test_reset_and_close_do_not_use_another_environments_resources(environments):
    envs, active, events = environments
    for env in envs:
        active['owner'] = 'foreign'
        env.reset(seed=100)
    for env in reversed(envs):
        active['owner'] = 'foreign'
        env.close()
    assert [kind for kind, _ in events] == ['bind', 'reset', 'bind', 'reset', 'bind', 'close', 'bind', 'close']
