"""Collection of NEL environments for OpenAI gym."""

from __future__ import absolute_import, division, print_function

try:
  from gym.envs.registration import register
  modules_loaded = True
except:
  modules_loaded = False

from .agent import Agent
from .direction import RelativeDirection
from .item import *
from .simulator import *
from .visualizer import MapVisualizer

if modules_loaded:
  # Specify the item types.
  items = []
  items.append(Item("banana", [1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0], True,
          intensity_fn=IntensityFunction.CONSTANT, intensity_fn_args=[-2.0],
          interaction_fns=[[InteractionFunction.PIECEWISE_BOX, 40.0, 200.0, 0.0, -40.0]]))

  # Construct the simulator configuration.
  sim_config = SimulatorConfig(max_steps_per_movement=1, vision_range=1,
    allowed_movement_directions=[RelativeDirection.FORWARD],
    allowed_turn_directions=[RelativeDirection.LEFT, RelativeDirection.RIGHT],
    patch_size=32, gibbs_num_iter=10, items=items, agent_color=[0.0, 0.0, 1.0],
    collision_policy=MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
    decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000)

  # Create a reward function.
  reward_fn = lambda prev_items, items: len(items) - len(prev_items)

  register(
      id='NEL-v0',
      entry_point='nel.environment:NELEnv',
      kwargs={
        'sim_config': sim_config,
        'reward_fn': reward_fn,
        'render': False})

  register(
      id='NEL-render-v0',
      entry_point='nel.environment:NELEnv',
      kwargs={
        'sim_config': sim_config,
        'reward_fn': reward_fn,
        'render': True})
