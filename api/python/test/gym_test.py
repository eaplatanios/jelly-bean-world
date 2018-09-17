from __future__ import absolute_import, division, print_function

import nel
import gym

env = gym.make('NEL-render-v0')

observation = env.reset()
for t in range(10000):
  env.render()
  action = env.action_space.sample()
  print(action)
  observation, reward, _, _ = env.step(action)
  print(observation)
