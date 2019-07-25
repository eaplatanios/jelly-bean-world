from __future__ import absolute_import, division, print_function

import jbw
import gym

env = gym.make('JBW-render-v0')

for t in range(10000):
  env.render()
  action = env.action_space.sample()
  print(action)
  observation, reward, _, _ = env.step(action)
  print(observation['moved'])
