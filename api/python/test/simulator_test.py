# Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import jbw
from timeit import default_timer
from math import pi

class SimpleAgent(jbw.Agent):
	def __init__(self, simulator, load_filepath=None):
		self.counter = 0
		super(SimpleAgent, self).__init__(simulator, load_filepath)

	def do_next_action(self):
		self.counter += 1
		if self.counter % 20 == 0:
			self.turn(jbw.RelativeDirection.LEFT)
		elif self.counter % 20 == 5:
			self.turn(jbw.RelativeDirection.LEFT)
		elif self.counter % 20 == 10:
			self.turn(jbw.RelativeDirection.RIGHT)
		elif self.counter % 20 == 15:
			self.turn(jbw.RelativeDirection.RIGHT)
		else:
			self.move(jbw.RelativeDirection.FORWARD)

	def save(self, filepath):
		with open(filepath, 'w') as fout:
			fout.write(str(self.counter))

	def _load(self, filepath):
		with open(filepath, 'r') as fin:
			self.counter = int(fin.read())

def make_config():
	# specify the item types
	items = []
	items.append(jbw.Item("banana",    [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
					   intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-5.3],
					   interaction_fns=[
						   [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 0.0, -6.0],      # parameters for interaction between item 0 and item 0
						   [jbw.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -6.0, -6.0],      # parameters for interaction between item 0 and item 1
						   [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],    # parameters for interaction between item 0 and item 2
						   [jbw.InteractionFunction.ZERO]                                        # parameters for interaction between item 0 and item 3
						]))
	items.append(jbw.Item("onion",     [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0, 1, 0, 0], [0, 0, 0, 0], False, 0.0,
					   intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-5.0],
					   interaction_fns=[
						   [jbw.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -6.0, -6.0],      # parameters for interaction between item 1 and item 0
						   [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 1 and item 1
						   [jbw.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -100.0, -100.0],  # parameters for interaction between item 1 and item 2
						   [jbw.InteractionFunction.ZERO]                                        # parameters for interaction between item 1 and item 3
						]))
	items.append(jbw.Item("jellybean", [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0, 0, 0, 0], [0, 0, 0, 0], False, 0.0,
					   intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[-5.3],
					   interaction_fns=[
						   [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],    # parameters for interaction between item 2 and item 0
						   [jbw.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -100.0, -100.0],  # parameters for interaction between item 2 and item 1
						   [jbw.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 0.0, -6.0],      # parameters for interaction between item 2 and item 2
						   [jbw.InteractionFunction.ZERO]                                        # parameters for interaction between item 2 and item 3
						]))
	items.append(jbw.Item("wall",      [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0, 0, 0, 1], [0, 0, 0, 0], True, 1.0,
					   intensity_fn=jbw.IntensityFunction.CONSTANT, intensity_fn_args=[0.0],
					   interaction_fns=[
						   [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 0
						   [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 1
						   [jbw.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 2
						   [jbw.InteractionFunction.CROSS, 10.0, 15.0, 20.0, -200.0, -20.0, 1.0] # parameters for interaction between item 3 and item 3
						]))

	# construct the simulator configuration
	return jbw.SimulatorConfig(max_steps_per_movement=1, vision_range=5,
		allowed_movement_directions=[jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED],
		allowed_turn_directions=[jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.DISALLOWED, jbw.ActionPolicy.ALLOWED, jbw.ActionPolicy.ALLOWED],
		no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items, agent_color=[0.0, 0.0, 1.0],
		collision_policy=jbw.MovementConflictPolicy.FIRST_COME_FIRST_SERVED, agent_field_of_view=2*pi,
		decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000, seed=1234567890)

if __name__ == "__main__":
	# create a local simulator
	config = make_config()
	sim = jbw.Simulator(sim_config=config)

	# add agents to the simulation
	agents = []
	while len(agents) < 1:
		print("adding agent " + str(len(agents)))
		try:
			agent = SimpleAgent(sim)
			agents.append(agent)
		except jbw.AddAgentError:
			pass

		# move agents to avoid collision at (0,0)
		for agent in agents:
			agent.do_next_action()

	# construct the visualizer and start the main loop
	painter = jbw.MapVisualizer(sim, config, (-70, -70), (70, 70))
	start_time = default_timer()
	elapsed = 0.0
	sim_start_time = sim.time()
	for t in range(100000000):
		for agent in agents:
			agent.do_next_action()
		if default_timer() - start_time > 1.0:
			elapsed += default_timer() - start_time
			print(str((sim.time() - sim_start_time) / elapsed) + " simulation steps per second.")
			start_time = default_timer()
		#print("time: " + str(sim.time()))
		#print("agents[0].position(): " + str(agents[0].position()))
		#print("agents[0].collected_items(): " + str(agents[0].collected_items()))
		#print("agents[0].scent(): " + str(agents[0].scent()))
		#print("agents[0].vision(): " + str(agents[0].vision()))
		painter.draw()
