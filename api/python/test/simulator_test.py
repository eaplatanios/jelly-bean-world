import matplotlib
import nel
from random import choice
from timeit import default_timer

class SimpleAgent(nel.Agent):
	def __init__(self, simulator, load_filepath=None):
		self.counter = 0
		super(SimpleAgent, self).__init__(simulator, load_filepath)

	def do_next_action(self):
		self.counter += 1
		if self.counter % 20 == 0:
			self.turn(nel.RelativeDirection.LEFT)
		elif self.counter % 20 == 5:
			self.turn(nel.RelativeDirection.LEFT)
		elif self.counter % 20 == 10:
			self.turn(nel.RelativeDirection.RIGHT)
		elif self.counter % 20 == 15:
			self.turn(nel.RelativeDirection.RIGHT)
		else:
			self.move(nel.RelativeDirection.FORWARD)

	def save(self, filepath):
		with open(filepath, 'w') as fout:
			fout.write(str(self.counter))

	def _load(self, filepath):
		with open(filepath, 'r') as fin:
			self.counter = int(fin.read())

def make_config():
	# specify the item types
	items = []
	items.append(nel.Item("banana",    [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1, 0, 0, 0], [0, 0, 0, 0], False,
					   intensity_fn=nel.IntensityFunction.CONSTANT, intensity_fn_args=[-5.3],
					   interaction_fns=[
						   [nel.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 0.0, -6.0],      # parameters for interaction between item 0 and item 0
						   [nel.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -6.0, -6.0],      # parameters for interaction between item 0 and item 1
						   [nel.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],    # parameters for interaction between item 0 and item 2
						   [nel.InteractionFunction.ZERO]                                        # parameters for interaction between item 0 and item 3
						]))
	items.append(nel.Item("onion",     [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0, 1, 0, 0], [0, 0, 0, 0], False,
					   intensity_fn=nel.IntensityFunction.CONSTANT, intensity_fn_args=[-5.0],
					   interaction_fns=[
						   [nel.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -6.0, -6.0],      # parameters for interaction between item 1 and item 0
						   [nel.InteractionFunction.ZERO],                                       # parameters for interaction between item 1 and item 1
						   [nel.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -100.0, -100.0],  # parameters for interaction between item 1 and item 2
						   [nel.InteractionFunction.ZERO]                                        # parameters for interaction between item 1 and item 3
						]))
	items.append(nel.Item("jellybean", [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0, 0, 0, 0], [0, 0, 0, 0], False,
					   intensity_fn=nel.IntensityFunction.CONSTANT, intensity_fn_args=[-5.3],
					   interaction_fns=[
						   [nel.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 2.0, -100.0],    # parameters for interaction between item 2 and item 0
						   [nel.InteractionFunction.PIECEWISE_BOX, 200.0, 0.0, -100.0, -100.0],  # parameters for interaction between item 2 and item 1
						   [nel.InteractionFunction.PIECEWISE_BOX, 10.0, 200.0, 0.0, -6.0],      # parameters for interaction between item 2 and item 2
						   [nel.InteractionFunction.ZERO]                                        # parameters for interaction between item 2 and item 3
						]))
	items.append(nel.Item("wall",      [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0, 0, 0, 1], [0, 0, 0, 0], True,
					   intensity_fn=nel.IntensityFunction.CONSTANT, intensity_fn_args=[0.0],
					   interaction_fns=[
						   [nel.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 0
						   [nel.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 1
						   [nel.InteractionFunction.ZERO],                                       # parameters for interaction between item 3 and item 2
						   [nel.InteractionFunction.CROSS, 10.0, 15.0, 20.0, -200.0, -20.0, 1.0] # parameters for interaction between item 3 and item 3
						]))

	# construct the simulator configuration
	return nel.SimulatorConfig(max_steps_per_movement=1, vision_range=5,
		allowed_movement_directions=[nel.ActionPolicy.ALLOWED, nel.ActionPolicy.DISALLOWED, nel.ActionPolicy.DISALLOWED, nel.ActionPolicy.DISALLOWED],
		allowed_turn_directions=[nel.ActionPolicy.DISALLOWED, nel.ActionPolicy.DISALLOWED, nel.ActionPolicy.ALLOWED, nel.ActionPolicy.ALLOWED],
		no_op_allowed=False, patch_size=32, mcmc_num_iter=4000, items=items, agent_color=[0.0, 0.0, 1.0],
		collision_policy=nel.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
		decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000,
		seed=1234567890)

if __name__ == "__main__":
	# create a local simulator
	config = make_config()
	sim = nel.Simulator(sim_config=config)

	# add agents to the simulation
	agents = []
	while len(agents) < 1:
		print("adding agent " + str(len(agents)))
		try:
			agent = SimpleAgent(sim)
			agents.append(agent)
		except nel.AddAgentError:
			pass

		# move agents to avoid collision at (0,0)
		for agent in agents:
			agent.do_next_action()

	# construct the visualizer and start the main loop
	painter = nel.MapVisualizer(sim, config, (-70, -70), (70, 70))
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
