import nel
from math import pi, tan
from random import choice
from time import sleep

def try_move(agent):
	#dir = choice(list(nel.Direction))
	dir = nel.Direction.RIGHT
	return agent.move(dir)

class RandomAgent(nel.Agent):
	def __init__(self, simulator):
		super().__init__(simulator)

	def on_step(self):
		pass


items = []
items.append(nel.Item("banana", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], True))

intensity_fn_args = [-2.0]
interaction_fn_args = [len(items)]
interaction_fn_args.extend([40.0, 200.0, 0.0, -40.0]) # parameters for interaction between item 0 and item 0

config = nel.SimulatorConfig(max_steps_per_movement=1, vision_range=1,
	patch_size=32, gibbs_num_iter=10, items=items, agent_color=[0.0, 0.0, 1.0],
	collision_policy=nel.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
	decay_param=0.5, diffusion_param=0.12, deleted_item_lifetime=2000,
	intensity_fn=nel.IntensityFunction.CONSTANT, intensity_fn_args=intensity_fn_args,
	interaction_fn=nel.InteractionFunction.PIECEWISE_BOX, interaction_fn_args=interaction_fn_args)

sim = nel.Simulator(sim_config=config)

agents = []
while len(agents) < 1:
	print("adding agent " + str(len(agents)))
	try:
		agent = RandomAgent(sim)
		agents.append(agent)
	except:
		pass

	# move agents to avoid collision at (0,0)
	for agent in agents:
		try_move(agent)

for t in range(1000):
	for agent in agents:
		try_move(agent)
	print("time: " + str(sim.time()))
	print("agents[0].position(): " + str(agents[0].position()))
	print("agents[0].collected_items(): " + str(agents[0].collected_items()))
	print("agents[0].scent(): " + str(agents[0].scent()))
	print("agents[0].vision(): " + str(agents[0].vision()))
