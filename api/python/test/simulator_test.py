import nel
from math import pi, tan
from random import choice
from time import sleep
from timeit import default_timer

class EasterlyAgent(nel.Agent):
	def __init__(self, simulator):
		super(EasterlyAgent, self).__init__(simulator)

	def next_move(self):
		return nel.Direction.RIGHT

	def save(self, saved):
		pass


items = []
items.append(nel.Item("banana", [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], False))
items.append(nel.Item("onion", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], False))
items.append(nel.Item("jellybean", [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], True))

intensity_fn_args = [-5.3, -5.0, -5.3]
interaction_fn_args = [len(items)]
interaction_fn_args.extend([10.0, 200.0, 0.0, -6.0])     # parameters for interaction between item 0 and item 0
interaction_fn_args.extend([200.0, 0.0, -6.0, -6.0])     # parameters for interaction between item 0 and item 1
interaction_fn_args.extend([10.0, 200.0, 2.0, -100.0])   # parameters for interaction between item 0 and item 2
interaction_fn_args.extend([200.0, 0.0, -6.0, -6.0])     # parameters for interaction between item 1 and item 0
interaction_fn_args.extend([0.0, 0.0, 0.0, 0.0])         # parameters for interaction between item 1 and item 1
interaction_fn_args.extend([200.0, 0.0, -100.0, -100.0]) # parameters for interaction between item 1 and item 2
interaction_fn_args.extend([10.0, 200.0, 2.0, -100.0])   # parameters for interaction between item 2 and item 0
interaction_fn_args.extend([200.0, 0.0, -100.0, -100.0]) # parameters for interaction between item 2 and item 1
interaction_fn_args.extend([10.0, 200.0, 0.0, -6.0])     # parameters for interaction between item 2 and item 2

config = nel.SimulatorConfig(max_steps_per_movement=1, vision_range=1,
	patch_size=32, gibbs_num_iter=10, items=items, agent_color=[0.0, 0.0, 1.0],
	collision_policy=nel.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
	decay_param=0.4, diffusion_param=0.14, deleted_item_lifetime=2000,
	intensity_fn=nel.IntensityFunction.CONSTANT, intensity_fn_args=intensity_fn_args,
	interaction_fn=nel.InteractionFunction.PIECEWISE_BOX, interaction_fn_args=interaction_fn_args)

sim = nel.Simulator(sim_config=config)

agents = []
while len(agents) < 1:
	print("adding agent " + str(len(agents)))
	try:
		agent = EasterlyAgent(sim)
		agents.append(agent)
	except:
		pass

	# move agents to avoid collision at (0,0)
	for agent in agents:
		sim._move(agent._id, agent.next_move(), 1)

painter = nel.MapVisualizer(sim, config, (-30, -30), (150, 150))
start_time = default_timer()
elapsed = 0.0
for t in range(10000):
	for agent in agents:
		sim._move(agent._id, agent.next_move(), 1)
	if default_timer() - start_time > 1.0:
		elapsed += default_timer() - start_time
		print(str(sim.time() / elapsed) + " simulation steps per second.")
		start_time = default_timer()
	#print("time: " + str(sim.time()))
	#print("agents[0].position(): " + str(agents[0].position()))
	#print("agents[0].collected_items(): " + str(agents[0].collected_items()))
	#print("agents[0].scent(): " + str(agents[0].scent()))
	#print("agents[0].vision(): " + str(agents[0].vision()))
	painter.draw()
