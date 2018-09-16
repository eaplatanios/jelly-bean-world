import nel
from timeit import default_timer
from simulator_test import SimpleAgent, make_config
import numpy as np

def compare_patches(patch1, patch2):
	(fixed1, scent1, vision1, items1, agents1) = patch1
	(fixed2, scent2, vision2, items2, agents2) = patch2
	equal = True
	if fixed1 != fixed2:
		print('First patch has fixed = ' + str(fixed1) + ' but second has fixed = ' + str(fixed2) + '.')
		equal = False
	if np.any(scent1 != scent2):
		print('The scent values in the patches differ.')
		equal = False
	if np.any(vision1 != vision2):
		print('The vision values in the patches differ.')
		equal = False
	if sorted(agents1) != sorted(agents2):
		print('The agent positions in the patches differ.')
		equal = False
	return equal

def compare_simulators(sim1, sim2, config,
		min_agent_position_x, min_agent_position_y,
		max_agent_position_x, max_agent_position_y):
	if sim1.time() != sim2.time():
		print("The simulators have different times!")
		return False
	map1 = sim1._map((min_agent_position_x - config.patch_size*2, min_agent_position_y - config.patch_size*2),
					 (max_agent_position_x + config.patch_size*2, max_agent_position_y + config.patch_size*2))
	map2 = sim2._map((min_agent_position_x - config.patch_size*2, min_agent_position_y - config.patch_size*2),
					 (max_agent_position_x + config.patch_size*2, max_agent_position_y + config.patch_size*2))
	patches1, patches2 = (dict(), dict())
	equal = True
	for patch in map1:
		(patch_position, fixed, scent, vision, items, agents) = patch
		patches1[patch_position] = (fixed, scent, vision, items, agents)
	for patch in map2:
		(patch_position, fixed, scent, vision, items, agents) = patch
		patches2[patch_position] = (fixed, scent, vision, items, agents)
	for patch_position, patch1 in patches1.items():
		if patch_position not in patches2:
			print("Patch at " + str(patch_position) + " is in the first simulator but not the second.")
			equal = False
			continue
		patch2 = patches2[patch_position]
		if not compare_patches(patch1, patch2):
			print("Patch at " + str(patch_position) + " differs.")
			equal = False
	for patch_position, patch2 in patches2.items():
		if patch_position not in patches1:
			print("Patch at " + str(patch_position) + " is in the second simulator but not the first.")
			equal = False
	return equal

# create two simulators
save_frequency = 50
config = make_config()
sim1 = nel.Simulator(sim_config=config)
sim2 = nel.Simulator(sim_config=config, save_filepath="./temp/simulator_state", save_frequency=save_frequency)

# add one agent to each simulator
agent_type = SimpleAgent
agent1 = agent_type(sim1)
agent2 = agent_type(sim2)

# start main loop
start_time = default_timer()
elapsed = 0.0
sim_start_time = sim2.time()
(min_agent_position_x, min_agent_position_y) = (0, 0)
(max_agent_position_x, max_agent_position_y) = (0, 0)
for t in range(10000):
	agent1.do_next_action()
	agent2.do_next_action()
	if default_timer() - start_time > 1.0:
		elapsed += default_timer() - start_time
		print(str((sim2.time() - sim_start_time) / elapsed) + " simulation steps per second.")
		start_time = default_timer()

	agent_position = agent2.position()
	min_agent_position_x = min(min_agent_position_x, agent_position[0])
	max_agent_position_x = max(max_agent_position_x, agent_position[0])
	min_agent_position_y = min(min_agent_position_y, agent_position[1])
	max_agent_position_y = max(max_agent_position_y, agent_position[1])

	if sim2.time() % save_frequency == 0:
		# reload simulator 2 from file
		print("Reloading simulator 2 from file...")
		sim2 = nel.Simulator(load_filepath="./temp/simulator_state", save_filepath="./temp/simulator_state", load_time=sim2.time(), save_frequency=save_frequency)
		agents = sim2.get_agents()
		if len(agents) != 1 or type(agents[0]) != agent_type:
			print("Expected a single " + agent_type.__name__ + " in the loaded simulation.")
		agent2 = agents[0]

	# compare the two simulators and make sure they're the same
	if not compare_simulators(sim1, sim2, config,
			min_agent_position_x, min_agent_position_y,
			max_agent_position_x, max_agent_position_y):
		continue #break
