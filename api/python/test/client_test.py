import nel
from timeit import default_timer
from simulator_test import EasterlyAgent, make_config
from threading import Lock, Condition

cv = Condition(Lock())
waiting = False
running = True

def on_step():
	global waiting
	cv.acquire()
	waiting = False
	cv.notify()
	cv.release()

def on_lost_connection():
	global running
	running = False
	cv.notify()

# connect to server
sim = nel.Simulator(server_address="localhost", on_step_callback=on_step, on_lost_connection_callback=on_lost_connection)

# add agents to simulator
agents = []
while len(agents) < 1:
	print("adding agent " + str(len(agents)))
	try:
		agent = EasterlyAgent(sim)
		agents.append(agent)
	except nel.AddAgentError:
		pass

	# move agents to avoid collision at (0,0)
	waiting = True
	for agent in agents:
		sim.move(agent, agent.next_move(), 1)

# start main loop
start_time = default_timer()
elapsed = 0.0
sim_start_time = sim.time()
while running:
	if not waiting:
		waiting = True
		for agent in agents:
			result = sim.move(agent, agent.next_move(), 1)
	if default_timer() - start_time > 1.0:
		elapsed += default_timer() - start_time
		print(str((sim.time() - sim_start_time) / elapsed) + " simulation steps per second.")
		start_time = default_timer()

	# wait for simulation to advance or for 1 second
	cv.acquire()
	if waiting:
		cv.wait(timeout=1.0)
	cv.release()
