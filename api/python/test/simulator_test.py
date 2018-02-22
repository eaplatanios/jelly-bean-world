import nel

items = []
items.append(nel.Item("banana", [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], True))

intensity_fn_args = [-2.0]
interaction_fn_args = [len(items)]
interaction_fn_args.extend([40.0, 200.0, 0.0, -40.0]) # parameters for interaction between item 0 and item 0

config = nel.SimulatorConfig(max_steps_per_movement=1, vision_range=10,
	patch_size=32, gibbs_num_iter=10, items=items, agent_color=[0.0, 0.0, 1.0],
	collision_policy=nel.MovementConflictPolicy.FIRST_COME_FIRST_SERVED,
	decay_param=0.5, diffusion_param=0.12, deleted_item_lifetime=2000,
	intensity_fn=nel.IntensityFunction.CONSTANT, intensity_fn_args=intensity_fn_args,
	interaction_fn=nel.InteractionFunction.PIECEWISE_BOX, interaction_fn_args=interaction_fn_args)

sim = nel.Simulator(sim_config=config)
