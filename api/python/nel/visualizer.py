from __future__ import absolute_import, division, print_function

#try:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Rectangle
modules_loaded = True
#except ImportError:
#	modules_loaded = False

__all__ = ['MapVisualizer']

RECTANGLE_RADIUS = 0.4
ITEM_RADIUS = 0.4
MAXIMUM_SCENT = 6.0

class MapVisualizer(object):
	def __init__(self, simulator_config, bottom_left, top_right):
		if not modules_loaded:
			raise ImportError("numpy and matplotlib are required to use MapVisualizer.")
		plt.ion()
		self._config = simulator_config
		self._xlim = [bottom_left[0], top_right[0]]
		self._ylim = [bottom_left[1], top_right[1]]
		self._fig, self._ax = plt.subplots()
		self._fig.set_size_inches((8, 8))

	def draw(self, map):
		n = self._config.patch_size
		self._ax.clear()
		self._ax.set_xlim(self._xlim)
		self._ax.set_ylim(self._ylim)
		for patch in map:
			(patch_position, fixed, scent, vision, items, agents) = patch
			color = (0, 0, 0, 1) if fixed else (0, 0, 0, 0.3)

			vertical_lines = np.empty((n + 1, 2, 2))
			vertical_lines[:,0,0] = patch_position[0]*n + np.arange(n + 1) - 0.5
			vertical_lines[:,0,1] = patch_position[1]*n - 0.5
			vertical_lines[:,1,0] = patch_position[0]*n + np.arange(n + 1) - 0.5
			vertical_lines[:,1,1] = patch_position[1]*n + n - 0.5
			vertical_line_col = LineCollection(vertical_lines, colors=color, linewidths=0.4, linestyle='solid')
			self._ax.add_collection(vertical_line_col)

			horizontal_lines = np.empty((n + 1, 2, 2))
			horizontal_lines[:,0,0] = patch_position[0]*n - 0.5
			horizontal_lines[:,0,1] = patch_position[1]*n + np.arange(n + 1) - 0.5
			horizontal_lines[:,1,0] = patch_position[0]*n + n - 0.5
			horizontal_lines[:,1,1] = patch_position[1]*n + np.arange(n + 1) - 0.5
			horizontal_line_col = LineCollection(horizontal_lines, colors=color, linewidths=0.4, linestyle='solid')
			self._ax.add_collection(horizontal_line_col)

			patches = []
			for agent in agents:
				patches.append(Rectangle((agent[0] - RECTANGLE_RADIUS, agent[1] - RECTANGLE_RADIUS),
						2*RECTANGLE_RADIUS, 2*RECTANGLE_RADIUS, facecolor=self._config.agent_color,
						edgecolor=(0,0,0), linestyle='solid', linewidth=0.4))
			for item in items:
				(type, position) = item
				patches.append(Circle(position, ITEM_RADIUS, facecolor=self._config.items[type].color,
						edgecolor=(0,0,0), linestyle='solid', linewidth=0.4))

			# convert 'scent' to a numpy array and transform into a subtractive color space (so zero is white)
			scent_img = np.clip(np.array(scent) / MAXIMUM_SCENT, 0.0, 1.0)
			scent_img = 1.0 - np.dot(scent_img, np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
			self._ax.imshow(np.rot90(scent_img),
					extent=(patch_position[0]*n - 0.5, patch_position[0]*n + n - 0.5,
							patch_position[1]*n - 0.5, patch_position[1]*n + n - 0.5))

			self._ax.add_collection(PatchCollection(patches, match_original=True))

		plt.pause(1.0e-16)
		plt.draw()
