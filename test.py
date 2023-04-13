import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

viridis_cmap = cm.get_cmap('viridis')
colors = viridis_cmap(np.linspace(0, 1, 5))
hex_colors = [mcolors.to_hex(c) for c in colors]

print(hex_colors)