import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import itertools
from utils import *

# These are ordered such that angle 0 and 2 being in the "on" state indicates
# the polarization is +1, and vice versa for the -1 and angles 1 and 3
ROT_ANGLES = np.arange(0, 4) * np.pi / 2
BOX_ANGLES = ROT_ANGLES - np.pi / 4
def draw_cell(ax, pos, pol, rot, name=None, bg_color = "#CBB0C5", edge_color = "#483745", hole_color = "None", active_edge_color="#FBBFEE", dot_color = "#FF61DD", size = 0.9, dot_spacing = 0.25, dot_size = 0.1, border_radius=0.1, linewidth=2, text_color="black"):
    inner_size = size - 2 * border_radius
    center_offset = inner_size / 2
    padding = size / 2
    x, y = pos
    background = FancyBboxPatch((x - center_offset, y - center_offset), inner_size, inner_size, f"Round, pad={border_radius}", facecolor=bg_color, edgecolor=edge_color, linewidth=linewidth)
    ax.add_patch(background)

    angles = ROT_ANGLES if rot else BOX_ANGLES
    for (i, angle) in enumerate(angles):
        style = {}
        if pol == None:
            # There is no polarization info (i.e. indeterminate)
            style = {"facecolor": "None", "edgecolor": edge_color, "linewidth": linewidth}
        elif bit_to_polarization(i % 2) == -pol:
            # An electron lives in this circle
            style = {"facecolor": dot_color, "edgecolor": active_edge_color, "linewidth": linewidth}
        else:
            # No electron here
            style = {"facecolor": hole_color, "edgecolor": edge_color, "linewidth": linewidth}

        dot_x = pos[0] + dot_spacing * np.cos(angle)
        dot_y = pos[1] + dot_spacing * np.sin(angle)
        dot = plt.Circle((dot_x, dot_y), dot_size, **style)
        ax.add_patch(dot)

    if name:
        ax.text(x, y + 0.07, name, fontweight="bold", color=text_color, fontsize=7, bbox={"facecolor": "white", "edgecolor": "None", "alpha": 0.6}, horizontalalignment='center')

def plot_circuit(cells, drivers, inputs, outputs, polarizations = {}, title = None, filename = None, **kwargs):
    fig, ax = plt.subplots()
    # outputs is a dict from name -> pos. This is the opposite of what we want here, but luckily
    # the map should be bijective, so we can flip it.
    output_lookup = dict(zip(outputs.values(), outputs.keys()))

    for pos, (pol, rot) in drivers.items():
        draw_cell(ax, pos, pol, rot, bg_color="#797C8C", dot_color="#637AF9", active_edge_color="#ADB8F7", edge_color="#2a2f58", **kwargs)

    for pos, cell in cells.items():
        pol = polarizations.get(pos)
        rot = cell["rot"]

        name = output_lookup.get(pos)
        if name == None:
            # This is a normal cell.
            draw_cell(ax, pos, pol, rot, **kwargs)
        else:
            # This is an output cell!
            draw_cell(ax, pos, pol, rot, bg_color="#C5AB98", dot_color="#FBB582", active_edge_color="#FFCDA8", name=name, **kwargs)

    for name, (pos, rot) in inputs.items():
        draw_cell(ax, pos, polarizations.get(pos), rot, name=name, bg_color="#b7c8c5", dot_color="#80F9E4", active_edge_color="#B4F9ED", edge_color="#28564E", **kwargs)

    ax.set_aspect('equal', adjustable='box')

    # Determining the size of the grid
    x_coords, y_coords = zip(*(list(cells.keys()) + list(drivers.keys()) + [i[0] for i in inputs.values()]))
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    ax.set_xlim(x_min-0.5, x_max + 0.5)
    ax.set_ylim(y_max + 0.5, y_min-0.5)

    fig.suptitle(title)

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
