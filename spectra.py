import numpy as np
from matplotlib import pyplot as plt
import re
import argparse

# Commands used to generate figures for the report:
# python3 spectra.py spectra/xover_1.csv spectra/xover_3.csv --overlay --log --figwidth 9 --figheight 5 --save overlaid.png
# python3 spectra.py spectra/xover_1.csv spectra/xover_3.csv --ymax 3 --xmin 0.3 --figwidth 6 --save zoomed_crossover.png

parser = argparse.ArgumentParser(
                    prog='spectra',
                    description='runs a ',
                    epilog='Text at the bottom of help')

parser.add_argument('files', type=argparse.FileType('r'), nargs='+')
parser.add_argument('--log', action='store_true')
parser.add_argument('--overlay', action='store_true')
parser.add_argument('--save')
parser.add_argument('--ymax', type=float)
parser.add_argument('--xmin', type=float)
parser.add_argument('--figwidth', type=int, default=8)
parser.add_argument('--figheight', type=int, default=5)
args = parser.parse_args()

# List of colormaps to cycle through
colormaps = [('RdPu', 1.0, 0.5, "red"), ('YlGn', 1.0, 0.5, "green")]

re_number = re.compile(".*(\d+).*")

figsize = (args.figwidth, args.figheight)

# Create a new figure
if args.overlay:
    fig, axs = plt.subplots(1, figsize=figsize)
    axs = [axs]
else:
    fig, axs = plt.subplots(len(args.files), sharex=True, figsize=figsize)

    # Gotta love pythonic api design
    if len(args.files) == 1:
        axs = [axs]

# The first column is the normalized time, the time divided by the total
# annealing time, and the remaining columns give a set of the lowest energy
# eigenvalues relative to the ground state energy for the network as a function
# of time
for idx, file in enumerate(args.files):
    crossover_type = re_number.match(file.name).group(1)

    if args.overlay:
        ax = axs[0]
    else:
        ax = axs[idx]

    # Load data from file
    data = np.loadtxt(file, delimiter=",")
    time = data[:, 0]
    eigenvalues = data[:, 1:]
    N = eigenvalues.shape[1]

    # Set color cycle using a colormap
    cmap_name, start, stop, accent = colormaps[idx]
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(start, stop, N))

    if args.overlay:
        suffix = f" ({crossover_type}-cell)"
    else:
        suffix = ""

    # Plot each curve
    for i in range(N):

        if i == 0 and not args.log:
            # args.log has to be false: the ground state won't show up on a log plot as it's all zeros
            label = "Ground state" + suffix
        elif i == 1:
            label = "First excited state" + suffix
        elif i == N - 1:
            label = "(Higher excited states...)" + suffix
        else:
            label = None

        if args.log:
            ax.semilogy(time, eigenvalues[:, i], color = colors[i], label=label)
        else:
            ax.plot(time, eigenvalues[:, i], color = colors[i], label=label)

    # Compute the miniumum energy gap between the first and second energy levels:
    min_idx = np.argmin(eigenvalues[:, 1] - eigenvalues[:, 0])
    diff = eigenvalues[min_idx, 1] - eigenvalues[min_idx, 0]
    y_range = np.max(eigenvalues) - np.min(eigenvalues)
    label = f"Avoided Crossing (ΔE={diff:.2}GHz) {suffix}"

    if args.log:
        ax.axvline(time[min_idx], color=accent, linestyle='--', label=label)
        ax.legend(loc='lower left')
    else:
        marker_size = 10 * diff * y_range
        ax.plot(time[min_idx], (eigenvalues[min_idx, 1] + eigenvalues[min_idx, 0]) / 2, ms=marker_size, marker='o', color="None", mfc="None", mec=accent, mew=2, label=label)
        ax.legend(loc='upper right')

    if not args.overlay:
        ax.set_title(f"{crossover_type}-cell Crossover")

    if args.ymax and not args.log:
        ax.set_ylim(-0.5, args.ymax)

    if args.xmin:
        ax.set_xlim(args.xmin, time[-1])

    ax.set_xlabel("Normalized Time (unitless)")
    ax.set_ylabel("Energy / h (GHz)")

fig.suptitle("QCA Coplanar Crossover Eigenvalue Spectra")
fig.tight_layout()

if args.save:
    plt.savefig(args.save, dpi=300)
else:
    plt.show()
