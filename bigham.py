import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import itertools

def extract_bit(source, i):
    return (source & (1 << i)) >> i

# maps 0 to -1 and 1 to 1
def bit_to_polarization(bit):
    return 2 * bit - 1

def extract_polarization(source, i):
    return bit_to_polarization(extract_bit(source, i))

pauli = [
    np.array([[1, 0], [0, 1]]),
    np.array([[0, 1], [1, 0]]),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]])
]

def full_hamiltonian(circuit, drivers, Ek0, t):
    # The circuit is entered as a set, but it is useful to have a canonical
    # ordering for its elements.
    cell_order = list(circuit)
    # print(cell_order)

    # We construct a base hamiltonian containing just the tunnelling energies,
    # which are trivial to compute. The state energies are added later.
    H1 = -t * pauli[1]
    # Nc is the number of cells
    Nc = len(cell_order)
    # N is the size of the hamiltonian / number of definite states
    N = 2 ** Nc
    # H is the WIP variable storing the hamiltonian
    H = np.zeros((N, N))
    for i in range(Nc):
        # Compute H1(i)
        term = np.ones(1)
        for j in range(Nc):
            if j == i:
                term = np.kron(H1, term)
            else:
                term = np.kron(pauli[0], term)

        # Add it to H
        H += term

    for state in range(N):
        # This is a binary counter. The Nth LSB is 0 if cell_order[N] is in state -1,
        # and 1 if cell_order[N] is in state +1.
        # print(state, (state & (1 << 0)) >> 0, (state & (1 << 1)) >> 1)

        E = 0
        # Look at each unique pair of cells (A, B). Compute r, the distance between
        # them. Add +/-1 E_k/(2 * r**5) to the energy depending on their relative
        # polarization (-ve == same polarization).
        for (A, B) in itertools.combinations(range(Nc), 2):
            # 1 = +1, 0 = -1.
            A_polarization = extract_polarization(state, A)
            B_polarization = extract_polarization(state, B)
            r = np.linalg.norm(np.array(cell_order[A]) - np.array(cell_order[B]))
            if A_polarization == B_polarization:
                E -= Ek0 / (2 * r ** 5)
            else:
                E += Ek0 / (2 * r ** 5)


        # For each driver, look at each cell. Repeat the above, but use +/- E_k P_D/2
        for (driver_pos, driver_polarization) in drivers.items():
            for cell_id in range(len(cell_order)):
                # Here, we actually transform from {0, 1} -> {-1, 1}.
                cell_polarization = extract_polarization(state, cell_id)
                r = np.linalg.norm(np.array(driver_pos) - np.array(cell_order[cell_id]))
                # If the polarizations are the same sign, we want a negative energy
                # contribution. If they are opposite signs, we want a positive contribution.
                # The magnitude should be equal to driver_polarization * Ek0 / (2 * r**5).
                # Let's think about the quantity `cell_polarization * driver_polarization`.
                # if both are the same sign, it is positive. If one is a different sign than
                # the other, it is negative. So, `-(cell_polarization * driver_polarization)`
                # always has the sign we want on the energy contribution. Also, because the
                # cell polarization is +/-1, the magnitude is equal to |driver_polarization|.
                # Hence the contribution can be written
                E += -(cell_polarization * driver_polarization) * Ek0 / (2 * r**5)

        # Set the diagonal of H to the energy of this state:
        H[state, state] = E

    return (H, cell_order)

def simulate_circuit(circuit, drivers, Ek0, t):
    (H, cell_order) = full_hamiltonian(circuit, drivers, Ek0, t)
    # Find the determinate states of H (each will be a superposition of
    # our binary basis states))
    vals, vecs = np.linalg.eig(H)
    # The state we will observe in experiment is always the ground state by
    # design of QCA
    ground_state = vecs[:, np.argmin(vals)]
    # Now we find the probability of each basis state in the ground state.
    state_amplitudes = ground_state ** 2
    # The highest amplitude is assigned to the state we're most likely to actually
    # measure:
    most_likely_state = np.argmax(np.abs(ground_state))
    # And we can extract the states (0/1) by pulling out each bit.
    result = {cell: extract_polarization(most_likely_state, i) for (i, cell) in enumerate(cell_order)}
    return result

# # Chatgpt made this function:
# # https://chat.openai.com/share/4a11dc92-1f0f-4475-9ba8-f9692f2eebc3
# # I just adjusted the colors chosen and tweaked it a bit.
# def plot_circuit(data, drivers, title, filename):
#     """
#     Plot a grid where each square is colored based on the provided data.
#     Green squares represent a value of 1, and gray squares represent a value of -1.
#     The background is white.
# 
#     :param data: A dictionary where keys are (x, y) tuples and values are either 1 or -1.
#     """
#     # Extracting all coordinates and corresponding values
#     x_coords, y_coords = zip(*data.keys())
#     values = data.values()
# 
#     # Determining the size of the grid
#     x_min, x_max = min(x_coords), max(x_coords)
#     y_min, y_max = min(y_coords), max(y_coords)
# 
#     # Creating a figure and axis with a white background
#     fig, ax = plt.subplots()
#     ax.set_facecolor('white')
# 
#     # Plotting each square
#     for (x, y), value in data.items():
#         if (x, y) in drivers:
#             color = '#86a7cb' if value == 1 else 'black'
#         else:
#             color = '#007aff' if value == 1 else '#444'
#         ax.fill_between([x-0.5, x+0.5], y-0.5, y+0.5, color=color)
# 
#     # Setting the limits and aspect
#     ax.set_xlim(x_min-0.5, x_max + 0.5)
#     ax.set_ylim(y_max + 0.5, y_min-0.5)
#     ax.set_aspect('equal', adjustable='box')
# 
#     fig.suptitle(title)
#     plt.savefig(filename, dpi=300)

# These are ordered such that angle 0 and 2 being in the "on" state indicates
# the polarization is +1, and vice versa for the -1 and angles 1 and 3
ROT_ANGLES = np.arange(0, 4) * np.pi / 2
BOX_ANGLES = ROT_ANGLES + np.pi / 4
def draw_cell(ax, pos, pol, rot, name=None, bg_color = "gray", hole_color = "orange", dot_color = "black", size = 0.9, dot_spacing = 0.3, dot_size = 0.08):
    padding = size / 2
    x, y = pos
    ax.fill_between([x - padding, x + padding], y - padding, y + padding, color=bg_color)

    angles = ROT_ANGLES if rot else BOX_ANGLES
    for (i, angle) in enumerate(angles):
        style = {}
        if pol == None:
            # There is no polarization info (i.e. indeterminate)
            style = {"facecolor": dot_color, "edgecolor": "black", "hatch": "////////"}
        elif bit_to_polarization(i % 2) == pol:
            # An electron lives in this circle
            style = {"color": dot_color}
        else:
            # No electron here
            style = {"color": hole_color}

        dot_x = pos[0] + dot_spacing * np.cos(angle)
        dot_y = pos[1] + dot_spacing * np.sin(angle)
        dot = plt.Circle((dot_x, dot_y), dot_size, **style)
        ax.add_patch(dot)
    # else:

def plot_circuit(cells, drivers, inputs, outputs, polarizations = {}, title = None, filename = None, **kwargs):
    fig, ax = plt.subplots()
    # outputs is a dict from name -> pos. This is the opposite of what we want here, but luckily
    # the map should be bijective, so we can flip it.
    output_lookup = dict(zip(outputs.values(), outputs.keys()))

    for pos, (pol, rot) in drivers.items():
        draw_cell(ax, pos, pol, rot, bg_color="blue", **kwargs)

    for pos, cell in cells.items():
        pol = polarizations.get(pos)
        rot = cell["rot"]

        name = output_lookup.get(pos)
        if name == None:
            # This is a normal cell.
            draw_cell(ax, pos, pol, rot, **kwargs)
        else:
            # This is an output cell!
            draw_cell(ax, pos, pol, rot, bg_color="yellow", name=name, **kwargs)

    for name, (pos, rot) in inputs.items():
        draw_cell(ax, pos, polarizations.get(pos), rot, name=name, bg_color="pink", **kwargs)

    ax.set_aspect('equal', adjustable='box')
    fig.suptitle(title)

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)
