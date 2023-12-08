import argparse
from qcadtrans import QCACircuit

from bigham import *

parser = argparse.ArgumentParser(
                    prog='qca_on_qpu',
                    description='runs a ',
                    epilog='Text at the bottom of help')

parser.add_argument('qca_file') # The name of the qca file
parser.add_argument('--spacing', default=20) # The center-to-center qca cell spacing 

args = parser.parse_args()

# Different cell IDs
NORMAL_T = 0
INPUT_T  = 1
OUTPUT_T = 2
FIXED_T  = 3

# The kink energy for adjacent cells (i.e. cells with a center to center distance
# of 1 unit)
Ek0 = 1.0
# Tunnelling Energy
t = 0.01 * Ek0

# Supply the file name to the QCADesigner file and it will be parsed by the 
# QCACircuit class.
circuit = QCACircuit(fname=args.qca_file, verbose=False)

# Map from (x, y) tuples to the QCA Node data - for normal and output types only
cells = {}
# Map from (x, y) tuples of position to polarization strength
drivers = {}
# string name -> location. Inputs are not in the `cells` or `drivers` dictionaries: to actually
# produce a hamiltonian, a new drivers dict needs to be created that assigns input cells their set
# polarization values.
inputs = {}
# string name -> pos tuple. output cells DO exist in the cells array - they're normal cells, they
# just need to be measured at the end of simulation.
outputs = {}

# Destructure the heterogeneous cell listing 
for i in range(len(circuit.nodes)):
    node = circuit.nodes[i]

    cf = node["cf"]
    x = int(node["x"] / args.spacing)
    y = int(node["y"] / args.spacing)
    pos = (x, y)

    if cf == NORMAL_T:
        cells[pos] = node
    elif cf == INPUT_T:
        inputs[node["name"]] = pos
    elif cf == OUTPUT_T:
        cells[pos] = node
        outputs[node["name"]] = pos
    elif cf == FIXED_T:
        drivers[pos] = float(node["pol"])

num_input_states = 2 ** len(inputs)
for state in range(num_input_states):
    all_drivers = drivers

    for input_idx, (name, pos) in enumerate(inputs.items()):
        all_drivers[pos] = extract_polarization(state, input_idx)

    # print(full_hamiltonian(cells, all_drivers, Ek0, t))
    result = simulate_circuit(cells, all_drivers, Ek0, t)
    plot_circuit({**result, **all_drivers}, all_drivers, f"Drivers: {all_drivers}", f"images/{state}.png")
    # break
# def full_hamiltonian(circuit, drivers, Ek0, t):
