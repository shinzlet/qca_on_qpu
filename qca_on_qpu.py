import argparse
from bigham import extract_polarization, plot_circuit
from qcadtrans import QCACircuit

import dwave
import dwave.embedding
import dwave.inspector
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.cloud import Client
from dwave.cloud.exceptions import SolverNotFoundError
import dimod
from dimod.reference.samplers import ExactSolver
from minorminer import find_embedding
import neal

# general math and Python dependencies
import math
import numpy as np
import itertools

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

ADJACENT_DIRECTIONS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
DIAGONAL_DIRECTIONS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

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

# # TODO: REMOVE THIS SHIM
# cells = {(1, 0), (2, 1), (3, 1), (4, 0)}
# drivers = {(0, 0): 1}
# inputs = {}
# outputs = {}
# # ENDTODO

# TODO: iterate over all input states
input_state = 1
all_drivers = drivers

for input_idx, (name, pos) in enumerate(inputs.items()):
    all_drivers[pos] = extract_polarization(input_state, input_idx)

# Using ICHA, and only considering direct neighbours (diagonals included)
linear = {}
quadratic = {}

def sum_neighbours(pos, directions, cb):
    pos = np.array(pos)
    total = 0
    for i in range(directions.shape[0]):
        total += cb(pos, pos + directions[i, :])
    return total

def driver_contribution(pos, other_pos):
    r = np.linalg.norm(pos - other_pos)
    other_pos_tuple = (other_pos[0], other_pos[1])
    if other_pos_tuple in all_drivers:
        return all_drivers[other_pos_tuple] / r ** 5
    return 0

def scale_func(f, multiplier):
   return lambda *args, **kwargs: -f(*args, **kwargs)

cell_order = list(cells)
for (i, pos_i) in enumerate(cells):
    # The linear term includes the effect of drivers on this cell.
    linear[pos_i] = 0
    linear[pos_i] += sum_neighbours(pos_i, ADJACENT_DIRECTIONS, scale_func(driver_contribution, -Ek0))
    linear[pos_i] += sum_neighbours(pos_i, DIAGONAL_DIRECTIONS, scale_func(driver_contribution, Ek0))

    # Cells that are adjacent to this one should have a negative energy
    # contribution when the quadratic term is positive (i.e. they are of
    # the same sign) and a positive sign when the quadratic term is negative.

    # The quadratic term includes the effect of adjacent normal cells.
    for j in range(i+1, len(cells)):
        pos_j = cell_order[j]
        # If cells i and j are adjacent, 
        r = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
        if r > 0.99 and r < 1.01:
            # adjacent (negative energy terms means they should be the same signs)
            quadratic[(pos_i, pos_j)] = -Ek0
        elif r > 1.4 and r < 1.42:
            # diagonal (positive energy term means they should be opposite signs)
            quadratic[(pos_i, pos_j)] = Ek0 / r ** 5

# print(all_drivers)
# print(list(cells))
# print(linear)
# print(quadratic)
# exit()

# # create edgelist (note that {} initializes Python dicts)
# linear = {}         # qubit self-bias
# quadratic = {}      # inter-qubit bias
# for i in range(N):
#     linear[i] = h[i]
#     for j in range(i+1, N):
#         if J[i][j] != 0:
#             quadratic[(i,j)] = J[i][j]
# 
use_classical = True
qpu_arch = 'pegasus'

# construct a bqm containing the provided self-biases (linear) and couplings
# (quadratic). Specify the problem as SPIN (Ising).
print('Constructing BQM...')
bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, dimod.SPIN)

# get DWave sampler and target mapping edgelist
if use_classical:
    print('Choosing classical sampler...')
    sampler = neal.SimulatedAnnealingSampler()
else:
    print('Choosing solver...')
    client = Client.from_config()
    solver = None
    qca_arch = 'pegasus'
    try:
        if qpu_arch == 'zephyr':
            solver = client.get_solver('Advantage2_prototype1.1').id
        elif qpu_arch == 'pegasus':
            solver = client.get_solver('Advantage_system4.1').id
        elif qpu_arch == 'chimera':
            solver = client.get_solver('DW_2000Q_6').id
        else:
            raise ValueError('Specified QPU architecture is not supported.')
    except SolverNotFoundError:
        print(f'The pre-programmed D-Wave solver name for architecture '
                '\'{qpu_arch}\' is not available. Find the latest available '
                'solvers by:\n'
                'from dwave.cloud import Client\nclient = Client.from_config()\n'
                'client.get_solvers()\nAnd update this script.')
        raise

    # get the specified QPU
    dwave_sampler = DWaveSampler(solver=solver)

    # run the problem
    use_result = []
    sampler = None
    response = None
    print('Choosing D-Wave QPU as sampler...')
    sampler = EmbeddingComposite(dwave_sampler)

response = sampler.sample(bqm, num_reads=500)
print('Problem completed from selected sampler.')

# Find minimum energy solution
winning_record = response.record[0]
for record in response.record:
    # Find the minimum energy state
    if record[1] < winning_record[1]:
        winning_record = record

count = 0
for record in response.record:
    # Find the minimum energy state
    if np.all(record[0] == winning_record[0]):
        count += 1

states, energy, _ = winning_record
print(states)
print(energy)
print(count)
output_state = dict(zip([*response.variables], states))
print(outputs)
print("BQM:")
print(linear)
print(quadratic)
print("")
print(output_state)
for output, output_pos in outputs.items():
    print(f"{output}: {output_state[output_pos]}")

plot_circuit({**output_state, **all_drivers}, list(all_drivers), "hi", "output.png")
