import math
import numpy as np
import itertools

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

ADJACENT_DIRECTIONS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
DIAGONAL_DIRECTIONS = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])

# The kink energy for adjacent cells (i.e. cells with a center to center distance
# of 1 unit)
Ek0 = 1.0
# Tunnelling Energy
t = 0.01 * Ek0

def anneal(cells, drivers, samples = 500, qpu_arch = 'classical'):
    # get DWave sampler and target mapping edgelist
    if qpu_arch == 'classical':
        # print('Choosing classical sampler...')
        sampler = neal.SimulatedAnnealingSampler()
    else:
        # print('Choosing solver...')
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
            # print(f'The pre-programmed D-Wave solver name for architecture '
            #         '\'{qpu_arch}\' is not available. Find the latest available '
            #         'solvers by:\n'
            #         'from dwave.cloud import Client\nclient = Client.from_config()\n'
            #         'client.get_solvers()\nAnd update this script.')
            raise

        # get the specified QPU
        dwave_sampler = DWaveSampler(solver=solver)

        # run the problem
        use_result = []
        sampler = None
        response = None
        # print('Choosing D-Wave QPU as sampler...')
        sampler = EmbeddingComposite(dwave_sampler)

    bqm = construct_bqm(cells, drivers)
    response = sampler.sample(bqm, num_reads=samples)
    # print('Problem completed from selected sampler.')

    return response


def construct_bqm(cells, drivers):
    # Using ICHA, and only considering direct neighbours (diagonals included)
    linear = {}
    quadratic = {}

    def sum_neighbours(pos, rot, directions, cb):
        pos = np.array(pos)
        total = 0
        for i in range(directions.shape[0]):
            total += cb(pos, rot, pos + directions[i, :])
        return total

    def driver_contribution(pos, rot, other_pos):
        other_pos_tuple = (other_pos[0], other_pos[1])

        if other_pos_tuple in drivers:
            other_pol, other_rot = drivers[other_pos_tuple]

            # We only consider interaction if the cells have the same
            # rotation. Alternately rotated cells have no interaction
            # due to the symmetry of the problem.
            if other_rot == rot:
                r = np.linalg.norm(pos - other_pos)

                # The cells should alternate if they're in a rotated wire
                if rot:
                    relationship = -1
                else:
                    relationship = 1

                return relationship * other_pol / r ** 5

        return 0

    def scale_func(f, multiplier):
       return lambda *args, **kwargs: -f(*args, **kwargs)

    cell_order = list(cells)
    for (i, pos_i) in enumerate(cells):
        # The linear term includes the effect of drivers on this cell.
        linear[pos_i] = 0
        rot_i = cells[pos_i]["rot"]
        linear[pos_i] += sum_neighbours(pos_i, rot_i, ADJACENT_DIRECTIONS, scale_func(driver_contribution, -Ek0))
        linear[pos_i] += sum_neighbours(pos_i, rot_i, DIAGONAL_DIRECTIONS, scale_func(driver_contribution, Ek0))

        # Cells that are adjacent to this one should have a negative energy
        # contribution when the quadratic term is positive (i.e. they are of
        # the same sign) and a positive sign when the quadratic term is negative.

        # The quadratic term includes the effect of adjacent non-driver cells. This could also
        # be computed by a fixed-time neighbour lookup as we do for the linear case,
        # but the code would get more complex because we need to know the indexes
        # and avoid double-counting embedding graph edges. Instead we just eat this O(n^2),
        # because n is small.
        for j in range(i+1, len(cells)):
            pos_j = cell_order[j]
            # We assume (and it is true when cell i and j are directly adjacent) that
            # there is no interaction between rotated and unrotated cells.
            if cells[pos_j]["rot"] != rot_i:
                continue

            # -1 if the cells want to alternate, 1 if they don't alternate.
            relationship = -1 if rot_i else 1

            # If cells i and j are adjacent, 
            r = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
            if (r > 0.99 and r < 1.01) or (r > 1.99 and r < 2.01): # r ~= 1, r ~= 2
                # adjacent (negative energy terms means they should be the same signs)
                quadratic[(pos_i, pos_j)] = -Ek0 / r ** 5 * relationship
            elif r > 1.4 and r < 1.42: # r ~= sqrt(2)
                # diagonal (positive energy term means they should be opposite signs)
                quadratic[(pos_i, pos_j)] = Ek0 / r ** 5 * relationship

    # construct a bqm containing the provided self-biases (linear) and couplings
    # (quadratic). Specify the problem as SPIN (Ising).
    # print('Constructing BQM...')
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, dimod.SPIN)
    return bqm
