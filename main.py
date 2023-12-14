import argparse
from bigham import plot_circuit
from load_qca import load_qca, assign_inputs
from qca_on_qpu import anneal
import numpy as np

parser = argparse.ArgumentParser(
                    prog='qca_on_qpu',
                    description='runs a ',
                    epilog='Text at the bottom of help')

parser.add_argument('qca_file') # The name of the qca file
parser.add_argument('--spacing', default=20) # The center-to-center qca cell spacing 
parser.add_argument('--arch', default='classical') # The QPU architecture to run on (or classical)
parser.add_argument('--samples', default=500) # The number of samples that should be taken to find the minimum energy state
parser.add_argument('--ignore-rotated', action='store_true', dest="ignore_rotated") # Deletes rotated cells if true

args = parser.parse_args()

# Load the QCA file
cells, drivers, inputs, outputs = load_qca(args.qca_file, args.ignore_rotated, args.spacing)

# For each input, create a BQM and anneal it. Extract statistics, outputs, and
# create visualizations.
num_input_states = 2 ** len(inputs)
for input_state in range(num_input_states):
    all_drivers = assign_inputs(drivers, inputs, input_state)
    response = anneal(cells, all_drivers, samples=args.samples, qpu_arch=args.arch)

    # import pdb; pdb.set_trace()
    # The classical annealer outputs very different data. Tallying is broken and
    # energies aren't sorted.
    if args.arch == 'classical':
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
    else:
        # On real hardware, the tallying works and we can use the first (lowest energy) soln.
        states, energy, count, _ = response.record[0]

    print(states)
    print(energy)
    print(count)
    output_state = dict(zip([*response.variables], states))
    print(outputs)
    print("BQM:")
    print(output_state)

    for output, output_pos in outputs.items():
        print(f"{output}: {output_state[output_pos]}")

    # The output state only contains the state of cells which the QPU solved. For every polarization,
    # we inject the driver states back in:
    polarizations = {pos: pol for pos, (pol, _) in all_drivers.items()}
    polarizations = {**polarizations, **output_state}
    plot_circuit(cells, drivers, inputs, outputs, polarizations = polarizations, title=f"state {input_state}")#, filename=f"{input_state}.png")
    break
