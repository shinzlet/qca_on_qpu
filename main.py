import argparse
from qca_plotting import plot_circuit
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
parser.add_argument('--samples', type=int, default=500) # The number of samples that should be taken to find the minimum energy state
parser.add_argument('--ignore-rotated', action='store_true', dest="ignore_rotated") # Deletes rotated cells if true
parser.add_argument('--only-plot', action='store_true', dest="only_plot")
parser.add_argument('--title') # The title: %s is where the state info should be appended (unless only plot)
parser.add_argument('--save') # The save filepath: %s is where the state info should be appended (unless only plot)
parser.add_argument('--no-plot', action='store_true', dest='no_plot')
parser.add_argument('--broken', action='store_true', dest='broken') # plots the most common state whose outputs differ from the ground state

args = parser.parse_args()

# Load the QCA file
cells, drivers, inputs, outputs = load_qca(args.qca_file, args.ignore_rotated, args.spacing)

if args.only_plot:
    plot_circuit(cells, drivers, inputs, outputs, title=args.title, filename=args.save)
    exit()

if args.broken and args.arch == 'classical':
    print("The classical annealer gives results in a different format than the actual qpu. Because of this, --broken is unsupported for classical simulation. Aborting")
    exit(1)

# For each input, create a BQM and anneal it. Extract statistics, outputs, and
# create visualizations.
num_input_states = 2 ** len(inputs)
for input_state in range(num_input_states):
    (all_drivers, state_name) = assign_inputs(drivers, inputs, input_state)
    response = anneal(cells, all_drivers, samples=args.samples, qpu_arch=args.arch)

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
        # import pdb; pdb.set_trace()

    ground_state_outputs = {}
    for output, output_pos in outputs.items():
        output_idx = [*response.variables].index(output_pos)
        ground_state_outputs[output] = states[output_idx]

    # Here, states, energy, and count all correspond to the ground state
    # if we want to find the best broken state, we fork here
    if args.broken:
        found = False
        for record in response.record:
            if found:
                break

            states, energy, count, _ = record
            for output, output_pos in outputs.items():
                output_idx = [*response.variables].index(output_pos)
                if states[output_idx] != ground_state_outputs[output]:
                    found = True
        print(f"{count} / {args.samples} ({100 * count / args.samples:.2f}%) of samples were in the broken state chosen")

    output_state = dict(zip([*response.variables], states))

    if not args.broken:
        print(f"============= State {state_name} =================")
        print(f"{count} / {args.samples} ({100 * count / args.samples:.2f}%) of samples found the ground state")
        for output, output_pos in outputs.items():
            print(f"output '{output}':")
            print(f"  Ground state configuration: {ground_state_outputs[output]}")

            pos1_count = 0
            output_idx = [*response.variables].index(output_pos)
            for record in response.record:
                if record[0][output_idx] == 1:
                    pos1_count += 1
            neg1_count = args.samples - pos1_count

            print(f"  {pos1_count}/{args.samples} ({100 * pos1_count / args.samples:.2f}%) of states had {output} = +1")
            print(f"  {neg1_count}/{args.samples} ({100 * neg1_count / args.samples:.2f}%) of states had {output} = -1")
            print("")

    if args.no_plot:
        continue

    # The output state only contains the state of cells which the QPU solved. For every polarization,
    # we inject the driver states back in:
    polarizations = {pos: pol for pos, (pol, _) in all_drivers.items()}
    polarizations = {**polarizations, **output_state}
    filename = None
    if args.save:
        filename = args.save % state_name
    title = None
    if args.title:
        title = args.title % state_name
    plot_circuit(cells, drivers, inputs, outputs, polarizations = polarizations, title=title, filename=filename)
