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

args = parser.parse_args()

# Load the QCA file
cells, drivers, inputs, outputs = load_qca(args.qca_file, args.spacing)

# For each input, create a BQM and anneal it. Extract statistics, outputs, and
# create visualizations.
num_input_states = 2 ** len(inputs)
for input_state in range(num_input_states):
    all_drivers = assign_inputs(drivers, inputs, input_state)
    response = anneal(cells, all_drivers)

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
    print(output_state)
    for output, output_pos in outputs.items():
        print(f"{output}: {output_state[output_pos]}")
    # plot_circuit({**output_state, **all_drivers}, list(all_drivers), "hi", f"{input_state}.png")
    break
