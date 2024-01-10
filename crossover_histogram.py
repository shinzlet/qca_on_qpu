import argparse
from load_qca import load_qca, assign_inputs
from qca_on_qpu import anneal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import re

parser = argparse.ArgumentParser(
                    prog='crossover_histogram',
                    description='Simulates crossovers on the QPU and collects histograms of successes',
    epilog='Example: python3 crossover_histogram.py crossovers/1\ cell\ crossover.py --arch zephyr --samples 1000 --save')

parser.add_argument('qca_file') # The name of the qca file
parser.add_argument('--spacing', default=20) # The center-to-center qca cell spacing 
parser.add_argument('--arch', default='classical') # The QPU architecture to run on (or classical)
parser.add_argument('--samples', type=int, default=500) # The number of samples that should be taken to find the minimum energy state
parser.add_argument('--title') # The graph title
parser.add_argument('--save') # The save filepath

args = parser.parse_args()

re_number = re.compile(".*(\d+).*")
crossover_type = re_number.match(args.qca_file).group(1)

# Load the QCA file
cells, drivers, inputs, outputs = load_qca(args.qca_file, True, args.spacing)

# For each input, create a BQM and anneal it. Extract statistics, outputs, and
# create visualizations.
num_input_states = 2 ** len(inputs)
input_state = 1
(all_drivers, _) = assign_inputs(drivers, inputs, input_state)
response = anneal(cells, all_drivers, samples=args.samples, qpu_arch=args.arch)

energies = np.array([record[1] for record in response.record])
weights = np.array([record[2] for record in response.record])

assert(len(outputs) == 1)
output_pos = list(outputs.values())[0]
output_idx = [*response.variables].index(output_pos)
# import pdb; pdb.set_trace()
print(response.record)
num_acceptable = sum(int(record[0][output_idx] == 1) * record[2] for record in response.record)

print(f"% in acceptable state: {num_acceptable / args.samples * 100:.3}")
width = (np.max(energies) - np.min(energies)) / energies.size * 0.8
plt.hist(energies, bins=15, weights = weights / args.samples)
plt.ylabel(r"% in state")
plt.xlabel("Energy of State")
plt.title(f"Annealed State Energies ({args.arch}, {crossover_type}-cell Crossover, N={args.samples})")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

if args.title:
    plt.title(args.title)

if args.save:
    plt.savefig(args.save, dpi=300)
else:
    plt.show()
