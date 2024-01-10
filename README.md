# QCA on QPU

This is a simulation framework for running Quantum-dot Cellular Automata (QCA) Circuits on D-Wave's
quantum annealer hardware. It was written as part of a school project.

The software is able to read a `.qca` file from QCADesigner [1] and capture the interactions between close
cells using the kink energy for their given geometry (i.e. the energy splitting between the homogenous and
heterogeneous states). This produces a pruned undirected weighted graph representation of the circuit, which
D-Wave's Ocean SDK uses as a binary quadratic model (the QCA circuit is effectively an Ising hamiltonian).

This model is then either simulated classically on the cpu (`--arch classical`) or on a D-Wave quantum computer
(`--arch pegasus`). The results are collected, reinterpreted, and plotted. This process is repeated once for
every input state that the circuit can be in, so that a circuit's truth table is effectively computed (QCADesigner
allows cells to be specified as inputs).

A number of plotting options are available. The `--save` and `--title` options accept strings for the savefile name
and title name of each plot produced. The name of the input state will be interpolated into these strings. For example,
using `--save "images/%s.png"` might produce a file called `images/A = -1.png` and `images/A = 1.png` (assuming
a single input called "A").

## Example Commands
First, make sure you've entered the venv and have your `DWAVE_API_TOKEN` environment variable set.
A simple invocation looks like this:
`python3 main.py sparse\ XOR/unclocked/design.qca --samples 1000 --arch zephyr --title "XOR Gate (zephyr, N=1000, state=%s)" --save "xor zephyr %s.png"`

By default, the lowest energy configuration is the one plotted. To plot the most common state with an incorrect output, rather than the ground state, the `--broken` flag can be passed:
`python3 main.py sparse\ XOR/unclocked/design.qca --samples 1000 --arch zephyr --title "Top XOR Gate Failure Mode (zephyr, N=1000, state=%s)" --save "broken xor zephyr %s.png" --broken`.

## Example Outputs
### XOR Gate
Note that there are much simpler XOR gates possible in QCA, but they require clocking (which this software does not
support).
![A simulated QCA XOR gate. The output state is correct here, as +1 xor -1 = +1.](https://raw.githubusercontent.com/shinzlet/qca_on_qpu/main/images/xor%20zephyr/xor%20zephyr%20B%20%3D%20-1%2C%20A%20%3D%201.png?token=GHSAT0AAAAAACMGGSK37AZG2B4LGJI3SJPUZM6Y64Q)

### Majority Gate
![A majority gate](https://raw.githubusercontent.com/shinzlet/qca_on_qpu/main/validation/majority/A%20%3D%20-1%2C%20C%20%3D%201%2C%20B%20%3D%201.png?token=GHSAT0AAAAAACMGGSK25CMWJXUISYR36LHAZM6ZABQ)

## References
K. Walus, T. J. Dysart, G. A. Jullien and R. A. Budiman, "QCADesigner: a rapid design and Simulation tool for quantum-dot cellular automata," in IEEE Transactions on Nanotechnology, vol. 3, no. 1, pp. 26-31, March 2004, doi: 10.1109/TNANO.2003.820815.
