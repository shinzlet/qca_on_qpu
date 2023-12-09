from qcadtrans import QCACircuit
from bigham import extract_polarization

# Different cell IDs that QCADesigner uses
NORMAL_T = 0
INPUT_T  = 1
OUTPUT_T = 2
FIXED_T  = 3

def assign_inputs(drivers, inputs, input_state):
    # Create a copy of the drivers array that converts the input state
    # into fixed polarization cells:
    all_drivers = drivers
    for input_idx, (name, pos) in enumerate(inputs.items()):
        all_drivers[pos] = extract_polarization(input_state, input_idx)

    return all_drivers

def load_qca(filename, spacing = 20):
    # Supply the file name to the QCADesigner file and it will be parsed by the 
    # QCACircuit class.
    circuit = QCACircuit(fname=filename, verbose=False)

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
        x = int(node["x"] / spacing)
        y = int(node["y"] / spacing)
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

    return cells, drivers, inputs, outputs
