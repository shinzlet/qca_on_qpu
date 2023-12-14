# Computes the energy of various charge configurations that occur
# in the QCA to QPU translation

import numpy as np

charge_spacing = np.sqrt(2) * 9
cell_spacing = 20
x = np.array([cell_spacing, 0])
y = np.array([0, cell_spacing])

# box: electrons are on a square grid
box_1 = [np.array([-4.5, -4.5]), np.array([4.5, 4.5])]
box_n1 = [np.array([-4.5, 4.5]), np.array([4.5, -4.5])]
# rot: electrons are on the rotated grid. i.e. box rotated 45deg CW
rot_1 = [np.array([-charge_spacing / 2, 0]), np.array([charge_spacing / 2, 0])]
rot_n1 = [np.array([0, -charge_spacing / 2]), np.array([0, charge_spacing / 2])]

def translate(config, amount):
    return [amount + pos for pos in config]

def ensemble_energy(config):
    acc = 0
    for (i, posi) in enumerate(config):
        for j in range(i + 1, len(config)):
            posj = config[j]
            displacement = np.linalg.norm(posi - posj)
            acc += 1 / displacement

    # U = electronCharge ** 2 / 4 * pi * eps0, but we ignore this scaling factor as we're
    # working in arbitrary length units and all that matters is how these quantities are
    # sized relative to Ek
    return acc

# First, we compute what Ek is. Note that we're using a lot of arbitrary lengths and dropping
# scaling constants here, so all we can actually compute is the ratio between Ek (the normal
# box-wire kink energy), and other energy differences.
#
# As a note regarding sign conventions: In the box wire case, the energy of the kinked state
# is higher than the energy of the unkinked state. But this is not true for rotated wires,
# and is hard to even define in wires where we have a box cell next to a rotated cell.
#
# The purpose of this calculation is to compute the coupling strengths in our QUBO model - 
# i.e. for a cell i and a cell j, we want to find the multiplier E_k^(ij) s.t. the energy
# contribution caused by those cells interacting is E_k^(ij) * state_i * state_j. (state_? = +/-1).
# For normal cells, we often define E_k to be some positive value, and then introduce minus signs
# such that the lowest energy state is negative. To preserve this sign convention, we'll generalize
# this:
# E_k(normal) is defined to equal (E_kinked(normal) - E_unkinked(normal))/2, i.e. (E(i=1,j=-1)-E(i=1,j=1))/2.
# So in the case of the rot wire, [cell i (box)] [cell j (rot)],
# E_k = (E(i=1,j=-1)-E(i=1,j=1))/2
# is actually a negative quantity, unlike the E_k of a box wire.
# However, when we plug it into our simple qubo model (negative sign is there by the normal cell Ek convention)
# -E_k(rot) * state_i * state_j will be:
# => positive if state_i == state_j
#    => this is good: the energy IS in fact highest in this state (electrons are like / | or \ |
# => negative if state_i != state_j
#    => also good - this is the lowest energy state (electrons look like / |
kinked_box_wire = box_1 + translate(box_n1, x)
unkinked_box_wire = box_1 + translate(box_1, x)
# print("k")
# print(ensemble_energy(kinked_box_wire))
# print("uk")
# print(ensemble_energy(unkinked_box_wire))
# exit()
# The factor of 2 is a matter of convention - each cell involved "takes half the blame" for
# the kink, and we need the total energies to add up properly.
Ek = (ensemble_energy(kinked_box_wire) - ensemble_energy(unkinked_box_wire)) / 2

# Now, we can measure ratios by dividing other quantities by this Ek value (so that a normal kinked
# box wire's energy difference will equal 1, as we assume in the qca_on_qpu sim.

# Note that rotated wires have alternating
# signs in their cells in the lowest energy configuration.
kinked_rot_wire = rot_1 + translate(rot_1, x)
unkinked_rot_wire = rot_1 + translate(rot_n1, x)
Ek_rot = (ensemble_energy(kinked_rot_wire) - ensemble_energy(unkinked_rot_wire)) / 2
print("The kink energy of a rotated wire is", Ek_rot / Ek, "* Ek")

# Now we compute the energy difference 
eq = box_1 + translate(rot_1, x)
ne = box_1 + translate(rot_n1, x)
print((ensemble_energy(eq) - ensemble_energy(ne)) / Ek)

same = box_1 + translate(box_1, 2 * x + y)
diff = box_1 + translate(box_n1, 2 * x + y)
print("sqrt(5) distance energy", (ensemble_energy(same) - ensemble_energy(diff)) / Ek)
