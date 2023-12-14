# maps 0 to -1 and 1 to 1
def bit_to_polarization(bit):
    return 2 * bit - 1

def extract_bit(source, i):
    return (source & (1 << i)) >> i

def extract_polarization(source, i):
    return bit_to_polarization(extract_bit(source, i))
