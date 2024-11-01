import pymsym

elements = {
    1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
    21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
    31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
    41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
    51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
    61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
    71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
    81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
    91: 'Pa', 92: 'U'
}

# cf. https://github.com/patonlab/GoodVibes/blob/master/goodvibes/thermo.py#L23
SYMMNO_BY_POINT_GROUP = {
    "C1": 1,
    "Cs": 1,
    "Ci": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
    "C6": 6,
    "C7": 7,
    "C8": 8,
    "D2": 4,
    "D3": 6,
    "D4": 8,
    "D5": 10,
    "D6": 12,
    "D7": 14,
    "D8": 16,
    "C2v": 2,
    "C3v": 3,
    "C4v": 4,
    "C5v": 5,
    "C6v": 6,
    "C7v": 7,
    "C8v": 8,
    "C2h": 2,
    "C3h": 3,
    "C4h": 4,
    "C5h": 5,
    "C6h": 6,
    "C7h": 7,
    "C8h": 8,
    "D2h": 4,
    "D3h": 6,
    "D4h": 8,
    "D5h": 10,
    "D6h": 12,
    "D7h": 14,
    "D8h": 16,
    "D2d": 4,
    "D3d": 6,
    "D4d": 8,
    "D5d": 10,
    "D6d": 12,
    "D7d": 14,
    "D8d": 16,
    "S4": 4,
    "S6": 6,
    "S8": 8,
    "T": 6,
    "Th": 12,
    "Td": 12,
    "O": 12,
    "Oh": 24,
    "Cinfv": 1,
    "Dinfh": 2,
    "I": 30,
    "Ih": 60,
    "Kh": 1,
}

"""
Code for assigning point groups and symmetry numbers.
We still cannot do ``Ih`` point groups, although I doubt this will bother anyone.
"""


def get_point_group(atomic_numbers: list[int], positions: list[list[float]]) -> str:
    assert len(atomic_numbers) > 0, "no atoms!"
    assert len(atomic_numbers) == len(positions), "len(atomic_numbers) doesn't match len(positions)"

    if len(atomic_numbers) == 1:
        return "Kh"

    for Z in atomic_numbers:
        assert Z > 0 and Z <= 92, "only elements H–U are supported"

    msym_elements = [pymsym.Element(name=elements[Z], coordinates=xyz) for Z, xyz in zip(atomic_numbers, positions, strict=True)]

    msym_basis_functions = list()
    for element in msym_elements:
        bfs = [pymsym.RealSphericalHarmonic(element=element, n=2, l=1, m=m, name=f"p{m+1}") for m in (-1, 0, 1)]
        element.basis_functions = bfs
        msym_basis_functions += bfs

    try:
        with pymsym.Context(elements=msym_elements, basis_functions=msym_basis_functions) as ctx:
            group = ctx.find_symmetry()
    except Exception:
        # diff versions throw libmsym.main.Error or libmsym.libmsym.Error, so I'll drop a blanket Exception
        # incredibly, this is the desired behavior of libmsym!
        return "C1"

    # rename things more sensibly
    if group == "D0h":
        group = "Dinfh"
    elif group == "C0v":
        group = "Cinfv"

    return group


def get_symmetry_number(atomic_numbers: list[int], positions: list[list[float]]) -> int:
    point_group = get_point_group(atomic_numbers, positions)
    return SYMMNO_BY_POINT_GROUP[point_group]

