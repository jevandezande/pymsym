import pymsym
import cctk

from pymsym.pymsym import Element

def get_symmetry(molecule: cctk.Molecule) -> str:
    msym_elements = list()
    for i in range(1, molecule.num_atoms() + 1):
        msym_elements.append(pymsym.Element(
            name=cctk.helper_functions.get_symbol(molecule.atomic_numbers[i]),
            coordinates=molecule.geometry[i].tolist(),
        ))

    msym_basis_functions = list()
    for element in msym_elements:
        bfs = [pymsym.RealSphericalHarmonic(element=element, n=2, l=1, m=m, name=f"p{m+1}") for m in (-1, 0, 1)]
        element.basis_functions = bfs
        msym_basis_functions += bfs

#    try:
    with pymsym.Context(elements=msym_elements, basis_functions=msym_basis_functions) as ctx:
        return ctx.find_symmetry()
#    except Exception as e:
#        print(e)
#        # diff versions throw libmsym.main.Error or libmsym.libmsym.Error, so I'll drop a blanket Exception
#        # incredibly, this is the desired behavior of libmsym!
#        return "C1"

print(get_symmetry(cctk.Molecule.new_from_name("water")))
