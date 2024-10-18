from ase.io import read, write
from ase import Atoms

### Filenames
input_filename = 'POSCAR'
input_format = 'vasp'
output_filename = 'POSCAR_no_H'
output_format = 'vasp'

atoms = read(input_filename, format=input_format)

atoms_no_hydrogen = atoms[[atom.index for atom in atoms if atom.symbol != 'H']]

write(output_filename, atoms_no_hydrogen, format=output_format)

print(f'Hydrogen atoms removed and structure saved to {output_filename}')
