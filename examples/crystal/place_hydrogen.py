from ase.io import read, write
from ase.geometry import get_distances
from ase.geometry import find_mic
from ase import Atoms
import numpy as np
from ase.neighborlist import NeighborList
from collections import defaultdict
import math
import spglib
import warnings
warnings.filterwarnings('ignore')


### Filenames
input_filename = '0_POSCAR'                     
input_format = 'vasp'  
output_filename = '1_POSCAR_H'     
output_format = 'vasp'

# Forces to maintain symmetry
force_symmetry = True

put_hydrogens = 'all'            # Where to put hydrogen atoms - either list or 'all' (all C, N, O atoms without hydrogen atoms)
N_charged = True                 # Charged N group (adds extra hydrogen on N atom) - either True/False or list of N atoms

# Forcing double/triple/aromatic bonds
single = [[9,7],[8,6]]          # Force some bonds to be single - e.g. force single bonds between 1 and 2; 3 and 4 will be [[1,2],[3,4]]
double = []
triple = []
aromatic = []
saturate_all = False             # Make all bonds single bonds

cutoff = 2                      # Cutoff for neighboring atoms

# Library of bond lenghts
c_c_bonds = {
        'c_c_single': 1.5,
        'c_c_aromatic': 1.4,
        'c_c_double': 1.2,
        'c_c_triple': 1.1
        }
c_n_bonds = {
        'c_n_single': 1.45,
        'c_n_double': 1.25,
        'c_n_triple': 1.1}
c_o_bonds = {
        'c_o_single': 1.4,
        'c_o_double': 1.1}

n_o_bonds = {
        'n_o_single': 1.4,
        'n_o_double': 1.2}


### Converting to ASE indices (ASE labeling starts from 0)
def process_list(input_list):
    if isinstance(input_list, list) and len(input_list) > 0:
        if isinstance(input_list[0], list):
            input_list = [np.array(pair) - 1 for pair in input_list]
            input_list = [pair.tolist() for pair in input_list]
        else:
            input_list = np.array(input_list) - 1
            input_list = input_list.tolist()
    return input_list

N_charged = process_list(N_charged)
single = process_list(single)
double = process_list(double)
triple = process_list(triple)
aromatic = process_list(aromatic)
put_hydrogens = process_list(put_hydrogens) 


### Vector rotation function
def rotate_point(MB, B, axis, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    u_x, u_y, u_z = axis
    rotation_matrix = np.array([
        [cos_theta + u_x**2 * (1 - cos_theta), u_x * u_y * (1 - cos_theta) - u_z * sin_theta, u_x * u_z * (1 - cos_theta) + u_y * sin_theta],
        [u_y * u_x * (1 - cos_theta) + u_z * sin_theta, cos_theta + u_y**2 * (1 - cos_theta), u_y * u_z * (1 - cos_theta) - u_x * sin_theta],
        [u_z * u_x * (1 - cos_theta) - u_y * sin_theta, u_z * u_y * (1 - cos_theta) + u_x * sin_theta, cos_theta + u_z**2 * (1 - cos_theta)]
    ])
    return rotation_matrix.dot(MB) + B

### Normalizing vector function
def normalize(vector):
    magnitude = np.linalg.norm(vector)
    return vector / magnitude


### Function to get neighbors within defined cutoff
def get_neighbors(atom_index, cutoff):
    target_position = structure[atom_index].position
    distances = get_distances(target_position, structure.positions, cell=structure.cell, pbc=True)[1]
    vectors = get_distances(target_position, structure.positions, cell=structure.cell, pbc=True)[0][0]
    carbon = [i for i, atom in enumerate(structure) if atom.symbol == 'C']
    nitrogen = [i for i, atom in enumerate(structure) if atom.symbol == 'N']
    oxygen = [i for i, atom in enumerate(structure) if atom.symbol == 'O']
    neighbors = np.where((distances[0] < cutoff) & (distances[0] != 0))[0]
    close_idx = np.intersect1d(neighbors, np.array(carbon + nitrogen + oxygen))        
    return close_idx,vectors

# Calculate number of hydrogen atoms
def get_valence(atom_index,cutoff):
    target_position = structure[atom_index].position
    distances = get_distances(target_position, structure.positions, cell=structure.cell, pbc=True)[1]
    carbon = [i for i, atom in enumerate(structure) if atom.symbol == 'C']
    nitrogen = [i for i, atom in enumerate(structure) if atom.symbol == 'N']
    oxygen = [i for i, atom in enumerate(structure) if atom.symbol == 'O']
    neighbors = np.where((distances[0] < cutoff) & (distances[0] != 0))[0]
    close_idx = np.intersect1d(neighbors, np.array(carbon + nitrogen + oxygen))

    dist_n = [distances[0][at] for at in close_idx]
    symb_n = [structure[at].symbol for at in close_idx]

    for pair in single:
        if atom_index in pair:
            for number in pair:
                if number in close_idx:
                    idx = np.where(close_idx == number)[0][0]
                    dist_n[idx] += c_c_bonds['c_c_single']
                    break
                
    for pair in double:
        if atom_index in pair:
            for number in pair:
                if number in close_idx:
                    idx = np.where(close_idx == number)[0][0]
                    dist_n[idx] = c_c_bonds['c_c_double']
                    break
                
    for pair in triple:
        if atom_index in pair:
            for number in pair:
                if number in close_idx:
                    idx = np.where(close_idx == number)[0][0]
                    dist_n[idx] = c_c_bonds['c_c_triple']
                    break
                
    for pair in aromatic:
        if atom_index in pair:
            for number in pair:
                if number in close_idx:
                    idx = np.where(close_idx == number)[0][0]
                    dist_n[idx] = c_c_bonds['c_c_aromatic']
                    break     

    data_n = defaultdict(list, {'C': [], 'N': [], 'O': []})
    for symb, dist in zip(symb_n, dist_n):
        data_n[symb].append(dist)
    data_n = dict(data_n)


    def find_closest(bond_lenght, bond_types, saturate_all = saturate_all):
        if saturate_all:
            bond_lenght += 1.5
        type_bond = min(bond_types, key=lambda x: abs(bond_lenght - bond_types[x]))
        return type_bond


    if structure[atom_index].symbol == "C":
        types_c_c_bond = [find_closest(dist, c_c_bonds) for dist in data_n['C']]
        types_c_n_bond = [find_closest(dist, c_n_bonds) for dist in data_n['N']]
        types_c_o_bond = [find_closest(dist, c_o_bonds) for dist in data_n['O']]

        all_bonds = types_c_c_bond + types_c_n_bond + types_c_o_bond
        bond_sum = sum(1 if 'single' in bond else 2 if 'double' in bond else 3 if 'triple' in bond else 1.5 if 'aromatic' in bond else 0 for bond in all_bonds)

        default_valence = 4
        
        n_hydr =  math.ceil(default_valence - bond_sum)


    if structure[atom_index].symbol == "N":
        types_c_n_bond = [find_closest(dist, c_n_bonds) for dist in data_n['C']]
        types_n_o_bond = [find_closest(dist, n_o_bonds) for dist in data_n['O']]

        all_bonds = types_c_n_bond + types_n_o_bond
        bond_sum = sum(1 if 'single' in bond else 2 if 'double' in bond else 3 if 'triple' in bond else 1.5 if 'aromatic' in bond else 0 for bond in all_bonds)
            
        default_valence = 3
        
        if isinstance(N_charged, bool):
            if N_charged:
                default_valence = 4
        else:
            if atom_index in N_charged:
                default_valence = 4
        
        n_hydr = default_valence - bond_sum


    if structure[atom_index].symbol == "O":
        types_c_o_bond = [find_closest(dist, c_o_bonds) for dist in data_n['C']]
        types_n_o_bond = [find_closest(dist, n_o_bonds) for dist in data_n['N']]

        all_bonds = types_c_o_bond + types_n_o_bond
        bond_sum = sum(1 if 'single' in bond else 2 if 'double' in bond else 3 if 'triple' in bond else 1.5 if 'aromatic' in bond else 0 for bond in all_bonds)
        default_valence = 2
        n_hydr = default_valence - bond_sum

    n_neigh = len(dist_n)

    return [n_hydr,n_neigh]

    

### Hydrogen placements functions
def one_hydrogen_two_neighbor(atom_index,cutoff):
    nei_cn,vectors = get_neighbors(atom_index, cutoff)
    B = structure[atom_index].position
    A = normalize(vectors[nei_cn[0]]) + B
    C = normalize(vectors[nei_cn[1]]) + B
    M = (A + C) / 2
    MB = normalize(B - M)
    hydrogen1 = B + MB

    hydrogen = [hydrogen1]
    return hydrogen

def one_hydrogen_three_neighbor(atom_index,cutoff):
    nei_cn,vectors = get_neighbors(atom_index, cutoff)
    B = structure[atom_index].position
    A = normalize(vectors[nei_cn[0]]) + B
    C = normalize(vectors[nei_cn[1]]) + B
    D = normalize(vectors[nei_cn[2]]) + B
    
    M = (A + C + D) / 3
    MB = normalize(B - M)
    hydrogen1 = B + MB
    
    hydrogen = [hydrogen1]
    return hydrogen

    
def two_hydrogen_two_neighbor(atom_index,cutoff):
    nei_cn,vectors = get_neighbors(atom_index, cutoff)
    B = structure[atom_index].position
    A = normalize(vectors[nei_cn[0]]) + B
    C = normalize(vectors[nei_cn[1]]) + B
    
    AB = B - A
    D = B + AB
    M = (A + C) / 2
    MB = normalize(B - M)
    axis = normalize(A - C)
    
    hydrogen1 = rotate_point(MB, B, axis, 0.954)

    hydrogen2 = rotate_point(MB, B, axis, -0.954)

    hydrogen = [hydrogen1,hydrogen2]
    return hydrogen


def three_hydrogen_one_neighbor(atom_index,cutoff):
    nei_cn,vectors = get_neighbors(atom_index, cutoff)
    B = structure[atom_index].position
    A = normalize(vectors[nei_cn[0]]) + B

    AB = normalize(B - A)
    arbitrary_vector = np.array([1,0,0])
    axis = normalize(np.cross(AB, arbitrary_vector))
    H1 = rotate_point(AB, B, axis, 1.23)
    BH = normalize(H1 - B)
    hydrogen1 = B + BH
    hydrogen2 = rotate_point(BH, B, AB, 2*np.pi/3)
    hydrogen3 = rotate_point(BH, B, AB, -2*np.pi/3)

    hydrogen = [hydrogen1,hydrogen2,hydrogen3]
    return hydrogen
    
def two_hydrogen_one_neighbor(atom_index,cutoff):
    nei_cn,vectors = get_neighbors(atom_index, cutoff)
    B = structure[atom_index].position
    A = normalize(vectors[nei_cn[0]]) + B

    AB = normalize(B - A)
    arbitrary_vector = np.array([1,0,0])
    axis = normalize(np.cross(AB, arbitrary_vector))
    H1 = rotate_point(AB, B, axis, np.pi/3)
    BH = normalize(H1 - B)
    hydrogen1 = B + BH
    hydrogen2 = rotate_point(BH, B, AB, np.pi)
    
    hydrogen = [hydrogen1,hydrogen2]
    return hydrogen

def one_hydrogen_one_neighbor(atom_index,cutoff):
    nei_cn,vectors = get_neighbors(atom_index, cutoff)
    A = structure[nei_cn[0]].position
    B = structure[atom_index].position
    AB = normalize(B - A)
    hydrogen1 = AB + B
    
    hydrogen = [hydrogen1]
    return hydrogen

def four_hydrogen(atom_index):
    target_position = structure[atom_index].position
    hydrogen1 = target_position + np.array([0.00000,    0.00000,    1.08900])
    hydrogen2 = target_position + np.array([1.02672,    0.00000,   -0.36300])
    hydrogen3 = target_position + np.array([-0.51336,   -0.88916,   -0.36300])
    hydrogen4 = target_position + np.array([-0.51336,    0.88916,   -0.36300])

    hydrogen = [hydrogen1,hydrogen2,hydrogen3,hydrogen4]
    return hydrogen

def three_hydrogen(atom_index):
    atom_pos = structure[atom_index].position
    hydrogen1 = atom_pos + np.array([1.0, -0.5, -0.5]) 
    hydrogen2 = atom_pos + np.array([0.0, (3**0.5)/2, -(3**0.5)/2])
    hydrogen3 = atom_pos + np.array([0.0, 0.0, 0.0]) 
    hydrogen = [hydrogen1,hydrogen2,hydrogen3]
    return hydrogen

def water_hydrogen(atom_index):
    O = structure[atom_index].position
    hydrogen1 = O + np.array([0,0,1])
    arbitrary_vector = np.array([1,0,0])
    axis = normalize(np.cross(O-hydrogen1, arbitrary_vector))
    OH = normalize(O - hydrogen1)
    hydrogen2 = rotate_point(OH, target_position, axis, 0.58)
    
    hydrogen = [hydrogen1,hydrogen2]
    return hydrogen

### Determining the type of hydrogen placement
def hydrogen_positions(n_hydr,n_neigh,atom_index,cutoff):
    if n_hydr < 1:
        hydrogens = []
    if n_neigh == 1 and n_hydr == 1:
        hydrogens = one_hydrogen_one_neighbor(atom_index,cutoff)
    if n_neigh == 2 and n_hydr == 1:
        hydrogens = one_hydrogen_two_neighbor(atom_index,cutoff)
    if n_neigh == 3 and n_hydr == 1:
        hydrogens = one_hydrogen_three_neighbor(atom_index,cutoff)
    if n_neigh == 0 and n_hydr == 2 and structure[atom_index] == "O":
        hydrogens = water_hydrogen(atom_index)
    if n_neigh == 1 and n_hydr == 2:
        hydrogens = two_hydrogen_one_neighbor(atom_index,cutoff)
    if n_neigh == 2 and n_hydr == 2:
        hydrogens = two_hydrogen_two_neighbor(atom_index,cutoff)
    if n_neigh == 0 and n_hydr == 3:
        hydrogens = three_hydrogen(atom_index)
    if n_neigh == 1 and n_hydr == 3:
        hydrogens = three_hydrogen_one_neighbor(atom_index,cutoff)
    if n_neigh == 0 and n_hydr == 4:
        hydrogens = four_hydrogen(atom_index)

    return hydrogens


### Process structure
structure = read(input_filename, format = input_format)

# Check if it is a crystal. If not, add an arbitrary unit cell
crystal = False
molecule = False

if np.linalg.det(structure.get_cell()) != 0:
    crystal = True
else:
    molecule = True

if molecule:
    positions = structure.get_positions()
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)
    cell_size = (max_coords - min_coords) + 5.0
    centered_positions = positions - (min_coords + max_coords) / 2.0
    shifted_positions = centered_positions + (cell_size / 2.0)
    structure = Atoms(
        symbols=structure.get_chemical_symbols(),
        positions=shifted_positions,
        cell=cell_size,
        pbc=True)



if isinstance(put_hydrogens, str) and put_hydrogens == 'all':
    hydrogen_atoms = [i for i, atom in enumerate(structure) if atom.symbol == 'H']
    #put_hydrogens = [i for i, atom in enumerate(structure) if atom.symbol in ['C', 'N', 'O']] # All atoms

    # Do not consider atoms with already placed hydrogen atoms
    put_hydrogens = []
    for i, atom in enumerate(structure):
        if atom.symbol in ['C', 'N', 'O']:
            close_to_hydrogen = False
            for h in hydrogen_atoms:
                displacement, distance = find_mic(structure[h].position - structure[i].position, cell=structure.cell, pbc=structure.pbc)
                if distance < 1.5:
                    close_to_hydrogen = True
                    break
            if not close_to_hydrogen:
                put_hydrogens.append(i)



### Putting the hydrogens

# Place hydrogens normally
if force_symmetry == False:
    hydrogen_place = []
    for atom_index in put_hydrogens:
        n_hydr, n_neigh = get_valence(atom_index,cutoff)
        hydr_pos = hydrogen_positions(n_hydr,n_neigh,atom_index,cutoff)
        if len(hydr_pos) > 0:
            for hydr in hydr_pos:
                hydrogen_place.append(hydr)
    print(f"Placing {len(hydrogen_place)} hydrogen atoms...")

# Place hydrogen with forced maintained symmetry
if force_symmetry == True:
    cell = (structure.cell, structure.get_scaled_positions(), structure.get_atomic_numbers())
    symmetry_data = spglib.get_symmetry_dataset(cell)
    equiv_atoms = symmetry_data['equivalent_atoms']
    num_atoms = len(equiv_atoms)

    equiv_atom_list = []
    for i in range(num_atoms):
        while len(equiv_atom_list) <= equiv_atoms[i]:
            equiv_atom_list.append([])
        equiv_atom_list[equiv_atoms[i]].append(i)
    equiv_atom_list = [sublist for sublist in equiv_atom_list if sublist]
    unique_indices = [sublist[0] for sublist in equiv_atom_list if sublist]
    unique_put_hydrogens = [index for index in put_hydrogens if index in unique_indices]

    unique_hydrogen_place = []
    for atom_index in unique_put_hydrogens:
        n_hydr, n_neigh = get_valence(atom_index,cutoff)
        hydr_pos = hydrogen_positions(n_hydr,n_neigh,atom_index,cutoff)
        if len(hydr_pos) > 0:
            for hydr in hydr_pos:
                unique_hydrogen_place.append(hydr)

    cell_matrix = structure.get_cell()
    inverse_cell_matrix = np.linalg.inv(cell_matrix)
    unique_hydrogen_place = [np.dot(coord, inverse_cell_matrix) for coord in unique_hydrogen_place]
    
    symmetry_operations = symmetry_data['rotations']
    translations = symmetry_data['translations']
    symmetric = []
    for position in unique_hydrogen_place:
        for rotation, translation in zip(symmetry_operations, translations):
            rotated_position = np.dot(rotation, position)
            translated_position = rotated_position + translation
            symmetric.append(translated_position)
    hydrogen_place = np.unique(np.round(symmetric, decimals=6), axis=0)
    hydrogen_place = [np.dot(hydrogen_coord, cell_matrix) for hydrogen_coord in hydrogen_place]

    if crystal:
        print(f"International Symbol: {symmetry_data['international']}")
        print(f"International Number: {symmetry_data['number']}")
        print()

    if molecule:
        hydrogen_place = [pos for pos in hydrogen_place if np.all(pos >= 0)]

    print(f"Placing {len(hydrogen_place)} hydrogen atoms...")


hydr_atoms = Atoms('H' * len(hydrogen_place), positions=hydrogen_place)
hydrogen_structure = structure + hydr_atoms

if molecule:
    positions = hydrogen_structure.get_positions()
    symbols = hydrogen_structure.get_chemical_symbols()
    hydrogen_structure = Atoms(symbols=symbols, positions=positions)



write(output_filename, hydrogen_structure, format=output_format)
print(f"Hydrogens succesfully placed and written to {output_filename}.")

