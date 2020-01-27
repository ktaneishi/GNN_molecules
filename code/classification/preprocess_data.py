#!/usr/bin/env python
import numpy as np
from collections import defaultdict
import pickle
import os

from rdkit import Chem

def create_atoms(mol):
    '''Create a list of atom (e.g., hydrogen and oxygen) IDs considering the aromaticity.'''
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol):
    '''Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs.'''
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    '''Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm.'''

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            '''Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints).'''
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            '''Also update each edge ID considering two nodes
            on its both sides.'''
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)

def main(radius):
    # DATASET = yourdata
    DATASET = 'HIV'

    with open('../../dataset/classification/%s/original/data.txt' % DATASET, 'r') as f:
        data_list = f.read().strip().split('\n')

    '''Exclude data contains '.' in the SMILES format.'''
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    
    Smiles, molecules, adjacencies, properties = '', [], [], []

    for index, data in enumerate(data_list, 1):
        smiles, property = data.strip().split()
        Smiles += smiles + '\n'

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        molecules.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        properties.append(np.array([float(property)]))
        
        print('\rradius: %5d, %5d/%5d' % (radius, index, len(data_list)), end='')
    print('')

    dir_input = '../../dataset/classification/%s/input/radius%d/' % (DATASET, radius)
    os.makedirs(dir_input, exist_ok=True)

    with open(dir_input + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    np.save(os.path.join(dir_input, 'molecules'), molecules)
    np.save(os.path.join(dir_input, 'adjacencies'), adjacencies)
    np.save(os.path.join(dir_input, 'properties'), properties)
    dump_dictionary(fingerprint_dict, os.path.join(dir_input, 'fingerprint_dict.pkl'))

if __name__ == '__main__':
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))

    # radius=0...3  # w/o fingerprints (i.e., atoms).
    for radius in range(0, 4):
        main(radius)

    print('The preprocess of dataset has finished!')
