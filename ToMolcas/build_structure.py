#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

__author__ = 'Weiguo Jing'
__version__ = "1.0.0"

from method.structure import Structure
from method.atom import Atom
from method.read import poscar
import numpy as np


def structure_species(structure):
    from method.defect import specieset
    atoms_type = specieset(structure)
    species = {}
    for name in atoms_type:
        count = 0
        for atom in structure:
            if atom.type == name:
                count += 1
        species[name] = count
    return sorted(species.items(), key=lambda item: item[1])


def build_cluster(structure, doped_atom=None, cluster_r=5, core_r=None, tolerance=0.2):
    from coordination_shells import coordination_shells
    """

    :param structure: Pylada.crystal.Structure
    :param doped_atom: Pylada.crystal.Atom
    :param cluster_r: float
    :param core_r: float
    :return: cluster, Pylada.crystal.Structure
    """

    if doped_atom is None:
        species = structure_species(structure)
        doped_name = species[0][0]
        num = species[0][1]
        if num != 1:
            print('WARNING: The number of doped atom is not exclusive, please give doped atom name and position.\n')
        for atom in structure:
            if atom.type == doped_name:
                doped_atom = Atom(atom.pos, atom.type)
                break
    cluster = Structure(structure.cell, scale=structure.scale)
    doped_center = doped_atom.pos
    if core_r is not None:
        for atom in structure:
            position = atom.pos
            d = np.linalg.norm(doped_center - position)
            if d <= core_r:
                cluster.append(Atom(atom.pos - doped_center, atom.type, pseudo=0))
            elif d <= cluster_r:
                cluster.append(Atom(atom.pos - doped_center, atom.type, pseudo=-1))
    else:
        # try to find the nearest neighbors of doped atom
        neighbors = coordination_shells(structure, nshells=4, center=doped_center, tolerance=tolerance)
        nearest_neighbors = []
        flag = 0
        for item in neighbors[0]:
            atom = item[0]
            nearest_neighbors.append(atom)
            atom.pseudo = 0
        if len(sorted({a.type for a in nearest_neighbors})) > 1:
            print('WARNING: The species of nearest neighbor atoms are more than one.\n')
        atom = neighbors[0][0][0]
        nearest_name = atom.type
        for i in range(1, 4):
            if flag != 0:
                break
            temp = [item[0] for item in neighbors[i]]
            if len(sorted({a.type for a in temp})) > 1:
                print('WARNING: The species of next nearest neighbor atoms:\n')
                print(sorted({a.type for a in temp}))
                print('\nwhich means the tolerance parameter might be too big to distinguish different shell of '
                      'neighbors\n')
                # when this situation happened, we discard this shell.
                break
            for item in neighbors[i]:
                atom = item[0]
                if nearest_name == atom.type:
                    nearest_neighbors.append(atom)
                    atom.pseudo = 0
                else:
                    flag = 1
        for atom in structure:
            position = atom.pos
            d = np.linalg.norm(doped_center - position)
            if d <= cluster_r:
                if hasattr(atom, 'pseudo'):
                    cluster.append(Atom(atom.pos - doped_center, atom.type, pseudo=0))
                elif atom.type == doped_atom.type and d < 0.01:
                    cluster.append(Atom(atom.pos - doped_center, atom.type, pseudo=0))
                else:
                    cluster.append(Atom(atom.pos - doped_center, atom.type, pseudo=-1))
    return cluster


def remove_doped_atoms(structure, remove_dict):
    temp = []
    for atom in structure:
        if atom.type in remove_dict.keys():
            value = remove_dict[atom.type]
            if value != '':
                temp.append(Atom(atom.pos, value))
        else:
            temp.append(Atom(atom.pos, atom.type))
    structure.clear()
    structure.extend(temp)
    return structure


def extend_structure(structure, shell_r=30):
    cell = structure.cell.T
    volume = structure.volume
    hz = volume/np.linalg.norm(np.cross(cell[0], cell[1]))
    hx = volume/np.linalg.norm(np.cross(cell[1], cell[2]))
    hy = volume/np.linalg.norm(np.cross(cell[2], cell[0]))
    if min(hx, hy, hz) >= shell_r:
        return structure
    nz = int(np.ceil(shell_r/hz - 0.5))
    nx = int(np.ceil(shell_r/hx - 0.5))
    ny = int(np.ceil(shell_r/hy - 0.5))
    temp = []
    for i in range(-nx, nx+1):
        for j in range(-ny, ny+1):
            for k in range(-nz, nz+1):
                for atom in structure:
                    position = atom.pos + i*cell[0] + j*cell[1] + k*cell[2]
                    temp.append(Atom(position, atom.type))
    supercell = Structure(structure.cell, scale=structure.scale)
    supercell.extend(temp)
    return supercell


def build_shell(structure, pure_substrate=False, doped_atom=None, remove_dict=None, cluster_r=5, shell_r=30):
    """

    :param structure: Pylada.crystal.Structure
    :param pure_substrate: Bool type
    :param doped_atom: Pylada.crystal.Atom
    :param remove_dict: dict
    :param cluster_r: float
    :param shell_r: float
    :return:
    """
    if pure_substrate is True:
        pass
    else:
        if doped_atom is None:
            species = structure_species(structure)
            doped_name = species[0][0]
            num = species[0][1]
            if num != 1:
                print('WARNING: The number of doped atom is not exclusive, please give doped atom name and position.\n')
            for atom in structure:
                if atom.type == doped_name:
                    doped_atom = Atom(atom.pos, atom.type)
                    break
    if pure_substrate is False:
        if remove_dict is None:
            print('ERROR: remove_list can not be None when substrate includes doped atoms.\n')
            exit(1)
        structure = remove_doped_atoms(structure, remove_dict)
    supercell = extend_structure(structure, shell_r)
    shell = Structure(structure.cell, scale=structure.scale)
    doped_center = doped_atom.pos
    temp = []
    for atom in supercell:
        position = atom.pos
        d = np.linalg.norm(position - doped_center)
        if cluster_r < d <= shell_r:
            temp.append(Atom(position - doped_center, atom.type))
    shell.extend(temp)
    return shell


def build_mosaic_structure(cluster, shell):
    for atom in shell:
        cluster.append(Atom(atom.pos, atom.type, pseudo=-1))
    return cluster


def write_seward_input(structure, core_pseudo, mosaic_pseudo, charge, mosaic_r=10, file_name='_sew.in', title=None):
    core = {}
    mosaic = {}
    for key in core_pseudo:
        core[key] = []
    for key in mosaic_pseudo:
        mosaic[key] = []
    ion = []
    for atom in structure:
        if atom.pseudo == 0:
            try:
                core[atom.type].append(atom)
            except KeyError:
                print('ERROR: Core pseudo potential do not include the information about {0}\n'.format(atom.type))
                exit(1)
        elif np.linalg.norm(atom.pos) <= mosaic_r:
            try:
                mosaic[atom.type].append(atom)
            except KeyError:
                print('ERROR: mosaic pseudo potential do not include the information about {0}\n'.format(atom.type))
                exit(1)
        else:
            ion.append(atom)
    string = " &SEWARD &END\nTitle\n{0}\n".format(title)
    for key in core.keys():
        temp = "Basis set\n{0}\n".format(core_pseudo[key])
        i = 1
        for atom in core[key]:
            temp = temp + "{name}{num}{pos0}{pos1}{pos2}  Angstrom \n".format(name=atom.type, num='{:<6d}'.format(i),
                                                                              pos0='{:10.4f}'.format(atom.pos[0]),
                                                                              pos1='{:10.4f}'.format(atom.pos[1]),
                                                                              pos2='{:10.4f}'.format(atom.pos[2]))
            i += 1
        temp = temp + "End of basis\n********************************************\n"
        string = string + temp
    abc_code = 65
    for key in mosaic.keys():
        temp = "Basis set\n{0}\n".format(mosaic_pseudo[key])
        i = 1
        for atom in mosaic[key]:
            temp = temp + "{char}{num}{pos0}{pos1}{pos2}  Angstrom \n".format(char=chr(abc_code),
                                                                              num='{:<6d}'.format(i),
                                                                              pos0='{:10.4f}'.format(atom.pos[0]),
                                                                              pos1='{:10.4f}'.format(atom.pos[1]),
                                                                              pos2='{:10.4f}'.format(atom.pos[2]))
            i += 1
        temp = temp + "End of basis\n********************************************\n"
        string = string + temp
        if i >= 1000:
            print('WARNING: There are too many {0} atoms in mosaic shell, which cause the label of atoms at first '
                  'column has more than 4 characters.\nIt will cause Molcas reduce the label to 4 characters and '
                  'cause *reduplicate* problem. e.g. "A1000" will be reduced to "A100" in Molcas.\n'.format(key))
        abc_code = (abc_code - 65 + 1) % 26 + 65
        if (abc_code - 65 + 1) // 26 > 1:
            print('WARNING: Too many species in mosaic shell. The label of atoms at first column will be '
                  '*reduplicate*.\n')
    string = string + "Xfield\n{0}  Angstrom\n".format(len(ion))
    for atom in ion:
        q = 0.0
        try:
            q = charge[atom.type]
        except KeyError:
            print('ERROR: The charge information about {0} is needed\n'.format(atom.type))
            exit(1)
        string = string + "{pos0}{pos1}{pos2}{charge}  0.0  0.0  0.0\n".format(pos0='{:10.4f}'.format(atom.pos[0]),
                                                                               pos1='{:10.4f}'.format(atom.pos[1]),
                                                                               pos2='{:10.4f}'.format(atom.pos[2]),
                                                                               charge='{:10.4f}'.format(q))
    string = string + 'AMFI\nSDIPOLE\nEnd of input \n'
    with open(file_name, 'w') as fp:
        fp.write(string)


if __name__ == "__main__":
    doped_crystal = 'CONTCAR'
    substrate = 'POSCAR'
    core_pseudo = {'Bi': 'Bi.ECP.Barandiaran.13s12p8d5f.3s4p3d2f.15e-CG-AIMP.',
                   'F': 'F.ECP.Barandiaran.5s6p1d.2s4p1d.7e-CG-AIMP.'}
    mosaic_pseudo = {'Ca': 'Ca.ECP.Pascual.0s.0s.0e-AIMP-CaF2.',
                     'F': 'F.ECP.Pascual.0s.0s.0e-AIMP-CaF2. '}
    charge = {'Ca': 2, 'F': -1}
    doped_structure = poscar(doped_crystal)
    substrate_structure = poscar(substrate)
    cluster = build_cluster(doped_structure)
    discard = build_cluster(substrate_structure)
    if len(cluster) != len(discard):
        print('WARNING: \n'
              'The number of atoms in cluster is different with the discard cluster of substrate.\n'
              'The radius of cluster may be too small.\n')
    shell = build_shell(substrate_structure, remove_dict={'Bi': 'Ca'})
    structure = build_mosaic_structure(cluster, shell)
    write_seward_input(structure, core_pseudo, mosaic_pseudo, charge, title="CaF2_Bi")
