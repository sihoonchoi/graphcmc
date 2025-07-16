import numpy as np

from enum import Enum
from ase import Atoms

from graphcmc.utils.constants import BOLTZMANN
from graphcmc.utils.math import _random_translation, _random_rotation, _random_position

class MoveType(Enum):
    INSERTION = 0
    DELETION = 1
    TRANSLATION = 2
    ROTATION = 3

class MoveExecutor:
    def __init__(self, forcefield, frame, ads):
        self.ff = forcefield
        self.frame = frame.copy()
        self.ads = ads.copy()
        self.cell = self.frame.get_cell()
        self.n_frame_atoms = len(self.frame)
        self.n_ads_atoms = len(self.ads)
        self.V = np.linalg.det(frame.cell[:] / 1e10)
        self.molecule_list = []
        self.energy = 0.0

    def perform_move(self, fugacity, beta):
        move = np.random.choice(list(MoveType))
        if move == MoveType.INSERTION:
            return 0, self._insertion(fugacity, beta)
        elif move == MoveType.DELETION:
            return 1, self._deletion(fugacity, beta)
        elif move == MoveType.TRANSLATION:
            return 2, self._translation(beta)
        elif move == MoveType.ROTATION:
            return 3, self._rotation(beta)
        return 0
    
    def _insertion(self, fugacity, beta):
        mol = self.ads.copy()
        pos = mol.get_positions()
        pos = _random_position(pos, self.cell)
        mol.set_positions(pos)
        old_atoms = self.frame + sum(self.molecule_list, Atoms())
        new_atoms = old_atoms + mol
        dE = self.ff.get_potential_energy(new_atoms, old_atoms, self.energy, len(self.molecule_list)) - self.energy
        prob = np.exp(-beta * dE) * self.V * beta * fugacity / (len(self.molecule_list) + 1)
        if np.random.rand() < min(1.0, prob):
            self.molecule_list.append(mol)
            self.energy += dE
            return 1
        return 0
    
    def _deletion(self, fugacity, beta):
        if not self.molecule_list:
            return 0
        old_atoms = self.frame + sum(self.molecule_list, Atoms())
        idx = np.random.randint(len(self.molecule_list))
        removed = self.molecule_list.pop(idx)
        new_atoms = self.frame + sum(self.molecule_list, Atoms())
        dE = self.ff.get_potential_energy(new_atoms, old_atoms, self.energy, idx) - self.energy
        prob = np.exp(-beta * dE) * (len(self.molecule_list) + 1) / self.V / beta / fugacity
        if np.random.rand() < min(1.0, prob):
            self.energy += dE
            return 1
        self.molecule_list.insert(idx, removed)
        return 0

    def _translation(self, beta):
        if not self.molecule_list:
            return 0
        old_atoms = self.frame + sum(self.molecule_list, Atoms())
        idx = np.random.randint(len(self.molecule_list))
        mol = self.molecule_list[idx].copy()
        pos = mol.get_positions()
        pos = _random_translation(pos, self.cell)
        mol.set_positions(pos)
        new_list = self.molecule_list[:idx] + [mol] + self.molecule_list[idx + 1:]
        new_atoms = self.frame + sum(new_list, Atoms())
        dE = self.ff.get_potential_energy(new_atoms, old_atoms, self.energy, idx) - self.energy
        if np.random.rand() < min(1.0, np.exp(-beta * dE)):
            self.molecule_list[idx] = mol
            self.energy += dE
            return 1
        return 0
    
    def _rotation(self, beta):
        if not self.molecule_list:
            return 0
        old_atoms = self.frame + sum(self.molecule_list, Atoms())
        idx = np.random.randint(len(self.molecule_list))
        mol = self.molecule_list[idx].copy()
        pos = mol.get_positions()
        pos = _random_rotation(pos, self.cell)
        mol.set_positions(pos)
        new_list = self.molecule_list[:idx] + [mol] + self.molecule_list[idx + 1:]
        new_atoms = self.frame + sum(new_list, Atoms())
        dE = self.ff.get_potential_energy(new_atoms, old_atoms, self.energy, idx) - self.energy
        if np.random.rand() < min(1.0, np.exp(-beta * dE)):
            self.molecule_list[idx] = mol
            self.energy += dE
            return 1
        return 0
    
    def _widom(self):
        mol = self.ads.copy()
        pos = mol.get_positions()
        pos = _random_position(pos, self.cell)
        mol.set_positions(pos)
        new_atoms = self.frame + mol
        dE = self.ff.get_potential_energy(new_atoms, self.frame, 0, 0)
        return dE
    
    def count_adsorbates(self):
        return len(self.molecule_list)
