import numpy as np

from graphcmc.utils.constants import BOLTZMANN, J_TO_EV

class BaseForceField:
    def __init__(self, frame, unitcell, ads, vdw_cutoff = 14.0, tail_correction = True, charge = True, device = 'cpu'):
        self.frame = frame
        self.unitcell = unitcell
        self.ads = ads
        self.n_frame_atoms = len(frame)
        self.n_ads_atoms = len(ads)
        self.V = np.linalg.det(frame.cell[:])
        self.vdw_cutoff = vdw_cutoff
        self.tail_correction = tail_correction
        self.charge = charge
        self.device = device

        import json
        from importlib.resources import files
        from graphcmc import data

        with open(files(data).joinpath('lj_params.json'), 'r') as f:
            self.params = json.load(f)

        self.ads_params = np.array([
            [self.params[s]['sigma'], self.params[s]['epsilon']]
            for s in ads.get_chemical_symbols()
        ])

    def _tail_correction(self, elements, start_idx):
        if not self.tail_correction:
            return 0.0

        symbols, counts = np.unique(elements[start_idx:], return_counts = True)
        U_tail = 0
        for s, c in zip(symbols, counts):
            for y, u in zip(symbols, counts):
                epsilon = np.sqrt(self.params[s]['epsilon'] * self.params[y]['epsilon'])
                sigma = (self.params[s]['sigma'] + self.params[y]['sigma']) / 2.
                U_tail += 2 * np.pi / self.V * c * u * 4 / 3 * epsilon * sigma**3 * (((sigma / self.vdw_cutoff)**9) / 3 - (sigma / self.vdw_cutoff)**3)
        return U_tail * BOLTZMANN * J_TO_EV

class ClassicalForceField(BaseForceField):
    def get_energy_difference(self, new_atoms, old_atoms, i_ads, start_idx = 0):        
        old_symbols = np.array(old_atoms.get_chemical_symbols())
        new_symbols = np.array(new_atoms.get_chemical_symbols())

        old_q = np.array(old_atoms.get_initial_charges(), dtype = np.float64)
        new_q = np.array(new_atoms.get_initial_charges(), dtype = np.float64)

        old_tail = self._tail_correction(old_symbols, start_idx)
        new_tail = self._tail_correction(new_symbols, start_idx)
        tail = new_tail - old_tail

        ref_idx = np.r_[
            start_idx:self.n_frame_atoms + i_ads * self.n_ads_atoms,
            self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms:max(len(old_atoms), len(new_atoms))
        ]
        ref_idx = np.array(ref_idx, dtype = np.int64)
        i_ads_idx = np.array([self.n_frame_atoms + self.n_ads_atoms * i_ads + i for i in range(self.n_ads_atoms)], dtype = np.int64)

        if len(new_atoms) > len(old_atoms): # insertion
            vdw = self._lj(new_atoms, new_symbols, i_ads, ref_idx)
            ewald = self._ewald(new_atoms, new_q, ref_idx, i_ads_idx) if self.charge else 0.0
            return vdw + ewald + tail
        
        elif len(new_atoms) < len(old_atoms): # deletion
            vdw = -self._lj(old_atoms, old_symbols, i_ads, ref_idx)
            ewald = -self._ewald(old_atoms, old_q, ref_idx, i_ads_idx) if self.charge else 0.0
            return vdw + ewald + tail
        
        else: # translation or rotation
            vdw_remove = -self._lj(old_atoms, old_symbols, i_ads, ref_idx)
            ewald_remove = -self._ewald(old_atoms, old_q, ref_idx, i_ads_idx) if self.charge else 0.0

            vdw_add = self._lj(new_atoms, new_symbols, i_ads, ref_idx)
            ewald_add = self._ewald(new_atoms, new_q, ref_idx, i_ads_idx) if self.charge else 0.0
            return vdw_remove + ewald_remove + vdw_add + ewald_add

    def get_potential_energy(self, atoms):
        n_atoms = self.n_frame_atoms
        potential_energy = 0.0
        i = 0
        while n_atoms < len(atoms):
            potential_energy += self.get_energy_difference(atoms[:n_atoms + self.n_ads_atoms], atoms[:n_atoms], i)
            n_atoms += self.n_ads_atoms
            i += 1
        return potential_energy

    def _lj(self, atoms, symbols, i_ads, ref_idx):
        vdw = 0.0

        first_ads_atom_idx = self.n_frame_atoms + self.n_ads_atoms * i_ads
        for i in range(self.n_ads_atoms):
            idx = first_ads_atom_idx + i
            dist = atoms.get_distances(idx, ref_idx, mic = True, vector = False)
            dist = np.array(dist, dtype = np.float64)
            mask = (dist < self.vdw_cutoff) * (dist > 0.0)
            mask_idx = np.nonzero(mask)[0]

            if len(mask_idx) == 0:
                continue

            ads_atom_param = self.ads_params[i]
            params = np.array([
                [self.params[symbols[j]]['sigma'], self.params[symbols[j]]['epsilon']] for j in mask_idx
            ], dtype = np.float64)

            sigma = (params[:, 0] + ads_atom_param[0]) / 2
            epsilon = np.sqrt(params[:, 1] * ads_atom_param[1])
            rij = dist[mask_idx]

            lj = (4 * epsilon * ((sigma / rij)**(12) - (sigma / rij)**(6))).sum() * BOLTZMANN * J_TO_EV
            vdw += lj
        return vdw

    def _ewald(self, atoms, charges, ref_idx, i_ads_idx):
        from graphcmc.simulation.ewald import Ewald

        ewald = Ewald(atoms, charges, ref_idx, i_ads_idx)
        return ewald.get_ewaldsum()

class HybridForceField(ClassicalForceField):
    def __init__(self, frame, unitcell, ads, vdw_cutoff = 14.0, tail_correction = True, charge = True, device = 'cpu', mlff = None):
        super().__init__(frame, unitcell, ads, vdw_cutoff = 14.0, tail_correction = True, charge = True, device = 'cpu')

        from fairchem.core import OCPCalculator
        cpu = True if device == 'cpu' else False
        calc = OCPCalculator(model_name = mlff, local_cache = 'pretrained_models', cpu = cpu)
        self.model = calc

    def get_energy_difference(self, new_atoms, old_atoms, i_ads):
        if len(new_atoms) == self.n_frame_atoms or old_atoms is None:
            return 0.0
        
        if len(new_atoms) > len(old_atoms): # insertion
            ml = self._ml(new_atoms, i_ads)
            classical = super().get_energy_difference(new_atoms, old_atoms, i_ads, self.n_frame_atoms)
            return ml + classical

        elif len(new_atoms) < len(old_atoms): # deletion
            ml = -self._ml(old_atoms, i_ads)
            classical = super().get_energy_difference(new_atoms, old_atoms, i_ads, self.n_frame_atoms)
            return ml + classical
        
        else:
            ml_remove = -self._ml(old_atoms, i_ads)
            ml_add = self._ml(new_atoms, i_ads)
            classical = super().get_energy_difference(new_atoms, old_atoms, i_ads, self.n_frame_atoms)
            return ml_remove + ml_add + classical
        
    def _ml(self, atoms, i_ads):
        temp_ads = atoms[self.n_frame_atoms + i_ads * self.n_ads_atoms:
                         self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms].copy()
        temp_ads.set_cell(self.atoms_frame.cell)
        adjusted_pos = (temp_ads.get_scaled_positions() % 1) @ self.atoms_frame.cell
        temp_ads.set_positions(adjusted_pos)
        temp_ads.set_pbc([True, True, True])

        total_atoms = self.atoms_frame.copy() + temp_ads
        total_atoms.calc = self.model

        return total_atoms.get_potential_energy()
