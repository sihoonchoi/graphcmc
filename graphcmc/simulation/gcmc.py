import numpy as np
import os

class GCMC:
    def __init__(self, mofname, adsname, T, P, ff,
                 vdw_cutoff = 14.0,
                 tail_correction = True,
                 charge = True,
                 device = 'cpu',
                 print_every = 1000,
                 minimum_inner_steps = 10,
                 print_stats = True,
                 result_dir = 'results'
                 ):
        
        from graphcmc.simulation import ClassicalForceField, HybridForceField, MoveExecutor
        from graphcmc.utils.math import compute_supercell_size
        from graphcmc.utils.io import load_atoms_and_charges

        self.mofname = mofname
        self.adsname = adsname
        self.unitcell, self.ads = load_atoms_and_charges(self.mofname, self.adsname, charge)
        self.cell = self.unitcell.get_cell()
        self.vdw_cutoff = vdw_cutoff
        self.x, self.y, self.z = compute_supercell_size(self.cell, self.vdw_cutoff)
        self.supercell = self.unitcell * (self.x, self.y, self.z)

        self.ff = ff
        if self.ff == 'UFF':
            forcefield = ClassicalForceField(self.supercell, self.unitcell, self.ads, vdw_cutoff, tail_correction, charge, device)
        else:
            forcefield = HybridForceField(self.supercell, self.unitcell, self.ads, vdw_cutoff, tail_correction, charge, device, self.ff)

        self.executor = MoveExecutor(forcefield, self.supercell, self.ads)
        self.n_unitcell_atoms = len(self.unitcell)
        self.n_ads_atoms = len(self.ads)

        self._T = None
        self._P = None
        self.T = T
        self.P = P

        self.print_every = print_every
        self.minimum_inner_steps = minimum_inner_steps
        self.print_stats = print_stats
        
        self.result_dir = result_dir

    @property
    def T(self):
        return self._T
        
    @T.setter
    def T(self, new_T):
        self._T = new_T
        self._update()

    @property
    def P(self):
        return self._P
    
    @P.setter
    def P(self, new_P):
        self._P = new_P
        self._update()
        
    def _update(self):
        from graphcmc.simulation.eos import PREOS
        from graphcmc.utils.constants import BOLTZMANN, J_TO_EV

        if self._T is not None and self._P is not None:
            self.fugacity = PREOS.from_name(self.adsname).calculate_fugacity(self._T, self._P)
            self.beta = 1 / BOLTZMANN / self._T / J_TO_EV
            self.job_id =  f'{self.mofname}_{self.adsname}_{self._T}K_{int(self._P):07d}Pa_{self.ff}'

    def __str__(self):
        return '\n'.join([
            "################################################\n",
            "GCMC Calculation",
            "",
            f"\tFramework: {self.mofname}",
            f"\tAdsorbate: {self.adsname}",
            f"\tForcefield: {self.ff}",
            f"\tVDW Cutoff: {self.vdw_cutoff} \u212B",
            f"\tTemperature: {self.T} K",
            f"\tPressure: {self.P} Pa",
            "",
            "################################################",
            "\n"
        ])

    def __repr__(self):
        return self.__str__()

    def mc(self, N, initialize = False, continue_sim = False):
        from ase import Atoms
        from graphcmc.utils.io import setup_result_directory, save_checkpoint, load_checkpoint

        result_dir = setup_result_directory(self.result_dir, self.job_id, initialize, continue_sim)
        self.output = self.__str__()

        stats, molecule_list = load_checkpoint(result_dir, initialize, continue_sim)
        self.executor.molecule_list = molecule_list

        if continue_sim:
            self.executor.energy = stats['adsorption_energy'][-1]
        elif initialize:
            self.executor.energy = 0.0
        else:
            self.executor.ff.get_potential_energy(self.supercell + sum(self.executor.molecule_list, Atoms()))

        already_run = len(stats['uptake'])
        for iteration in range(already_run, N):
            for _ in range(max(self.minimum_inner_steps, self.executor.count_adsorbates())):
                i, accepted = self.executor.perform_move(self.fugacity, self.beta)
                stats['attempted'][i] += 1
                stats['accepted'][i] += accepted

            stats['uptake'].append(self.executor.count_adsorbates())
            stats['adsorption_energy'].append(self.executor.energy)

            if (iteration + 1) % self.print_every == 0 and (iteration + 1) != N:
                save_checkpoint(result_dir, stats, self.executor.molecule_list, iteration, initialize)
                if not initialize and self.print_stats:
                    self._print_stats(iteration, stats)

        else:
            save_checkpoint(result_dir, stats, self.executor.molecule_list, iteration, initialize)
            self._print_stats(iteration, stats)
            print(self.output)

        return

    def widom(self, N):
        energies = []
        for _ in range(N):
            dE = self.executor._widom()
            energies.append(dE)

        from graphcmc.utils.io import setup_result_directory
        result_dir = setup_result_directory(self.result_dir, self.job_id)
        np.save(f"{result_dir}/widom.npy", np.array(energies))

    def _print_stats(self, iteration, stats):
        self.output += f"Production cycle: {iteration + 1}\n\n"

        moves = ['Insertion', 'Deletion', 'Translation', 'Rotation']
        for i, move in enumerate(moves):
            self.output += f"\t**{move}**\n\tAttempted: {stats['attempted'][i]}\n\tAccepted: {stats['accepted'][i]}\n\n"
            if stats['attempted'][i]:
                self.output += f"\tAcceptance Ratio: {stats['accepted'][i] / stats['attempted'][i] * 100:.2f} %\n\n"

        self.output += "\t----------------------------------------\n\n"
        self.output += f"\tAdsorption loading: {np.array(stats['uptake']).mean() / self.x / self.y / self.z:.5f} per unit cell\n\n"
        self.output += "################################################\n\n"
