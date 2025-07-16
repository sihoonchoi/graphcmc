import numpy as np
import os

class GCMC:
    def __init__(self, framework, ads, T, P,
                 hybrid = False,
                 mlff = None,
                 vdw_cutoff = 14.0,
                 tail_correction = True,
                 charge = True,
                 device = 'cpu',
                 print_every = 1000,
                 minimum_inner_steps = 10,
                 continue_sim = False):
        
        from ase.io import read
        from graphcmc.simulation import ClassicalForceField, HybridForceField, MoveExecutor
        from graphcmc.simulation.eos import PREOS
        from graphcmc.utils.math import compute_supercell_size
        from graphcmc.utils.constants import BOLTZMANN
        from graphcmc.utils.io import load_atoms_and_charges, setup_result_directory

        self.unitcell, self.ads = load_atoms_and_charges(framework, ads, charge)
        self.cell = np.array(self.unitcell.get_cell())
        self.x, self.y, self.z = compute_supercell_size(self.cell, vdw_cutoff)
        self.supercell = self.unitcell * (self.x, self.y, self.z)

        if not hybrid or mlff is None:
            forcefield = ClassicalForceField(self.supercell, self.unitcell, self.ads, vdw_cutoff, tail_correction, charge, device)
        else:
            forcefield = HybridForceField(self.supercell, self.framework, self.ads, vdw_cutoff, tail_correction, charge, device, mlff)

        self.executor = MoveExecutor(forcefield, self.supercell, self.ads)
        self.n_unitcell_atoms = len(self.unitcell)
        self.n_ads_atoms = len(self.ads)

        self.T = T
        self.P = P
        self.fugacity = PREOS.from_name(self.ads).calculate_fugacity(self.T, self.P)
        self.beta = 1 / (BOLTZMANN * T)

        self.print_every = print_every
        self.minimum_inner_steps = minimum_inner_steps
        self.continue_sim = continue_sim

        self.job_id = f'{framework}_{ads}_{T}K_{int(P):07d}Pa_{forcefield}'
        self.result_dir = setup_result_directory(self.job_id, self.continue_sim)

    def mc(self, N, initialize = False):
        from graphcmc.utils.io import load_simulation_state, load_stats, save_checkpoint, save_stats

        if self.continue_sim:
            if initialize and os.path.isfile(os.path.join(self.result_dir, 'uptake.npy')):
                return
            uptake, already_run, adsorption_energy, molecule_list = load_simulation_state(self.result_dir, initialize)
            self.executor.molecule_list = molecule_list
            self.executor.energy = adsorption_energy[-1]
            self.accepted, self.attempted = load_stats(self.result_dir)
        else:
            uptake, already_run, adsorption_energy = [], 0, []
            self.accepted, self.attempted = [0] * 4, [0] * 4
            
        for iteration in range(already_run, N):
            for _ in range(max(self.minimum_inner_steps, self.executor.count_adsorbates())):
                i, accepted = self.executor.perform_move(self.fugacity, self.beta)
                self.attempted[i] += 1
                self.accepted[i] += accepted

            uptake.append(self.executor.count_adsorbates())
            adsorption_energy.append(self.executor.energy)

            if (iteration + 1) % self.print_every == 0:
                save_checkpoint(self.result_dir, uptake, adsorption_energy, self.executor.molecule_list, iteration, initialize)
                if not initialize:
                    save_stats(self.result_dir, self.accepted, self.attempted)
                    self._print_stats(iteration, uptake)

        if initialize:
            init_files = ["initialization_uptake.npy", "initialization_adsorption_energy.npy"]
            for f in init_files:
                p = os.path.join(self.result_dir, f)
                if os.path.exists(p):
                    os.remove(p)
            self.continue_sim = False
        
        else:
            save_checkpoint(self.result_dir, uptake, adsorption_energy, self.executor.molecule_list, iteration, initialize)
            save_stats(self.result_dir, self.accepted, self.attempted)
        
        return

    def widom(self, N):
        energies = []
        for _ in range(N):
            dE = self.executor._widom()
            energies.append(dE)

        np.save(f'results/{self.job_id}/widom_energies.npy', np.array(energies))

    def _print_stats(self, iteration, uptake):
        print(f'\tProduction cycle: {iteration + 1}')

        moves = ['Insertion', 'Deletion', 'Translation', 'Rotation']
        for i, move in enumerate(moves):
            print(f'{move}\nAttempted: {self.attempted[i]}\nAccepted: {self.accepted[i]}')
            if self.attempted[i]:
                print(f'Acceptance Ratio: {self.accepted[i] / self.attempted[i] * 100:.5f}%')

        print()
        print(f'Adsorption loading: {np.array(uptake).mean() / self.x / self.y / self.z} per unit cell')
