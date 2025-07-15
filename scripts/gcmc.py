import os
import shutil
import numpy as np

from enum import Enum
from constants import BOLTZMANN, AVOGADRO
from file_io import (
    setup_result_directory,
    save_checkpoint,
    load_simulation_state,
    load_stats,
    save_stats
)
from utilities import _random_translation, _random_rotation, _random_position

class Move(Enum):
    INSERTION = 0
    DELETION = 1
    TRANSLATION = 2
    ROTATION = 3

class GCMC():
    def __init__(self, args, forcefield, atoms_frame, atoms_ads, vdw_radii):
        self.forcefield = forcefield
        self.adsorbate = args.adsorbate
        self.atoms_frame = atoms_frame
        self.n_frame = len(self.atoms_frame)
        self.atoms_ads = atoms_ads
        self.n_ads = len(self.atoms_ads)

        self.cell = np.array(self.atoms_frame.get_cell()) / 1e10
        self.V = np.linalg.det(self.cell[:])
        self.fugacity = args.fugacity
        self.beta = 1 / (BOLTZMANN * args.T)

        self.energy_shift = args.energy_shift * 1000 / AVOGADRO
        self.print_every = args.print_every
        self.minimum_inner_steps = args.minimum_inner_steps
        self.continue_sim = args.continue_sim

        self.job_id = f'{args.framework}_{self.adsorbate}_{args.T}K_{int(args.P):07d}Pa_{args.FF}'
        if args.energy_shift:
            self.job_id += f"_shift{args.energy_shift:.1f}"
        
        self.result_dir = setup_result_directory(self.job_id, self.continue_sim)

        if not self.continue_sim:
            self.atoms = self.atoms_frame.copy()
            self.Z_ads = 0
            self.E = 0

    def get_potential_energy(self, new_atoms, old_atoms = None, old_e = None, i_ads = None):
        return self.forcefield.get_potential_energy(new_atoms, old_atoms, old_e, i_ads, shift = self.energy_shift)

    def _accept(self, delta_e, factor):
        return np.random.rand() < min(1, factor * np.exp(-self.beta * delta_e))
    
    def run(self, N, initialize = False):
        atoms_ads = self.atoms_ads.copy()

        if self.continue_sim:
            if initialize and os.path.isfile(f'results/{self.job_id}/uptake.npy'):
                return
            uptake, already_run, adsorption_energy, self.atoms = load_simulation_state(self.result_dir, self.atoms_frame, atoms_ads, self.n_ads, initialize)
            self.Z_ads = uptake[-1]
            self.E = adsorption_energy[-1]
            self.accepted, self.attempted = load_stats(self.result_dir)
        else:
            uptake, already_run, adsorption_energy = [], 0, []
            self.accepted, self.attempted = [0] * 4, [0] * 4

        for iteration in range(already_run, N):
            for _ in range(max(self.minimum_inner_steps, self.Z_ads)):
                move = np.random.choice(list(Move))
                self._execute_move(move, atoms_ads)

            uptake.append(self.Z_ads)
            adsorption_energy.append(self.E)

            if (iteration + 1) % self.print_every == 0:
                save_checkpoint(self.result_dir, self.atoms, uptake, adsorption_energy, self.Z_ads, self.n_ads, iteration, initialize)
                if not initialize:
                    save_stats(self.result_dir, self.accepted, self.attempted)

        if initialize:
            init_files = ["initialization_uptake.npy", "initialization_adsorption_energy.npy"]
            for f in init_files:
                p = os.path.join(self.result_dir, f)
                if os.path.exists(p):
                    os.remove(p)
            self.continue_sim = False

        else:
            save_checkpoint(self.result_dir, self.atoms, uptake, adsorption_energy, self.Z_ads, self.n_ads)
            save_stats(self.result_dir, self.accepted, self.attempted)
            self._print_stats()

        return np.array(uptake).mean()
    
    def _execute_move(self, move, atoms_ads):
        if move == Move.INSERTION:
            self.attempted[0] += 1
            self.Z_ads += 1
            atoms_trial = self.atoms.copy() + atoms_ads
            pos = atoms_trial.get_positions()
            pos[-self.n_ads:] = _random_position(pos[-self.n_ads:], atoms_trial.get_cell())
            atoms_trial.set_positions(pos)
            e_trial = self.get_potential_energy(atoms_trial, self.atoms, self.E, self.Z_ads - 1)
            
            if self._accept(e_trial - self.E, self.V * self.beta * self.fugacity / self.Z_ads):
                self.atoms = atoms_trial
                self.E = e_trial
                self.accepted[0] += 1
            else:
                self.Z_ads -= 1

        elif move == Move.DELETION and self.Z_ads > 0:
            self.attempted[1] += 1
            i_ads = np.random.randint(self.Z_ads)
            atoms_trial = self.atoms.copy()
            self.Z_ads -= 1
            del atoms_trial[self.n_frame + self.n_ads * i_ads : self.n_frame + self.n_ads * (i_ads + 1)]
            e_trial = self.get_potential_energy(atoms_trial, self.atoms, self.E, i_ads)
            
            if self._accept(e_trial - self.E, (self.Z_ads + 1) / self.V / self.beta / self.fugacity):
                self.atoms = atoms_trial
                self.E = e_trial
                self.accepted[1] += 1
            else:
                self.Z_ads += 1

        elif move == Move.TRANSLATION and self.Z_ads > 0:
            self.attempted[2] += 1
            i_ads = np.random.randint(self.Z_ads)
            atoms_trial = self.atoms.copy()
            pos = atoms_trial.get_positions()
            pos[self.n_frame + self.n_ads * i_ads : self.n_frame + self.n_ads * (i_ads + 1)] = _random_translation(pos[self.n_frame + self.n_ads * i_ads : self.n_frame + self.n_ads * (i_ads + 1)], atoms_trial.get_cell())
            atoms_trial.set_positions(pos)
            e_trial = self.get_potential_energy(atoms_trial, self.atoms, self.E, i_ads)
            
            if self._accept(e_trial - self.E, 1):
                self.atoms = atoms_trial
                self.E = e_trial
                self.accepted[2] += 1

        elif move == Move.ROTATION and self.Z_ads > 0:
            self.attempted[3] += 1
            i_ads = np.random.randint(self.Z_ads)
            atoms_trial = self.atoms.copy()
            pos = atoms_trial.get_positions()
            pos[self.n_frame + self.n_ads * i_ads : self.n_frame + self.n_ads * (i_ads + 1)] = _random_rotation(pos[self.n_frame + self.n_ads * i_ads : self.n_frame + self.n_ads * (i_ads + 1)])
            atoms_trial.set_positions(pos)
            e_trial = self.get_potential_energy(atoms_trial, self.atoms, self.E, i_ads)
            
            if self._accept(e_trial - self.E, 1):
                self.atoms = atoms_trial
                self.E = e_trial
                self.accepted[3] += 1

    def _print_stats(self):
        moves = ['Insertion', 'Deletion', 'Translation', 'Rotation']
        for i, move in enumerate(moves):
            print(f'{move}\nAttempted: {self.attempted[i]}\nAccepted: {self.accepted[i]}')
            if self.attempted[i]:
                print(f'Acceptance Ratio: {self.accepted[i] / self.attempted[i] * 100:.5f}%')

    def widom(self, N):
        atoms_ads = self.atoms_ads.copy()
        positions = []
        energies = []
        for _ in range(N):
            atoms_trial = self.atoms.copy() + atoms_ads
            pos = atoms_trial.get_positions()
            pos[-self.n_ads:] = _random_position(pos[-self.n_ads:], atoms_trial.get_cell())
            atoms_trial.set_positions(pos)
            e_trial = self.get_potential_energy(atoms_trial.copy(), self.atoms.copy(), 0, 0)

            positions.append(list(pos))
            energies.append(e_trial)

        np.save(f'results/{self.job_id}/widom_positions.npy', np.array(positions)[:, -3:])
        np.save(f'results/{self.job_id}/widom_energies.npy', np.array(energies) / 1000 * AVOGADRO)
