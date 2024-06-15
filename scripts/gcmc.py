import os
import shutil
import numpy as np

from ase.io import read, write
from utilities import _random_rotation, _random_position, vdw_overlap

BOLTZMANN    = 1.380649e-23

class GCMC():
    def __init__(self, args, model, atoms_frame, atoms_ads, fugacity, vdw_radii):
        self.model = model
        self.adsorbate = args.adsorbate
        self.atoms_frame = atoms_frame
        self.n_frame = len(self.atoms_frame)
        self.atoms_ads = atoms_ads
        self.n_ads = len(self.atoms_ads)

        self.cell = np.array(self.atoms_frame.get_cell()) / 1e10
        self.V = np.linalg.det(self.cell)
        self.T = args.T
        self.P = args.P
        self.fugacity = fugacity
        self.beta = 1 / (BOLTZMANN * args.T)
        self.vdw = vdw_radii - 0.35

        self.device = args.device
        self.FF = args.FF

        self.print_every = args.print_every
        self.minimum_inner_steps = args.minimum_inner_steps
        self.continue_sim = args.continue_sim

        if not os.path.exists('results'):
            os.mkdir('results')
        self.job_id = f'{args.framework}_{args.adsorbate}_{self.T}K_{int(self.P):07d}Pa_{args.FF}'

        if not self.continue_sim:
            if os.path.isdir(f'results/{self.job_id}'):
                shutil.rmtree(f'results/{self.job_id}')
            os.mkdir(f'results/{self.job_id}')

            self.atoms = self.atoms_frame.copy()
            self.Z_ads = 0
            self.E = 0

    def _insertion_acceptance(self, e_trial, e):
        exp_value = -self.beta * (e_trial - e)
        # if exp_value > 100:
        #     return True
        # elif exp_value < -100:
        #     return False
        # else:
        acc = min(1, self.V * self.beta * self.fugacity / self.Z_ads * np.exp(exp_value))
        return np.random.rand() < acc

    def _deletion_acceptance(self, e_trial, e):
        exp_value = -self.beta * (e_trial - e)
        # if exp_value > 100:
        #     return True
        # else:
        acc = min(1, (self.Z_ads + 1) / self.V / self.beta / self.fugacity * np.exp(exp_value))
        return np.random.rand() < acc

    def get_potential_energy(self, new_atoms, old_atoms = None, old_e = None, i_ads = None):
        if old_atoms is None:
            return 0
        else:
            return self.model.get_potential_energy(new_atoms, old_atoms, old_e, i_ads)

    def run(self, N, initialize = False):
        atoms_ads = self.atoms_ads.copy()

        if not self.continue_sim:
            uptake = []
            adsorption_energy = []
            already_run = 0
        else:
            self.atoms = read(f'results/{self.job_id}/last_movie.cif')
            uptake = list(np.load(f'results/{self.job_id}/uptake.npy'))
            adsorption_energy = list(np.load(f'results/{self.job_id}/adsorption_energy.npy'))
            already_run = len(uptake)
            self.Z_ads = uptake[-1]
            self.E = adsorption_energy[-1]

        attempted = [0, 0, 0, 0]
        accepted = [0, 0, 0, 0]
        for iteration in range(N - already_run):
            for _ in range(max(self.minimum_inner_steps, self.Z_ads)):
                switch = np.random.rand()
                # Insertion
                if switch < 0.25:
                    attempted[0] += 1
                    self.Z_ads += 1
                    atoms_trial = self.atoms.copy() + atoms_ads
                    pos = atoms_trial.get_positions()
                    pos[-self.n_ads:] = _random_position(pos[-self.n_ads:], atoms_trial.get_cell())
                    atoms_trial.set_positions(pos)
                    if self.model.hybrid and vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, self.Z_ads - 1):
                        e_trial = 10**10
                    else:
                        e_trial, _, _, _ = self.get_potential_energy(atoms_trial.copy(), self.atoms.copy(), self.E, self.Z_ads - 1)
                    if self._insertion_acceptance(e_trial, self.E):
                        self.atoms = atoms_trial.copy()
                        self.E = e_trial
                        accepted[0] += 1
                    else:
                        self.Z_ads -= 1

                # Deletion
                elif switch < 0.5:
                    attempted[1] += 1
                    if self.Z_ads != 0:
                        i_ads = np.random.randint(self.Z_ads)
                        atoms_trial = self.atoms.copy()
                        self.Z_ads -= 1
                        del atoms_trial[self.n_frame + self.n_ads * i_ads : self.n_frame + self.n_ads * (i_ads + 1)]
                        e_trial, _, _, _ = self.get_potential_energy(atoms_trial, self.atoms, self.E, i_ads)
                        if self._deletion_acceptance(e_trial, self.E):
                            self.atoms = atoms_trial.copy()
                            self.E = e_trial
                            accepted[1] += 1
                        else:
                            self.Z_ads += 1

                # Translation
                elif switch < 0.75:
                    if self.Z_ads != 0:
                        attempted[2] += 1
                        i_ads = np.random.randint(self.Z_ads)
                        atoms_trial = self.atoms.copy()
                        # pos = atoms_trial.get_positions()
                        # pos[self.n_frame + self.n_ads * i_ads : self.n_frame + self.n_ads * (i_ads + 1)] += 0.5 * (np.random.rand(3) - 0.5)
                        # atoms_trial.set_positions(pos)
                        sca_pos = atoms_trial.get_scaled_positions()
                        sca_pos[self.n_frame + self.n_ads * i_ads : self.n_frame + self.n_ads * (i_ads + 1)] += np.random.rand(3) - 0.5
                        atoms_trial.set_scaled_positions(sca_pos)
                        if self.model.hybrid and vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
                            e_trial = 10**10
                        else:
                            e_trial, _, _, _ = self.get_potential_energy(atoms_trial, self.atoms, self.E, i_ads)
                        acc = min(1, np.exp(-self.beta * (e_trial - self.E)))
                        if np.random.rand() < acc:
                            self.atoms = atoms_trial.copy()
                            self.E = e_trial
                            accepted[2] += 1

                # Rotation
                else:
                    if self.Z_ads != 0:
                        attempted[3] += 1
                        i_ads = np.random.randint(self.Z_ads)
                        atoms_trial = self.atoms.copy()
                        pos = atoms_trial.get_positions()
                        pos[self.n_frame + self.n_ads * i_ads : self.n_frame + self.n_ads * (i_ads + 1)] = _random_rotation(pos[self.n_frame + self.n_ads * i_ads : self.n_frame + self.n_ads * (i_ads + 1)], circlefrac = 1.0)
                        atoms_trial.set_positions(pos)
                        if self.model.hybrid and vdw_overlap(atoms_trial, self.vdw, self.n_frame, self.n_ads, i_ads):
                            e_trial = 10**10
                        else:
                            e_trial, _, _, _ = self.get_potential_energy(atoms_trial, self.atoms, self.E, i_ads)
                        acc = min(1, np.exp(-self.beta * (e_trial - self.E)))
                        if np.random.rand() < acc:
                            self.atoms = atoms_trial.copy()
                            self.E = e_trial
                            accepted[3] += 1

            uptake.append(self.Z_ads)
            adsorption_energy.append(self.E)

            if not initialize and (iteration + 1) % self.print_every == 0:
                np.save(f'results/{self.job_id}/uptake.npy', np.array(uptake))
                np.save(f'results/{self.job_id}/adsorption_energy.npy', np.array(adsorption_energy))
                write(f'results/{self.job_id}/last_movie.cif', self.atoms)

        if not initialize:
            print('Attempted')
            print(f'Insertion: {attempted[0]}\nDeletion: {attempted[1]}\nTranslation: {attempted[2]}\nRotation: {attempted[3]}')
            print('Accepted')
            print(f'Insertion: {accepted[0]}\nDeletion: {accepted[1]}\nTranslation: {accepted[2]}\nRotation: {accepted[3]}')
            print('Acceptance Ratio')
            print(f'Insertion: {accepted[0] / attempted[0] * 100:.5f}%\nDeletion: {accepted[1] / attempted[1] * 100:.5f}%\nTranslation: {accepted[2] / attempted[2] * 100:.5f}%\nRotation: {accepted[3] / attempted[3] * 100:.5f}%')

        np.save(f'results/{self.job_id}/uptake.npy', np.array(uptake))
        np.save(f'results/{self.job_id}/adsorption_energy.npy', np.array(adsorption_energy))
        write(f'results/{self.job_id}/last_movie.cif', self.atoms)

        return np.array(uptake).mean()
