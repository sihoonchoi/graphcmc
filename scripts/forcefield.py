import numpy as np
import torch
import json

from ase import Atoms

#  - AUTOA     =  1. a.u. in Angstroem
#  - RYTOEV    =  1 Ry in Ev
#  - FELECT    =  (the electronic charge)/(4*pi*the permittivity of free space)
#                  in atomic units this is just e^2

AUTOA        = 0.529177249
RYTOEV       = 13.605826
FELECT       = 2 * AUTOA * RYTOEV

BOLTZMANN    = 1.380649e-23
EV_TO_KJ_MOL = 96.48530749925793
MOL          = 6.02214076e23
J_TO_EV      = MOL / 1000.0 / EV_TO_KJ_MOL

class forcefield():
    def __init__(self, frame, unitcell, ads, hybrid = False, mlff = None, vdw_cutoff = 14.0, tail_correction = True, charge = True, device = 'cpu'):
        self.frame = frame
        self.unitcell = unitcell
        self.ads = ads
        self.n_frame_atoms = len(self.frame)
        self.n_ads_atoms = len(ads)
        self.V = np.linalg.det(self.frame.cell[:])

        self.hybrid = hybrid
        self.vdw_cutoff = vdw_cutoff
        self.tail_correction = tail_correction
        self.charge = charge
        self.device = device

        if mlff is not None:
            self.model = mlff

        with open('scripts/data/lj_params.json', 'r') as f:
            self.params = json.load(f)

        self.ads_params = np.array([[self.params[s]['sigma'], self.params[s]['epsilon']] for s in self.ads.get_chemical_symbols()])

    def get_tail_correction(self, elements, start_idx, tail_correction = True):
        if tail_correction:
            symbols, counts = np.unique(elements[start_idx:], return_counts = True)

            U_tail = 0
            for s, c in zip(symbols, counts):
                for y, u in zip(symbols, counts):
                    epsilon = np.sqrt(self.params[s]['epsilon'] * self.params[y]['epsilon'])
                    sigma = (self.params[s]['sigma'] + self.params[y]['sigma']) / 2.
                    U_tail += 2 * np.pi / self.V * c * u * 4 / 3 * epsilon * sigma**3 * (((sigma / self.vdw_cutoff)**9) / 3 - (sigma / self.vdw_cutoff)**3)
            return U_tail * BOLTZMANN
        else:
            return 0

    def insertion(self, atoms, chemical_symbols, initial_charges, i_ads):
        start_idx = 0
        ml = 0
        vdw = 0
        ewald = 0

        ## MLFF
        if self.hybrid:
            start_idx = self.n_frame_atoms
            temp_ads = atoms[(self.n_frame_atoms + i_ads * self.n_ads_atoms):(self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms)].copy()
            temp_ads.set_cell(self.unitcell.cell)
            adjusted_pos = (temp_ads.get_scaled_positions()) @ self.unitcell.cell
            temp_ads = Atoms(['C', 'O', 'O'], positions = adjusted_pos, cell = self.unitcell.cell, pbc = [True, True, True])
            
            temp_atoms = self.unitcell.copy() + temp_ads
            temp_atoms.calc = self.model
            ml += temp_atoms.get_potential_energy() / J_TO_EV

            if len(atoms) == self.n_frame_atoms + self.n_ads_atoms:
                return ml, vdw, ewald

        ## Classical FF
        wo_ads_idx = torch.tensor(list(np.arange(len(atoms))[start_idx:self.n_frame_atoms + i_ads * self.n_ads_atoms]) + list(np.arange(len(atoms))[self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms:]), dtype = torch.int32, device = self.device)
        for i in range(self.n_ads_atoms):
            ads_idx = self.n_frame_atoms + i_ads * self.n_ads_atoms + i

            new_dist = atoms.get_distances(ads_idx, wo_ads_idx.cpu().numpy(), mic = True, vector = False)
            new_dist = torch.from_numpy(new_dist).float().to(self.device)
            atoms_idx = wo_ads_idx[new_dist < self.vdw_cutoff]
            new_dist = new_dist[new_dist < self.vdw_cutoff]

            if new_dist.shape[0]:
                params = torch.tensor([[self.params[o]['sigma'], self.params[o]['epsilon']] for o in chemical_symbols[atoms_idx.cpu().numpy()]], dtype = torch.float32, device = self.device)
                mixing_sigma = (params[:, 0] + self.ads_params[i][0]) / 2
                mixing_epsilon = torch.sqrt(params[:, 1] * self.ads_params[i][1])

                vdw += (4 * mixing_epsilon * ((mixing_sigma / new_dist).pow(12) - (mixing_sigma / new_dist).pow(6))).sum().item() * BOLTZMANN

        ## Ewald summation
        if self.charge:
            ewald += ewaldsum(atoms, initial_charges, wo_ads_idx, torch.tensor([self.n_frame_atoms + i_ads * self.n_ads_atoms + i for i in range(self.n_ads_atoms)], dtype = torch.int32, device = self.device), device = self.device).get_ewaldsum() / J_TO_EV

        return ml, vdw, ewald
    
    def deletion(self, atoms, chemical_symbols, initial_charges, i_ads):
        start_idx = 0
        ml = 0
        vdw = 0
        ewald = 0

        ## MLFF
        if self.hybrid:
            start_idx = self.n_frame_atoms
            temp_ads = atoms[(self.n_frame_atoms + i_ads * self.n_ads_atoms):(self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms)].copy()
            temp_ads.set_cell(self.unitcell.cell)
            adjusted_pos = (temp_ads.get_scaled_positions()) @ self.unitcell.cell
            temp_ads = Atoms(['C', 'O', 'O'], positions = adjusted_pos, cell = self.unitcell.cell, pbc = [True, True, True])

            temp_atoms = self.unitcell.copy() + temp_ads
            temp_atoms.calc = self.model
            ml -= temp_atoms.get_potential_energy() / J_TO_EV

        ## Classical FF
        wo_ads_idx = torch.tensor(list(np.arange(len(atoms))[start_idx:self.n_frame_atoms + i_ads * self.n_ads_atoms]) + list(np.arange(len(atoms))[self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms:]), dtype = torch.int32, device = self.device)
        for i in range(self.n_ads_atoms):
            ads_idx = self.n_frame_atoms + i_ads * self.n_ads_atoms + i
            
            dist = atoms.get_distances(ads_idx, wo_ads_idx.cpu().numpy(), mic = True, vector = False)
            dist = torch.from_numpy(dist).float().to(self.device)
            atoms_idx = wo_ads_idx[dist < self.vdw_cutoff]
            dist = dist[dist < self.vdw_cutoff]
            
            if dist.shape[0]:
                params = torch.tensor([[self.params[o]['sigma'], self.params[o]['epsilon']] for o in chemical_symbols[atoms_idx.cpu().numpy()]], dtype = torch.float32, device = self.device)
                mixing_sigma = (params[:, 0] + self.ads_params[i][0]) / 2
                mixing_epsilon = torch.sqrt(params[:, 1] * self.ads_params[i][1])

                vdw -= (4 * mixing_epsilon * ((mixing_sigma / dist).pow(12) - (mixing_sigma / dist).pow(6))).sum().item() * BOLTZMANN

        ## Ewald summation
        if self.charge:
            ewald -= ewaldsum(atoms, initial_charges, wo_ads_idx, torch.tensor([self.n_frame_atoms + i_ads * self.n_ads_atoms + i for i in range(self.n_ads_atoms)], dtype = torch.int32, device = self.device), device = self.device).get_ewaldsum() / J_TO_EV

        return ml, vdw, ewald

    def get_potential_energy(self, new_atoms, old_atoms, old_e, i_ads, shift = 0):
        if self.n_frame_atoms == len(new_atoms) or old_atoms is None:
            return 0
        
        else:
            if self.hybrid:
                start_idx = self.n_frame_atoms
            else:
                start_idx = 0

            with torch.no_grad():
                old_chemical_symbols = np.array(old_atoms.get_chemical_symbols())
                new_chemical_symbols = np.array(new_atoms.get_chemical_symbols())
                old_initial_charges = torch.from_numpy(np.array(old_atoms.get_initial_charges())).float().to(self.device)
                new_initial_charges = torch.from_numpy(np.array(new_atoms.get_initial_charges())).float().to(self.device)

                old_tail = self.get_tail_correction(old_chemical_symbols, start_idx, self.tail_correction)
                new_tail = self.get_tail_correction(new_chemical_symbols, start_idx, self.tail_correction)
                tail = new_tail - old_tail

                # Insertion
                if len(old_atoms) < len(new_atoms):
                    ml, vdw, ewald = self.insertion(new_atoms, new_chemical_symbols, new_initial_charges, i_ads)
                    return old_e + ml + vdw + ewald + tail

                # Deletion
                elif len(old_atoms) > len(new_atoms):
                    if len(new_atoms) == self.n_frame_atoms:
                        return 0
                    
                    ml, vdw, ewald = self.deletion(old_atoms, old_chemical_symbols, old_initial_charges, i_ads)
                    return old_e + ml + vdw + ewald + tail

                # Rotation or translation
                else:
                    ml, vdw, ewald = self.deletion(old_atoms, old_chemical_symbols, old_initial_charges, i_ads)
                    new_e = old_e + ml + vdw + ewald

                    ml, vdw, ewald = self.insertion(new_atoms, new_chemical_symbols, new_initial_charges, i_ads)
                    new_e += ml + vdw + ewald

                    return new_e
                
                    # if self.hybrid:
                    #     temp_ads_delete = old_atoms[(self.n_frame_atoms + i_ads * self.n_ads_atoms):(self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms)].copy()
                    #     temp_ads_delete.set_cell(self.unitcell.cell)
                    #     adjusted_pos = (temp_ads_delete.get_scaled_positions()) @ self.unitcell.cell
                    #     temp_ads_delete = Atoms(['C', 'O', 'O'], positions = adjusted_pos, cell = self.unitcell.cell, pbc = [True, True, True])

                    #     temp_atoms = self.unitcell.copy() + temp_ads_delete
                    #     temp_atoms.calc = self.model
                    #     ml -= temp_atoms.get_potential_energy() / J_TO_EV

                    #     temp_ads_insert = new_atoms[(self.n_frame_atoms + i_ads * self.n_ads_atoms):(self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms)].copy()
                    #     temp_ads_insert.set_cell(self.unitcell.cell)
                    #     adjusted_pos = (temp_ads_insert.get_scaled_positions()) @ self.unitcell.cell
                    #     temp_ads_insert = Atoms(['C', 'O', 'O'], positions = adjusted_pos, cell = self.unitcell.cell, pbc = [True, True, True])
                        
                    #     temp_atoms = self.unitcell.copy() + temp_ads_insert
                    #     temp_atoms.calc = self.model
                    #     ml += temp_atoms.get_potential_energy() / J_TO_EV

                    #     if self.n_frame_atoms + self.n_ads_atoms == len(new_atoms):
                    #         return old_e + ml + vdw + ewald

                    # wo_ads_idx = torch.tensor(list(np.arange(len(old_atoms))[start_idx:self.n_frame_atoms + i_ads * self.n_ads_atoms]) + list(np.arange(len(old_atoms))[self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms:]), dtype = torch.int32, device = self.device)
                    # for i in range(self.n_ads_atoms):
                    #     ads_idx = self.n_frame_atoms + i_ads * self.n_ads_atoms + i
                        
                    #     old_dist = old_atoms.get_distances(ads_idx, wo_ads_idx.cpu().numpy(), mic = True, vector = False)
                    #     old_dist = torch.from_numpy(old_dist).float().to(self.device)
                    #     old_atoms_idx = wo_ads_idx[old_dist < self.vdw_cutoff]
                    #     old_dist = old_dist[old_dist < self.vdw_cutoff]

                    #     if old_dist.shape[0]:
                    #         params = torch.tensor([[self.params[o]['sigma'], self.params[o]['epsilon']] for o in old_chemical_symbols[old_atoms_idx.cpu().numpy()]], dtype = torch.float32, device = self.device)
                    #         mixing_sigma = (params[:, 0] + self.ads_params[i][0]) / 2
                    #         mixing_epsilon = torch.sqrt(params[:, 1] * self.ads_params[i][1])

                    #         vdw -= (4 * mixing_epsilon * ((mixing_sigma / old_dist).pow(12) - (mixing_sigma / old_dist).pow(6))).sum().item() * BOLTZMANN

                    #     new_dist = new_atoms.get_distances(ads_idx, wo_ads_idx.cpu().numpy(), mic = True, vector = False)
                    #     new_dist = torch.from_numpy(new_dist).float().to(self.device)
                    #     new_atoms_idx = wo_ads_idx[new_dist < self.vdw_cutoff]
                    #     new_dist = new_dist[new_dist < self.vdw_cutoff]

                    #     if new_dist.shape[0]:
                    #         params = torch.tensor([[self.params[o]['sigma'], self.params[o]['epsilon']] for o in new_chemical_symbols[new_atoms_idx.cpu().numpy()]], dtype = torch.float32, device = self.device)
                    #         mixing_sigma = (params[:, 0] + self.ads_params[i][0]) / 2
                    #         mixing_epsilon = torch.sqrt(params[:, 1] * self.ads_params[i][1])

                    #         vdw += (4 * mixing_epsilon * ((mixing_sigma / new_dist).pow(12) - (mixing_sigma / new_dist).pow(6))).sum().item() * BOLTZMANN

                    # if self.charge:
                    #     ewald -= ewaldsum(old_atoms, old_initial_charges, wo_ads_idx, torch.tensor([self.n_frame_atoms + i_ads * self.n_ads_atoms + i for i in range(self.n_ads_atoms)], dtype = torch.int32, device = self.device), device = self.device).get_ewaldsum() / J_TO_EV
                    #     ewald += ewaldsum(new_atoms, new_initial_charges, wo_ads_idx, torch.tensor([self.n_frame_atoms + i_ads * self.n_ads_atoms + i for i in range(self.n_ads_atoms)], dtype = torch.int32, device = self.device), device = self.device).get_ewaldsum() / J_TO_EV

                    # return old_e + ml + vdw + ewald


class ewaldsum(object):
    def __init__(self, atoms, Z, frame_idx, ads_idx, device = 'cpu',
            eta: float = None,
            Rcut: float = 4.0,
            Gcut: float = 4.0):
        
        self.device = device
        self._atoms  = atoms
        self._na = len(self._atoms)
        self._iframe = frame_idx
        self._iads = ads_idx
        self._scapos = torch.from_numpy(self._atoms.get_scaled_positions()).float().to(self.device)

        self._Acell = torch.from_numpy(self._atoms.cell[:]).float().to(self.device)        # real-space cell
        self._Bcell = torch.inverse(self._Acell).transpose(0, 1)      # reciprocal-space cell
        self._omega = torch.det(self._Acell)        # Volume of real-space cell

        self._ZZ = Z
        ZZ_frame_mesh, ZZ_ads_mesh = torch.meshgrid(self._ZZ[self._iframe], self._ZZ[self._iads], indexing = 'ij')
        self._Zij = ZZ_frame_mesh * ZZ_ads_mesh

        self._inv_4pi_epsilon0 = FELECT

        if eta is None:
            self._eta = torch.sqrt(torch.tensor(np.pi, device = self.device)) / (self._omega)**(1./3)
        else:
            self._eta = torch.tensor(eta, dtype = torch.float32, device = self.device)

        self._Rcut = Rcut
        self._Gcut = Gcut

    def get_sum_real(self):
        '''
        Real-space contribution to the Ewald sum.

                 1                              erfc(eta | r_ij + R_N |)
            U = --- \sum_{ij} \sum'_N Z_i Z_j -----------------------------
                 2                                    | r_ij + R_N |

        where the prime in \sum_N means i != j when N = 0.
        '''
        with torch.no_grad():
            ii, jj = torch.meshgrid(self._iframe, self._iads)
            rij = self._scapos[ii] - self._scapos[jj]

            # move rij to the range [-0.5, 0.5]
            rij = torch.where(rij >= 0.5, rij - 1.0, rij)
            rij = torch.where(rij < -0.5, rij + 1.0, rij)
        
            ############################################################
            # contribution from N = 0 cell
            ############################################################
            rij0 = torch.norm(
                torch.tensordot(self._Acell, rij.permute(2, 1, 0), dims = ([0], [0])),
                dim = 0)
            Uij = torch.erfc(rij0 * self._eta) / rij0

            ############################################################
            # contribution from N != 0 cells
            ############################################################
            rij = rij.permute(2, 1, 0).reshape(3, -1)
            nx, ny, nz = ((self._Rcut / self._eta / torch.norm(self._Acell, dim = 1)).to(torch.int32)) + 1
            
            Rn = torch.stack(torch.meshgrid(
                torch.arange(-nx.item(), nx.item() + 1, dtype = torch.int32, device = self.device),
                torch.arange(-ny.item(), ny.item() + 1, dtype = torch.int32, device = self.device),
                torch.arange(-nz.item(), nz.item() + 1, dtype = torch.int32, device = self.device)
            )).reshape(3, -1)
        
            # remove N = 0 term
            cut = torch.sum(torch.abs(Rn), dim = 0) != 0
            Rn  = Rn[:, cut]

            # R_N + rij
            Rr = torch.norm(
                torch.tensordot(self._Acell, Rn[:, None, :] + rij[:, :, None], dims = ([0], [0])),
                dim = 0)

            Uij += torch.sum(
                torch.erfc(self._eta * Rr) / Rr, dim = 1
            ).reshape((len(self._iads), len(self._iframe)))

            return Uij


    def get_sum_recp(self):
        '''
        Reciprocal-space contribution to the Ewald sum.

                  1            4pi              
            U = ----- \sum'_G ----- exp(-G^2/(4 eta^2)) \sum_{ij} Z_i Z_j exp(-i G r_ij)
                 2 V           G^2

        where the prime in \sum_G means G != 0.
        '''
        with torch.no_grad():
            nx, ny, nz = ((self._Gcut * self._eta / np.pi / torch.norm(self._Bcell, dim = 1)).to(torch.int32)) + 1
            Gn = torch.stack(torch.meshgrid(
                torch.arange(-nx.item(), nx.item() + 1, dtype = torch.int32, device = self.device),
                torch.arange(-ny.item(), ny.item() + 1, dtype = torch.int32, device = self.device),
                torch.arange(-nz.item(), nz.item() + 1, dtype = torch.int32, device = self.device)
                )).reshape((3, -1))

            # remove G = 0 term
            cut = torch.sum(torch.abs(Gn), dim = 0) != 0
            Gn  = Gn[:, cut]
            G2 = torch.norm(
                torch.tensordot(self._Bcell * 2 * torch.pi, Gn.float(), dims = ([0], [0])),
                dim = 0)**2
            expG2_invG2 = 4 * torch.pi * torch.exp(-G2 / 4 / self._eta**2) / G2

            ii, jj = torch.meshgrid(self._iframe, self._iads, indexing = 'ij')
            rij = self._scapos[ii] - self._scapos[jj]
            sfac = torch.exp(-2j * torch.pi * torch.matmul(rij, Gn.float()))
            Uij  = 0.5 * torch.sum(expG2_invG2 * sfac, dim = -1) / self._omega * 2.0

            return Uij.real


    def get_ewaldsum(self):
        '''
        Total Coulomb energy from Ewald summation.
        '''
        with torch.no_grad():
            # real-space contribution
            Ur = torch.sum(self.get_sum_real() * self._Zij.T)

            # reciprocal--space contribution
            Ug = torch.sum(self.get_sum_recp() * self._Zij)

            # total coulomb energy
            Ut = (Ur + Ug) * self._inv_4pi_epsilon0

            return Ut.item()
        