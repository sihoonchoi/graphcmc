import numpy as np
import json
import torch

from ase import Atoms

#  - AUTOA     =  1. a.u. in Angstroem
#  - RYTOEV    =  1 Ry in Ev
#  - FELECT    =  (the electronic charge)/(4*pi*the permittivity of free space)
#                  in atomic units this is just e^2

AUTOA        = 0.529177249
RYTOEV       = 13.605826
FELECT       = 2 * AUTOA * RYTOEV

BOLTZMANN    = 1.380649e-23
PLANCK       = 1.0545718e-34
EV_TO_KJ_MOL = 96.48530749925793
MOL          = 6.02214076e23
J_TO_EV      = MOL / 1000.0 / EV_TO_KJ_MOL

def _random_rotation(pos):
    # Translate to origin
    com = np.average(pos, axis=0)
    pos -= com

    randnums = np.random.uniform(size = (3,))
    theta, phi, z = randnums

    theta = theta * 2.0 * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.

    R1 = np.sqrt(1 - z)
    R2 = np.sqrt(z)

    U0 = np.cos(phi) * R2
    U1 = np.sin(theta) * R1
    U2 = np.cos(theta) * R1
    U3 = np.sin(phi) * R2
    coefI = 2.0 * U0**2 - 1.0
    M = np.array([[coefI + 2.0 * U1**2, 2.0 * U1 * U2 - 2.0 * U0 * U3, 2.0 * U1 * U3 + 2.0 * U0 * U2],
                  [2.0 * U1 * U2 + 2.0 * U0 * U3, coefI + 2.0 * U2**2, 2.0 * U2 * U3 - 2.0 * U0 * U1],
                  [2.0 * U3 * U1 - 2.0 * U0 * U2, 2 * U3 * U2 + 2.0 * U0 * U1, coefI + 2.0 * U3**2]])
    pos = np.einsum('ib,ab->ia', pos, M)
    return pos + com

def _random_translation(pos, rvecs):
    pos -= np.average(pos, axis = 0)
    rnd = np.random.rand(3)
    new_cos = rnd[0]*rvecs[0] + rnd[1]*rvecs[1] + rnd[2]*rvecs[2]
    return pos + new_cos

def _random_position(pos, rvecs):
    pos = _random_rotation(pos)
    pos = _random_translation(pos, rvecs)
    return pos

def vdw_overlap(atoms, vdw, n_frame, n_ads, select_ads):
    nat = len(atoms)
    pos, numbers = atoms.get_positions(), atoms.get_atomic_numbers()
    for i_ads in range(n_frame + n_ads * select_ads, n_frame + n_ads * (select_ads + 1)):
        dists = atoms.get_distances(i_ads, np.arange(nat), mic = True)
        for i, d in enumerate(dists):
            if i >= n_frame + n_ads * select_ads and i < n_frame + n_ads * (select_ads + 1):
                continue
            if d < vdw[numbers[i_ads]] + vdw[numbers[i]]:
                return True
    return False


class EOS(object):
    def __init__(self, mass = 0.0):
        self.mass = mass

    def calculate_fugacity(self, T, P):
        """
           Evaluate the excess chemical potential at given external conditions
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Returns:**
           f
                The fugacity
        """
        mu, Pref = self.calculate_mu_ex(T, P)
        fugacity = np.exp(mu / (BOLTZMANN * T)) * Pref
        return fugacity

    def calculate_mu(self, T, P):
        """
           Evaluate the chemical potential at given external conditions
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Returns:**
           mu
                The chemical potential
        """
        # Excess part
        mu, Pref = self.calculate_mu_ex(T, P)
        # Ideal gas contribution to chemical potential
        assert self.mass != 0.0
        lambd = 2.0 * np.pi * self.mass * BOLTZMANN * T / PLANCK**2
        mu0 = -BOLTZMANN * T * np.log(BOLTZMANN * T / Pref * lambd**1.5)
        return mu0 + mu

    def get_Pref(self, T, P0, deviation = 1e-3):
        """
           Find a reference pressure at the given temperature for which the
           fluidum is nearly ideal.
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Optional arguments:**
           deviation
                When the compressibility factor Z deviates less than this from
                1, ideal gas behavior is assumed.
        """
        Pref = P0
        for _ in range(100):
            rhoref = self.calculate_rho(T, Pref)
            Zref = Pref / rhoref / BOLTZMANN / T
            # Z close to 1.0 means ideal gas behavior
            if np.abs(Zref - 1.0) > deviation:
                Pref /= 2.0
            else: break
        if np.abs(Zref - 1.0) > deviation:
            raise ValueError("Failed to find pressure where the fluidum is ideal-gas like, check input parameters")
        return Pref


class PREOS(EOS):
    """The Peng-Robinson equation of state"""
    def __init__(self, Tc, Pc, omega, mass = 0.0, phase = "vapour"):
        """
           The Peng-Robinson EOS gives a relation between pressure, volume, and
           temperature with parameters based on the critical pressure, critical
           temperature and acentric factor.
           **Arguments:**
           Tc
                The critical temperature of the species
           Pc
                The critical pressure of the species
           omega
                The acentric factor of the species
           **Optional arguments:**
           mass
                The mass of one molecule of the species. Some properties can be
                computed without this, so it is an optional argument
           phase
                Either "vapour" or "liquid". If both phases coexist at certain
                conditions, properties for the selected phase will be reported.
        """
        self.Tc = Tc
        self.Pc = Pc
        self.omega = omega
        self.mass = mass
        self.phase = phase

        # Some parameters derived from the input parameters
        self.a = 0.457235 * self.Tc**2 / self.Pc
        self.b = 0.0777961 * self.Tc / self.Pc
        self.kappa = 0.37464 + 1.54226 * self.omega - 0.26992 * self.omega**2

    @classmethod
    def from_name(cls, compound):
        """
           Initialize a Peng-Robinson EOS based on the name of the compound.
           Only works if the given compound name is included in
           'scripts/data/critical_acentric.csv'
        """
        # Read the data file containing parameters for a number of selected compounds
        fn = 'scripts/data/critical_acentric.csv'
#        fn = pkg_resources.resource_filename(yaff.__name__, 'data/critical_acentric.csv')
        dtype=[('compound', 'S20'), ('mass', 'f8'), ('Tc', 'f8'), ('Pc', 'f8'), ('omega', 'f8')]
        data = np.genfromtxt(fn, dtype = dtype, delimiter = ',')
        # Select requested compound
        if not compound.encode('utf-8') in data['compound']:
            raise ValueError(f"Could not find data for {compound} in file {fn}")
        index = np.where(compound.encode('utf-8') == data['compound'])[0]
        assert index.shape[0] == 1
        mass = data['mass'][index[0]]
        Tc = data['Tc'][index[0]]
        Pc = data['Pc'][index[0]] * 1e6
        omega = data['omega'][index[0]]
        return cls(Tc, Pc, omega, mass = mass)

    def set_conditions(self, T, P):
        """
           Set the parameters that depend on T and P
           **Arguments:**
           T
                Temperature
           P
                Pressure
        """
        self.Tr = T / self.Tc  # reduced temperature
        self.alpha = (1 + self.kappa * (1 - np.sqrt(self.Tr)))**2
        self.A = self.a * self.alpha * P / T**2
        self.B = self.b * P / T

    def polynomial(self, Z):
        """
           Evaluate the polynomial form of the Peng-Robinson equation of state
           If returns zero, the point lies on the PR EOS curve
           **Arguments:**
           Z
                Compressibility factor
        """
        return Z**3 - (1 - self.B) * Z**2 + (self.A - 2*self.B - 3*self.B**2) * Z - (self.A * self.B - self.B**2 - self.B**3)

    def polynomial_roots(self):
        """
            Find the real roots of the polynomial form of the Peng-Robinson
            equation of state
        """
        a = - (1 - self.B)
        b = self.A - 2 * self.B - 3 * self.B**2
        c = - (self.A * self.B - self.B**2 - self.B**3)
        Q = (a**2 - 3 * b)/9
        R = (2 * a**3 - 9 * a * b + 27 * c) / 54
        M = R**2 - Q**3
        if M > 0:
            S = np.cbrt(-R + np.sqrt(M))
            T = np.cbrt(-R - np.sqrt(M))
            Z = S + T - a / 3

        else:
            theta = np.arccos(R / np.sqrt(Q**3))
            x1 = -2.0 * np.sqrt(Q) * np.cos(theta / 3) - a / 3
            x2 = -2.0 * np.sqrt(Q) * np.cos((theta + 2 * np.pi) / 3) - a / 3
            x3 = -2.0 * np.sqrt(Q) * np.cos((theta - 2 * np.pi) / 3) - a / 3
            solutions = np.array([x1, x2, x3])
            solutions = solutions[solutions > 0.0]
            if self.phase == 'vapour':
                Z = np.amax(solutions)
            elif self.phase == 'liquid':
                Z = np.amin(solutions)
            else: raise NotImplementedError
        return Z

    def calculate_rho(self, T, P):
        """
           Calculate the particle density at given external conditions
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Returns:**
           rho
                The particle density
        """
        self.set_conditions(T, P)
        Z = self.polynomial_roots()
        return P / Z / BOLTZMANN / T

    def calculate_mu_ex(self, T, P):
        """
           Evaluate the excess chemical potential at given external conditions
           **Arguments:**
           T
                Temperature
           P
                Pressure
           **Returns:**
           mu
                The excess chemical potential
           Pref
                The pressure at which the reference chemical potential was calculated
        """
        # Find a reference pressure at the given temperature for which the fluidum
        # is nearly ideal
        Pref = self.get_Pref(T, P)
        # Find compressibility factor using rho
        rho = self.calculate_rho(T, P)
        Z = P / rho / BOLTZMANN / T
        # Add contributions to chemical potential at requested pressure
        mu = Z - 1 - np.log(Z - self.B) - self.A / np.sqrt(8) / self.B * np.log((Z + (1 + np.sqrt(2)) * self.B) / (Z + (1 - np.sqrt(2)) * self.B))
        mu += np.log(P / Pref)
        mu *= T * BOLTZMANN
        return mu, Pref


class forcefield():
    def __init__(self, frame, unitcell, ads, hybrid = False, mlff = None, vdw_cutoff = 14.0, charge = True, device = 'cpu'):
        self.frame = frame
        self.unitcell = unitcell
        self.ads = ads
        self.n_frame_atoms = len(self.frame)
        self.n_ads_atoms = len(ads)
        self.hybrid = hybrid
        self.vdw_cutoff = vdw_cutoff
        self.charge = charge
        self.device = device

        if mlff is not None:
            self.model = mlff

        with open('scripts/data/lj_params.json', 'r') as f:
            self.params = json.load(f)

        self.ads_params = np.array([[self.params[s]['sigma'], self.params[s]['epsilon']] for s in self.ads.get_chemical_symbols()])

    def get_potential_energy(self, new_atoms, old_atoms, old_e, i_ads):
        if self.n_frame_atoms == len(new_atoms):
            return 0, 0, 0, 0
        
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

                # Insertion
                if len(old_atoms) < len(new_atoms):
                    ml = 0
                    vdw = 0
                    ewald = 0

                    if self.hybrid:
                        temp_ads = new_atoms[-(self.n_ads_atoms):].copy()
                        temp_ads.set_cell(self.unitcell.cell)
                        adjusted_pos = (temp_ads.get_scaled_positions() % 1) @ self.unitcell.cell
                        temp_ads = Atoms(['C', 'O', 'O'], positions = adjusted_pos, cell = self.unitcell.cell, pbc = [True, True, True])

                        temp_atoms = self.unitcell.copy() + temp_ads
                        temp_atoms.calc = self.model
                        ml += temp_atoms.get_potential_energy() / J_TO_EV

                        if self.n_frame_atoms == len(old_atoms):
                            return old_e + ml + vdw + ewald, ml, vdw, ewald

                    wo_ads_idx = torch.tensor(np.arange(len(old_atoms))[start_idx:], dtype = torch.int32, device = self.device)
                    for i in range(self.n_ads_atoms):
                        ads_idx = self.n_frame_atoms + i_ads * self.n_ads_atoms + i
                        
                        dist = new_atoms.get_distances(ads_idx, wo_ads_idx.cpu().numpy(), mic = True, vector = False)
                        dist = torch.from_numpy(dist).float().to(self.device)
                        atoms_idx = wo_ads_idx[dist < self.vdw_cutoff]
                        dist = dist[dist < self.vdw_cutoff]

                        if dist.shape[0]:
                            params = torch.tensor([[self.params[o]['sigma'], self.params[o]['epsilon']] for o in old_chemical_symbols[atoms_idx.cpu()]], dtype = torch.float32, device = self.device)
                            mixing_sigma = (params[:, 0] + self.ads_params[i][0]) / 2
                            mixing_epsilon = torch.sqrt(params[:, 1] * self.ads_params[i][1])

                            vdw += (4 * mixing_epsilon * ((mixing_sigma / dist).pow(12) - (mixing_sigma / dist).pow(6))).sum().item() * BOLTZMANN

                    if self.charge:
                        ewald += ewaldsum(new_atoms, new_initial_charges, wo_ads_idx, torch.tensor([self.n_frame_atoms + i_ads * self.n_ads_atoms + i for i in range(self.n_ads_atoms)], dtype = torch.int32, device = self.device), device = self.device).get_ewaldsum() / J_TO_EV

                    return old_e + ml + vdw + ewald, ml, vdw, ewald

                # Deletion
                elif len(old_atoms) > len(new_atoms):
                    if len(new_atoms) == self.n_frame_atoms:
                        return 0, 0, 0, 0
                    
                    ml = 0
                    vdw = 0
                    ewald = 0

                    if self.hybrid:
                        temp_ads = old_atoms[(self.n_frame_atoms + i_ads * self.n_ads_atoms):(self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms)].copy()
                        temp_ads.set_cell(self.unitcell.cell)
                        adjusted_pos = (temp_ads.get_scaled_positions() % 1) @ self.unitcell.cell
                        temp_ads = Atoms(['C', 'O', 'O'], positions = adjusted_pos, cell = self.unitcell.cell, pbc = [True, True, True])

                        temp_atoms = self.unitcell.copy() + temp_ads
                        temp_atoms.calc = self.model
                        ml -= temp_atoms.get_potential_energy() / J_TO_EV

                    wo_ads_idx = torch.tensor(list(np.arange(len(old_atoms))[start_idx:self.n_frame_atoms + i_ads * self.n_ads_atoms]) + list(np.arange(len(old_atoms))[self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms:]), dtype = torch.int32, device = self.device)
                    for i in range(self.n_ads_atoms):
                        ads_idx = self.n_frame_atoms + i_ads * self.n_ads_atoms + i
                        
                        dist = old_atoms.get_distances(ads_idx, wo_ads_idx.cpu().numpy(), mic = True, vector = False)
                        dist = torch.from_numpy(dist).float().to(self.device)
                        atoms_idx = wo_ads_idx[dist < self.vdw_cutoff]
                        dist = dist[dist < self.vdw_cutoff]
                        
                        if dist.shape[0]:
                            params = torch.tensor([[self.params[o]['sigma'], self.params[o]['epsilon']] for o in old_chemical_symbols[atoms_idx.cpu()]], dtype = torch.float32, device = self.device)
                            mixing_sigma = (params[:, 0] + self.ads_params[i][0]) / 2
                            mixing_epsilon = torch.sqrt(params[:, 1] * self.ads_params[i][1])

                            vdw -= (4 * mixing_epsilon * ((mixing_sigma / dist).pow(12) - (mixing_sigma / dist).pow(6))).sum().item() * BOLTZMANN

                    if self.charge:
                        ewald -= ewaldsum(old_atoms, old_initial_charges, wo_ads_idx, torch.tensor([self.n_frame_atoms + i_ads * self.n_ads_atoms + i for i in range(self.n_ads_atoms)], dtype = torch.int32, device = self.device), device = self.device).get_ewaldsum() / J_TO_EV

                    return old_e + ml + vdw + ewald, ml, vdw, ewald
                    
                # Rotation or translation
                else:
                    ml = 0
                    vdw = 0
                    ewald = 0

                    if self.hybrid:
                        temp_ads_delete = old_atoms[(self.n_frame_atoms + i_ads * self.n_ads_atoms):(self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms)].copy()
                        temp_ads_delete.set_cell(self.unitcell.cell)
                        adjusted_pos = (temp_ads_delete.get_scaled_positions() % 1) @ self.unitcell.cell
                        temp_ads_delete = Atoms(['C', 'O', 'O'], positions = adjusted_pos, cell = self.unitcell.cell, pbc = [True, True, True])

                        temp_atoms = self.unitcell.copy() + temp_ads_delete
                        temp_atoms.calc = self.model
                        ml -= temp_atoms.get_potential_energy() / J_TO_EV

                        temp_ads_insert = new_atoms[(self.n_frame_atoms + i_ads * self.n_ads_atoms):(self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms)].copy()
                        temp_ads_insert.set_cell(self.unitcell.cell)
                        adjusted_pos = (temp_ads_insert.get_scaled_positions() % 1) @ self.unitcell.cell
                        temp_ads_insert = Atoms(['C', 'O', 'O'], positions = adjusted_pos, cell = self.unitcell.cell, pbc = [True, True, True])
                        
                        temp_atoms = self.unitcell.copy() + temp_ads_insert
                        temp_atoms.calc = self.model
                        ml += temp_atoms.get_potential_energy() / J_TO_EV

                        if self.n_frame_atoms + self.n_ads_atoms == len(new_atoms):
                            return old_e + ml + vdw + ewald, ml, vdw, ewald

                    wo_ads_idx = torch.tensor(list(np.arange(len(old_atoms))[start_idx:self.n_frame_atoms + i_ads * self.n_ads_atoms]) + list(np.arange(len(old_atoms))[self.n_frame_atoms + (i_ads + 1) * self.n_ads_atoms:]), dtype = torch.int32, device = self.device)
                    for i in range(self.n_ads_atoms):
                        ads_idx = self.n_frame_atoms + i_ads * self.n_ads_atoms + i
                        
                        old_dist = old_atoms.get_distances(ads_idx, wo_ads_idx.cpu().numpy(), mic = True, vector = False)
                        old_dist = torch.from_numpy(old_dist).float().to(self.device)
                        old_atoms_idx = wo_ads_idx[old_dist < self.vdw_cutoff]
                        old_dist = old_dist[old_dist < self.vdw_cutoff]

                        if old_dist.shape[0]:
                            params = torch.tensor([[self.params[o]['sigma'], self.params[o]['epsilon']] for o in old_chemical_symbols[old_atoms_idx.cpu()]], dtype = torch.float32, device = self.device)
                            mixing_sigma = (params[:, 0] + self.ads_params[i][0]) / 2
                            mixing_epsilon = torch.sqrt(params[:, 1] * self.ads_params[i][1])

                            vdw -= (4 * mixing_epsilon * ((mixing_sigma / old_dist).pow(12) - (mixing_sigma / old_dist).pow(6))).sum().item() * BOLTZMANN

                        new_dist = new_atoms.get_distances(ads_idx, wo_ads_idx.cpu().numpy(), mic = True, vector = False)
                        new_dist = torch.from_numpy(new_dist).float().to(self.device)
                        new_atoms_idx = wo_ads_idx[new_dist < self.vdw_cutoff]
                        new_dist = new_dist[new_dist < self.vdw_cutoff]

                        if new_dist.shape[0]:
                            params = torch.tensor([[self.params[o]['sigma'], self.params[o]['epsilon']] for o in new_chemical_symbols[new_atoms_idx.cpu()]], dtype = torch.float32, device = self.device)
                            mixing_sigma = (params[:, 0] + self.ads_params[i][0]) / 2
                            mixing_epsilon = torch.sqrt(params[:, 1] * self.ads_params[i][1])

                            vdw += (4 * mixing_epsilon * ((mixing_sigma / new_dist).pow(12) - (mixing_sigma / new_dist).pow(6))).sum().item() * BOLTZMANN

                    if self.charge:
                        ewald -= ewaldsum(old_atoms, old_initial_charges, wo_ads_idx, torch.tensor([self.n_frame_atoms + i_ads * self.n_ads_atoms + i for i in range(self.n_ads_atoms)], dtype = torch.int32, device = self.device), device = self.device).get_ewaldsum() / J_TO_EV
                        ewald += ewaldsum(new_atoms, new_initial_charges, wo_ads_idx, torch.tensor([self.n_frame_atoms + i_ads * self.n_ads_atoms + i for i in range(self.n_ads_atoms)], dtype = torch.int32, device = self.device), device = self.device).get_ewaldsum() / J_TO_EV

                    return old_e + ml + vdw + ewald, ml, vdw, ewald


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

            # interaction between charges
            # Us = -self._eta / np.sqrt(np.pi) * np.sum(self._ZZ**2)
            # interaction with the neutralizing background
            # Un = -(2*np.pi / self._eta**2 / self._omega) * self._ZZ.sum()**2 / 4

            # total coulomb energy
            Ut = (Ur + Ug) * self._inv_4pi_epsilon0
            # Ut = (Ur + Ug + Us + Un)*self._inv_4pi_epsilon0

            return Ut.item()
