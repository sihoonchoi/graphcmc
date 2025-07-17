import numpy as np

class Ewald(object):
    def __init__(self, atoms, Z, frame_idx, ads_idx,
            eta: float = None,
            Rcut: float = 4.0,
            Gcut: float = 4.0):
        
        from graphcmc.utils.constants import FELECT
        
        self._atoms  = atoms
        self._na = len(self._atoms)
        self._iframe = frame_idx
        self._iads = ads_idx
        self._scapos = self._atoms.get_scaled_positions()

        self._Acell = self._atoms.cell[:]             # real-space cell
        self._Bcell = np.linalg.inv(self._Acell).T    # reciprocal-space cell
        self._omega = np.linalg.det(self._Acell)      # volume of real-space cell

        self._ZZ = Z
        ZZ_frame_mesh, ZZ_ads_mesh = np.meshgrid(self._ZZ[self._iframe], self._ZZ[self._iads], indexing = 'ij')
        self._Zij = ZZ_frame_mesh * ZZ_ads_mesh

        self._inv_4pi_epsilon0 = FELECT

        if eta is None:
            self._eta = np.sqrt(np.pi) / self._omega**(1./3)
        else:
            self._eta = np.array(eta)

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
        from scipy.special import erfc

        ii, jj = np.meshgrid(self._iframe, self._iads, indexing = 'ij')
        rij = self._scapos[ii] - self._scapos[jj]

        # move rij to the range [-0.5, 0.5]
        rij = np.where(rij >= 0.5, rij - 1.0, rij)
        rij = np.where(rij < -0.5, rij + 1.0, rij)
    
        ############################################################
        # contribution from N = 0 cell
        ############################################################
        rij0 = np.linalg.norm(
            np.tensordot(self._Acell, np.transpose(rij, (2, 1, 0)), axes = ([0], [0])),
            axis = 0)
        Uij = erfc(rij0 * self._eta) / rij0

        ############################################################
        # contribution from N != 0 cells
        ############################################################
        rij = np.transpose(rij, (2, 1, 0)).reshape(3, -1)
        nx, ny, nz = ((self._Rcut / self._eta / np.linalg.norm(self._Acell, axis = 1)).astype(np.int64)) + 1
        
        Rn = np.stack(np.meshgrid(
            np.arange(-nx.item(), nx.item() + 1),
            np.arange(-ny.item(), ny.item() + 1),
            np.arange(-nz.item(), nz.item() + 1), indexing = 'ij'
        )).reshape(3, -1)
    
        # remove N = 0 term
        cut = np.sum(np.abs(Rn), axis = 0) != 0
        Rn  = Rn[:, cut]

        # R_N + rij
        Rr = np.linalg.norm(
            np.tensordot(self._Acell, Rn[:, None, :] + rij[:, :, None], axes = ([0], [0])),
            axis = 0)

        Uij += np.sum(
            erfc(self._eta * Rr) / Rr, axis = 1
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
        nx, ny, nz = ((self._Gcut * self._eta / np.pi / np.linalg.norm(self._Bcell, axis = 1)).astype(np.int64)) + 1
        Gn = np.stack(np.meshgrid(
            np.arange(-nx.item(), nx.item() + 1),
            np.arange(-ny.item(), ny.item() + 1),
            np.arange(-nz.item(), nz.item() + 1), indexing = 'ij'
            )).reshape((3, -1))

        # remove G = 0 term
        cut = np.sum(np.abs(Gn), axis = 0) != 0
        Gn  = Gn[:, cut]
        G2 = np.linalg.norm(
            np.tensordot(self._Bcell * 2 * np.pi, Gn, axes = ([0], [0])),
            axis = 0)**2
        expG2_invG2 = 4 * np.pi * np.exp(-G2 / 4 / self._eta**2) / G2

        ii, jj = np.meshgrid(self._iframe, self._iads, indexing = 'ij')
        rij = self._scapos[ii] - self._scapos[jj]
        sfac = np.exp(-2j * np.pi * np.matmul(rij, Gn))
        Uij  = 0.5 * np.sum(expG2_invG2 * sfac, axis = -1) / self._omega * 2.0

        return Uij.real

    def get_ewaldsum(self):
        '''
        Total Coulomb energy from Ewald summation.
        '''
        # real-space contribution
        Ur = np.sum(self.get_sum_real() * self._Zij.T)

        # reciprocal--space contribution
        Ug = np.sum(self.get_sum_recp() * self._Zij)

        # total coulomb energy
        Ut = (Ur + Ug) * self._inv_4pi_epsilon0

        return Ut.item()
        