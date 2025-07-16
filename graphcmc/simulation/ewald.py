import numpy as np
import torch



class Ewald(object):
    def __init__(self, atoms, Z, frame_idx, ads_idx, device = 'cpu',
            eta: float = None,
            Rcut: float = 4.0,
            Gcut: float = 4.0):
        
        from graphcmc.utils.constants import FELECT
        
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
        