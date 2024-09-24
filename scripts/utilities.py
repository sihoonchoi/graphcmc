import numpy as np

def compute_supercell_size(cell, cutoff):
    a, b, c, alpha, beta, gamma = cell.cellpar()

    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    volume = np.linalg.det(cell)

    da = volume / (b * c * np.sin(alpha_rad))
    db = volume / (a * c * np.sin(beta_rad))
    dc = volume / (a * b * np.sin(gamma_rad))

    na = int(np.ceil(2 * cutoff / da))
    nb = int(np.ceil(2 * cutoff / db))
    nc = int(np.ceil(2 * cutoff / dc))

    return na, nb, nc

def _random_rotation(pos):
    # Translate to origin
    com = np.average(pos, axis = 0)
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
