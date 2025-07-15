import numpy as np
import os
import torch

from forcefield import forcefield

def load_atoms_and_charges(args):
    from ase.io import read

    mof_path = f'{args.home_dir}/data/mof/core_relaxed_ddec/{args.framework}.cif'
    if not os.path.isfile(mof_path):
        mof_path = f'{args.home_dir}/data/mof/numat_modified/{args.framework}.cif'

    atoms_frame = read(mof_path)
    atoms_ads = read(f'scripts/data/{args.adsorbate}.xyz')

    if not args.framework_charge_off:
        with open(mof_path, 'r') as f:
            lines = f.readlines()
        if ' _atom_site_charge\n' in lines:
            idx = lines.index(' _atom_site_charge\n') + 1
            charges = [float(line.strip().split()[-1]) for line in lines[idx:]]
            atoms_frame.set_initial_charges(charges)

        charge_map = {
            'co2': [0.70, -0.35, -0.35],
            'methane': [0.0],
            'h2o': [0.0, 0.241, 0.241, -0.241, -0.241],
        }
        if args.adsorbate in charge_map:
            atoms_ads.set_initial_charges(charge_map[args.adsorbate])

    return atoms_frame, atoms_ads

def initialize_forcefield(args, atoms_supercell, atoms_frame, atoms_ads, charge):
    from fairchem.core import pretrained_mlip, FAIRChemCalculator

    if args.FF == 'uff':
        return forcefield(
            atoms_supercell, atoms_frame, atoms_ads,
            hybrid = False,
            vdw_cutoff = args.vdw_cutoff,
            tail_correction = args.tail_correction,
            charge = charge
        )
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictor = pretrained_mlip.get_predict_unit(args.FF, device = device)
        mlff = FAIRChemCalculator(predictor, task_name = 'odac')
        return forcefield(
            atoms_supercell, atoms_frame, atoms_ads,
            hybrid = True,
            mlff = mlff,
            vdw_cutoff = args.vdw_cutoff,
            tail_correction = args.tail_correction,
            charge = charge
        )

def print_simulation_info(args):
    print(f"Framework: {args.framework}")
    print(f"Adsorbate: {args.adsorbate}")
    print(f"Initialization: {args.initialization_cycle}")
    print(f"Cycles: {args.cycle}")
    print(f"Framework charge: {not args.framework_charge_off}")
    print(f"Minimum inner steps: {args.minimum_inner_steps}")
    print(f"Temperature: {args.T} K")
    print(f"Pressure: {args.P} Pa")
    print(f"Fugacity: {args.fugacity:.5f}")
    print()

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
