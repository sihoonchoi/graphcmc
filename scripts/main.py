import warnings
import os

from ase.io import read
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from gcmc import GCMC
from utilities import compute_supercell_size
from eos import PREOS
from forcefield import forcefield

warnings.simplefilter('ignore')

def main(args):
    from ase.data import vdw_radii

    mof_path = f'{args.home_dir}/data/MOFs/core_relaxed_ddec/{args.framework}.cif'
    if not os.path.isfile(mof_path):
        mof_path = f'{args.home_dir}/data/MOFs/numat_modified/{args.framework}.cif'
    ads_path = f'scripts/data/{args.adsorbate}.xyz'
    atoms_frame = read(mof_path)
    atoms_ads = read(ads_path)
    if not args.framework_charge_off:
        with open(mof_path, 'r') as f:
            lines = f.readlines()
        charges_frame = []
        if ' _atom_site_charge\n' in lines:
            for i in range(lines.index(' _atom_site_charge\n') + 1, len(lines)):
                charges_frame.append(float(lines[i].strip().split()[-1]))
            atoms_frame.set_initial_charges(charges_frame)

        if args.adsorbate.casefold() == 'co2':
            atoms_ads.set_initial_charges([0.70, -0.35, -0.35])
        elif args.adsorbate.casefold() == 'methane':
            atoms_ads.set_initial_charges([0.0])
        elif args.adsorbate.casefold() == 'h2o':
            atoms_ads.set_initial_charges([0.0, 0.241, 0.241, -0.241, -0.241])
    charge = atoms_ads.get_initial_charges().any() or atoms_frame.get_initial_charges().any()

    # C and O in CO2 were renamed to Cs and Os to differentiate them from framework atoms
    # H, O, and massless charge points were renamed to Fr, At, and Pa to differentiate them from framework atoms

    # Expand the unit cell based on the vdW cutoff
    x, y, z = compute_supercell_size(atoms_frame.cell, args.vdw_cutoff)
    atoms_supercell = atoms_frame.copy() * (x, y, z)

    eos = PREOS.from_name(args.adsorbate)
    fugacity = eos.calculate_fugacity(args.T, args.P)
    
    print(f'FF: {args.FF.casefold()}')
    print(f'framework: {args.framework}')
    print(f'adsorbate: {args.adsorbate.casefold()}')
    print(f'initialization: {args.initialization_cycle}')
    print(f'cycles: {args.cycle}')
    print(f'framework charge: {not args.framework_charge_off}')
    print(f'minimum inner steps: {args.minimum_inner_steps}')
    print()
    print(f'temperature: {args.T} K')
    print(f'pressure: {args.P} Pa')
    print(f'fugacity: {fugacity:.5f}\n')

    if args.FF.casefold() == 'uff':
        ff = forcefield(atoms_supercell, atoms_frame, atoms_ads, hybrid = False, vdw_cutoff = args.vdw_cutoff, tail_correction = not args.tail_correction_off, charge = charge, device = args.device)
    else:
        config_path = f'{args.home_dir}/github/ocp/configs/odac/s2ef/{args.FF}.yml'
        checkpoint_path = f'{args.home_dir}/github/ocp/checkpoints/odac/s2ef/{args.FF}.pt'
        if args.device.casefold() == 'cpu':
            cpu = True
        else:
            cpu = False
        mlff = OCPCalculator(config_yml = config_path, checkpoint_path = checkpoint_path, cpu = cpu)
        ff = forcefield(atoms_supercell, atoms_frame, atoms_ads, hybrid = True, mlff = mlff, vdw_cutoff = args.vdw_cutoff, tail_correction = not args.tail_correction_off, charge = charge, device = args.device)

    gcmc = GCMC(args, ff, atoms_supercell, atoms_ads, fugacity, vdw_radii)
    if args.simulation_type.casefold() == 'mc':
        loading = gcmc.run(args.initialization_cycle, initialize = True)
        if args.cycle:
            loading = gcmc.run(args.cycle)

        print(f'loading: {(loading / x / y / z):.10f} molecule per unit cell')
    elif args.simulation_type.casefold() == 'widom':
        gcmc.widom(args.cycle)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation-type", default = 'mc', type = str)
    parser.add_argument("--cycle", required = True, type = int)
    parser.add_argument("--initialization-cycle", default = 0, type = int)
    parser.add_argument("--framework", required = True, type = str)
    parser.add_argument("--adsorbate", required = True, type = str)
    parser.add_argument("--T", required = True, type = float)
    parser.add_argument("--P", required = True, type = float)
    parser.add_argument("--FF", required = True, type = str)
    parser.add_argument("--vdw-cutoff", default = 14.0, type = float)
    parser.add_argument("--tail-correction-off", action = 'store_true')
    parser.add_argument("--framework-charge-off", action = 'store_true')
    parser.add_argument("--device", default = 'cpu', type = str)
    parser.add_argument("--print-every", default = 1000, type = int)
    parser.add_argument("--minimum-inner-steps", default = 20, type = int)
    parser.add_argument("--continue-sim", action = 'store_true')
    parser.add_argument("--home-dir", default = '.', type = str)
    parser.add_argument("--energy-shift", default = 0.0, type = float)

    args = parser.parse_args()
    main(args)
