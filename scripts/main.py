import warnings
import os

from gcmc import GCMC
from utilities import load_atoms_and_charges, compute_supercell_size, initialize_forcefield, print_simulation_info
from eos import PREOS

warnings.simplefilter('ignore')

def main(args):
    from ase.data import vdw_radii

    args.FF = args.FF.lower()
    args.adsorbate = args.adsorbate.lower()
    args.simulation_type = args.simulation_type.lower()

    atoms_frame, atoms_ads = load_atoms_and_charges(args)
    charge = atoms_ads.get_initial_charges().any() or atoms_frame.get_initial_charges().any()

    # C and O in CO2 were renamed to Cs and Os to differentiate them from framework atoms
    # H, O, and massless charge points were renamed to Fr, At, and Pa to differentiate them from framework atoms

    # Expand the unit cell based on the vdW cutoff
    x, y, z = compute_supercell_size(atoms_frame.cell, args.vdw_cutoff)
    atoms_supercell = atoms_frame.copy() * (x, y, z)

    args.fugacity = PREOS.from_name(args.adsorbate).calculate_fugacity(args.T, args.P)
    
    print_simulation_info(args)

    ff = initialize_forcefield(args, atoms_supercell, atoms_frame, atoms_ads, charge)
    gcmc = GCMC(args, ff, atoms_supercell, atoms_ads, vdw_radii)

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
    parser.add_argument("--FF", required = True, type = str)
    parser.add_argument("--simulation-type", default = 'mc', type = str)
    parser.add_argument("--cycle", required = True, type = int)
    parser.add_argument("--initialization-cycle", default = 0, type = int)
    parser.add_argument("--framework", required = True, type = str)
    parser.add_argument("--adsorbate", required = True, type = str)
    parser.add_argument("--T", required = True, type = float)
    parser.add_argument("--P", required = True, type = float)
    parser.add_argument("--vdw-cutoff", default = 14.0, type = float)
    parser.add_argument("--tail-correction", action = 'store_true')
    parser.add_argument("--framework-charge-off", action = 'store_true')
    parser.add_argument("--print-every", default = 1000, type = int)
    parser.add_argument("--minimum-inner-steps", default = 20, type = int)
    parser.add_argument("--continue-sim", action = 'store_true')
    parser.add_argument("--home-dir", default = '.', type = str)
    parser.add_argument("--energy-shift", default = 0.0, type = float)

    args = parser.parse_args()
    main(args)
