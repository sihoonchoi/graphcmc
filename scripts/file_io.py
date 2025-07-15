import os
import numpy as np

def setup_result_directory(job_id, continue_sim):
    result_dir = os.path.join("results", job_id)
    if not continue_sim:
        if os.path.exists(result_dir):
            for f in os.listdir(result_dir):
                os.remove(os.path.join(result_dir, f))
        else:
            os.makedirs(result_dir)
    return result_dir

def save_checkpoint(result_dir, atoms, uptake, adsorption_energy, Z_ads, n_ads, iteration=None, initialize=False):
    tag = 'initialization_' if initialize else ''
    np.save(os.path.join(result_dir, f"{tag}uptake.npy"), np.array(uptake))
    np.save(os.path.join(result_dir, f"{tag}adsorption_energy.npy"), np.array(adsorption_energy))
    np.save(os.path.join(result_dir, "last_adsorbate_positions.npy"), atoms[-(Z_ads * n_ads):].get_positions())

    if iteration is not None and not initialize:
        np.save(os.path.join(result_dir, f"adsorbate_{iteration + 1:010d}.npy"), atoms[-(Z_ads * n_ads):].get_positions())

def load_simulation_state(result_dir, atoms_frame, atoms_ads, n_ads, initialize):
    tag = 'initialization_' if initialize else ''
    uptake = list(np.load(os.path.join(result_dir, f"{tag}uptake.npy")))
    adsorption_energy = list(np.load(os.path.join(result_dir, f"{tag}adsorption_energy.npy")))
    pos = np.load(os.path.join(result_dir, "last_adsorbate_positions.npy"))

    atoms = atoms_frame.copy()
    for i in range(uptake[-1]):
        temp_ads = atoms_ads.copy()
        temp_ads.set_positions(pos[i * n_ads:(i + 1) * n_ads])
        atoms += temp_ads

    return uptake, len(uptake), adsorption_energy, atoms

def load_stats(result_dir):
    def _load(filename):
        path = os.path.join(result_dir, filename)
        return list(np.load(path))

    accepted = _load("accepted.npy")
    attempted = _load("attempted.npy")

    return accepted, attempted

def save_stats(result_dir, accepted, attempted):
    np.save(os.path.join(result_dir, "accepted.npy"), np.array(accepted))
    np.save(os.path.join(result_dir, "attempted.npy"), np.array(attempted))
