from os import name
from typing import Tuple, Union, List, Dict
from pathlib import Path
import argparse

import numpy as np
from numba import jit
from tqdm import trange

# parser
parser = argparse.ArgumentParser()

parser.add_argument("--beta", type=float, help="Inverse temperature")
parser.add_argument("--num_mc_steps", type=int, help="Montecarlo steps")
parser.add_argument("--sample_path", type=str, help="Path to the saved proposals")
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="Seed to generate couplings (default: 12345)",
)
parser.add_argument(
    "--save",
    dest="save",
    action="store_true",
    help="Flag if you want to save samples after MCMC",
)
parser.add_argument(
    "--verbose",
    dest="verbose",
    action="store_true",
    help="Flag if you want to see prints in MCMC",
)


@jit(nopython=True)
def compute_prob(eng: float, beta: float, num_spin: int) -> float:
    """Boltzmann probability distribution

    Args:
        eng (float): Energy of the sample.
        beta (float): Inverse temperature
        num_spin (int): Number of spins in the sample.

    Returns:
        float: Boltzmann probability.
    """
    return np.exp(-beta * num_spin * eng)


# TODO: Can be faster than this?
@jit(nopython=True)
def compute_eng(Lx, J, S0):
    energy = 0.0
    for i in range(Lx):
        for j in range(Lx):

            k = i + (Lx * j)

            S = S0[i, j]
            nb = S0[(i + 1) % Lx, j] * J[k, 0] + S0[i, (j + 1) % Lx] * J[k, 1]
            energy += -S * nb
    return energy / (Lx ** 2)


@jit(nopython=True)
def get_couplings(L: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    return np.random.normal(loc=0.0, scale=1.0, size=(L ** 2, 2))


def load_data(
    sample_path: Union[str, Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(sample_path, str):
        data = np.load(sample_path)
    elif isinstance(sample_path, Dict):
        data = sample_path
    else:
        raise ValueError("Neither a path or a Numpy dataset!")
    return data["sample"], data["prob"]


@jit(nopython=True)
def compute_avg_std(energies: np.ndarray) -> Tuple[float, float]:
    return np.mean(energies), np.std(energies)


def mcmc(
    beta: float,
    num_mc_steps: int,
    sample_path: Union[str, np.lib.npyio.NpzFile],
    seed: int = 12345,
    verbose: bool = False,
    save: bool = False,
) -> None:

    # load data generate by PixelCNN
    proposals, probs = load_data(sample_path)

    # get the first sample and its energy
    accepted_sample, accepted_prob = proposals[0], probs[0]

    # get the dimension of the sample from the
    L = accepted_sample.shape[-1]

    # get the coupling matrix
    J = get_couplings(L, seed)

    # initialisation
    energies = []
    samples = []
    transition_prob = []
    prob_ratio = []
    accepted = 0

    # compute the energy of the new configuration
    accepted_eng = compute_eng(L, J, accepted_sample)
    # compute boltzmann probability
    accepted_boltz_prob = compute_prob(accepted_eng, beta, L ** 2)

    # for idx in trange(num_mc_steps, leave=True):
    for idx in range(num_mc_steps):
        # get next sample and its energy
        trial_sample, trial_prob = proposals[idx + 1], probs[idx + 1]
        trial_eng = compute_eng(L, J, trial_sample)
        # compute Boltzmann probability
        trial_boltz_prob = compute_prob(trial_eng, beta, L ** 2)

        # get the transition probability
        prob_ratio.append(
            (accepted_prob / trial_prob) * (trial_boltz_prob / accepted_boltz_prob)
        )
        transition_prob.append(min(1.0, prob_ratio[idx]))

        if (
            transition_prob[idx] == 1.0
            or np.random.random_sample() < transition_prob[idx]
        ):
            # update energy, prob and sample
            accepted_eng = np.copy(trial_eng)
            accepted_prob = np.copy(trial_prob)
            accepted_sample = np.copy(trial_sample)
            accepted_boltz_prob = np.copy(trial_boltz_prob)
            accepted += 1

        # save acceped sample and its energy
        samples.append(accepted_sample)
        energies.append(accepted_eng)

        if verbose:
            # update mean and std of energies for print
            avg_eng, std_eng = compute_avg_std(np.asarray(energies))
            print(
                f"\n\nStep {idx+1}\nAccepted energy {accepted_eng}\nAverage energy {avg_eng}\nStd energy {std_eng}"
            )

    if save:
        avg_eng, std_eng = compute_avg_std(np.asarray(energies))
        # use path class
        sample_path = Path(sample_path)
        filename = (
            str(L ** 2)
            + "_lattice_2d_ising_spins_PIXELMCMC"
            + sample_path.parts[-1].split("-")[-1][:-4]
            + ".npy"
        )
        out = {
            "accepted": accepted,
            "avg_eng": avg_eng,
            "std_eng": std_eng,
            "samples": samples,
            "energies": energies,
        }
        np.savez(filename, out)

    print("\nAccepted proposals: {0} ({1})".format(accepted, accepted / num_mc_steps))


if __name__ == "__main__":
    args = parser.parse_args()

    mcmc(
        args.beta,
        args.num_mc_steps,
        args.sample_path,
        seed=args.seed,
        verbose=args.verbose,
        save=args.save,
    )

