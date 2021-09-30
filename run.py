from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import argparse
import numpy as np

from brain_controller import brain_controller
from rl_controller import rl_controller, rl_lqr

from gym_brt.envs import (
    QubeSwingupEnv,
    QubeSwingupSparseEnv,
    QubeSwingupFollowEnv,
    QubeSwingupFollowSparseEnv,
    QubeBalanceEnv,
    QubeBalanceSparseEnv,
    QubeBalanceFollowEnv,
    QubeBalanceFollowSparseEnv,
    QubeDampenEnv,
    QubeDampenSparseEnv,
    QubeDampenFollowEnv,
    QubeDampenFollowSparseEnv,
    QubeRotorEnv,
    QubeRotorFollowEnv,
    QubeBalanceFollowSineWaveEnv,
    QubeSwingupFollowSineWaveEnv,
    QubeRotorFollowSineWaveEnv,
    QubeDampenFollowSineWaveEnv,
)

from gym_brt.control import (
    zero_policy,
    constant_policy,
    random_policy,
    square_wave_policy,
    energy_control_policy,
    pd_control_policy,
    flip_and_hold_policy,
    square_wave_flip_and_hold_policy,
    dampen_policy,
    pd_tracking_control_policy,
)


def print_info(state_info, action, reward):
    theta = state_info["theta"]
    alpha = state_info["alpha"]
    theta_dot = state_info["theta_dot"]
    alpha_dot = state_info["alpha_dot"]
    print(
        "State: theta={:06.3f}, alpha={:06.3f}, theta_dot={:06.3f}, alpha_dot={:06.3f}".format(
            theta, alpha, theta_dot, alpha_dot
        )
    )
    print("Action={}, Reward={}".format(action, reward))


def test_env(
    env_name,
    policy,
    frequency=250,
    state_keys=None,
    verbose=False,
    use_simulator=False,
    render=False,
    steps=100000,
    terminate=False,
):

    with env_name(use_simulator=use_simulator, frequency=frequency) as env:
        step = 0
        while True:
            state = env.reset()
            state, reward, done, info = env.step(np.array([0], dtype=np.float64))
            for _ in range(steps):
                action = policy(state, step=step, frequency=frequency)
                state, reward, done, info = env.step(action)
                if terminate and done:  # Restart the episode
                   break
                if verbose:
                    print_info(info, action, reward)
                if render:
                    env.render()

                step += 1


def main():
    envs = {
        "QubeSwingupEnv": QubeSwingupEnv,
        "QubeSwingupSparseEnv": QubeSwingupSparseEnv,
        "QubeSwingupFollowEnv": QubeSwingupFollowEnv,
        "QubeSwingupFollowSparseEnv": QubeSwingupFollowSparseEnv,
        "QubeBalanceEnv": QubeBalanceEnv,
        "QubeBalanceSparseEnv": QubeBalanceSparseEnv,
        "QubeBalanceFollowEnv": QubeBalanceFollowEnv,
        "QubeBalanceFollowSparseEnv": QubeBalanceFollowSparseEnv,
        "QubeDampenEnv": QubeDampenEnv,
        "QubeDampenSparseEnv": QubeDampenSparseEnv,
        "QubeDampenFollowEnv": QubeDampenFollowEnv,
        "QubeDampenFollowSparseEnv": QubeDampenFollowSparseEnv,
        "QubeRotorEnv": QubeRotorEnv,
        "QubeRotorFollowEnv": QubeRotorFollowEnv,
        "QubeBalanceFollowSineWaveEnv": QubeBalanceFollowSineWaveEnv,
        "QubeSwingupFollowSineWaveEnv": QubeSwingupFollowSineWaveEnv,
        "QubeRotorFollowSineWaveEnv": QubeRotorFollowSineWaveEnv,
        "QubeDampenFollowSineWaveEnv": QubeDampenFollowSineWaveEnv,
    }
    policies = {
        "none": zero_policy,
        "zero": zero_policy,
        "const": constant_policy,
        "rand": random_policy,
        "random": random_policy,
        "sw": square_wave_policy,
        "energy": energy_control_policy,
        "pd": pd_control_policy,
        "hold": pd_control_policy,
        "flip": flip_and_hold_policy,
        "sw-hold": square_wave_flip_and_hold_policy,
        "damp": dampen_policy,
        "track": pd_tracking_control_policy,
        "bonsai": None,  # Add a bit later
        "brain": None,  # Add a bit later
        "rl_baseline": rl_controller,
        "rl_lqr_baseline": rl_lqr,        
    }

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env",
        default="QubeSwingupEnv",
        choices=list(envs.keys()),
        help="Enviroment to run.",
    )
    parser.add_argument(
        "-c",
        "--controller",
        default="random",
        choices=list(policies.keys()),
        help="Select what type of action to take.",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        "--sample-frequency",
        default="250",
        type=float,
        help="The frequency of samples on the Quanser hardware.",
    )

    parser.add_argument("-p", "--port", default=5000, type=int, help="Port for bonsai.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-r", "--render", action="store_true")
    parser.add_argument("-s", "--use_simulator", action="store_true")
    parser.add_argument("-n", "--num_steps", default=1000000, type=int)
    parser.add_argument("-t", "--terminate", action="store_true", help="Allow episode termination when out of bounds.")
    args, _ = parser.parse_known_args()

    brain = brain_controller(port=args.port)
    policies["bonsai"] = brain
    policies["brain"] = brain

    test_env(
        envs[args.env],
        policies[args.controller],
        frequency=args.frequency,
        verbose=args.verbose,
        use_simulator=args.use_simulator,
        render=args.render,
        steps=args.num_steps,
        terminate=args.terminate,
        )


if __name__ == "__main__":
    main()
