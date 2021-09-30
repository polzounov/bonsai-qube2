import numpy as np
import argparse
import requests


def brain_controller(port=5000, **kwargs):
    prediction_url = f"http://localhost:{port}/v1/prediction"

    def next_action(state, **kwargs):
        theta, alpha, theta_dot, alpha_dot = state

        observables = {
            "theta": theta,  # radians, motor angle
            "alpha": alpha,  # radians, pendulum angle
            "theta_dot": theta_dot,  # radians / s, motor angular velocity
            "alpha_dot": alpha_dot,  # radians / s, pendulum angular velocity
        }

        # Trap on GET failures so we can restart the brain without
        # bringing down this run loop. Plate will default to level
        # when it loses the connection.
        try:
            # Get action from brain
            response = requests.get(prediction_url, json=observables)
            info = {"status": response.status_code, "resp": response.json()}
            action_json = response.json()

            if response.ok:
                action_json = requests.get(prediction_url, json=observables).json()
                return np.array([action_json["Vm"]])

        except requests.exceptions.ConnectionError as e:
            print(f"No brain listening on port: {port}", file=sys.stderr)
            raise BrainNotFound

        except Exception as e:
            print(f"Brain exception: {e}")

        return np.array([0.0])

    return next_action


def main(port):
    import time

    brain = brain_controller(port=port)

    action = np.random.uniform(low=-1, high=1, size=(4,))
    # Run controller and time it
    t = time.time()
    brain(action)
    t2 = time.time() - t

    n = 100
    avg_time = t2
    min_time = t2
    max_time = t2
    idx = 1
    count = 2

    while True:
        action = np.random.uniform(low=-1, high=1, size=(4,))

        # Run controller and time it
        t = time.time()
        brain(action)
        t2 = time.time() - t

        if t2 < min_time:
            min_time = t2
        elif t2 > max_time:
            max_time = t2

        avg_time = t2 * (1 / count) + avg_time * ((count - 1) / count)

        count += 1
        idx = (idx + 1) % n
        if idx == 0:
            print(
                f"Avg time: {avg_time:.5f}s, "
                f"Min time: {min_time:.5f}s, "
                f"Max time: {max_time:.5f}s"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default=5000, type=int, help="Port for bonsai.")
    args, _ = parser.parse_known_args()

    main(port=args.port)

