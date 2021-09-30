import gym
import pygame
import argparse
import numpy as np
import sys

from gym_brt.envs import QubeSwingupEnv
from gym_brt.control import pd_control_policy, flip_and_hold_policy
from rl_controller import rl_controller
from brain_controller import brain_controller as brain_controller_fn


# All of these are in the range (-1.0, +1.0)
# Thumbs directions are top and right are positive directions
# Triggers are -1.0 when unpressed, +1.0 when fully pressed
AXIS = {
    "left-thumb-x": 0,
    "left-thumb-y": 1,
    "left-trigger": 2,
    "right-thumb-x": 3,
    "right-thumb-y": 4,
    "right-trigger": 5,
}
BUTTON = {"A": 0, "B": 1, "X": 2, "Y": 3}

STATES = {"manual": 0, "brain": 1, "rl": 2, "flip": 4}
LED = {
    "manual": [1, 0, 0],  # Red
    "brain": [1, 1, 0],  # Yellow
    "flip": [0, 1, 0],  # White
    "rl": [0, 0, 1],  # Blue
}


class QubeSwingupLEDEnv(QubeSwingupEnv):
    """A swingup environment that supports changing the LEDs for the 3 states"""

    def __init__(self, **kwargs):
        super(QubeSwingupEnv, self).__init__(**kwargs)
        self.led_state = LED["manual"]

    def set_led_state(self, s):
        self.led_state = LED[s]

    def _led(self):
        is_upright = np.abs(self._alpha) < (10 * np.pi / 180)
        if is_upright:
            return [1, 1, 1]
        else:
            return self.led_state


def run(use_simulator=False, port=5000, frequency=250):
    # Connect to the xbox controller ===========================================
    pygame.init()
    pygame.joystick.init()
    clock = pygame.time.Clock()
    joysticks = []
    # for all the connected joysticks
    try:
        for i in range(0, pygame.joystick.get_count()):
            # create an Joystick object in our list
            joysticks.append(pygame.joystick.Joystick(i))
            # initialize them all (-1 means loop forever)
            joysticks[-1].init()
            # print a statement telling what the name of the controller is
            print("Detected joystick '", joysticks[-1].get_name(), "'")
        joystick = joysticks[-1]
    except:
        print("Joystick not detected\n")
        sys.exit(-1)

    # Init the brain controller
    brain_controller = brain_controller_fn(port=port)

    # Open the Qube Environment ================================================
    # with QubeSwingupLEDEnv(use_simulator=use_simulator) as env:
    env = QubeSwingupLEDEnv(use_simulator=use_simulator, frequency=frequency)
    try:
        state = env.reset()
        state, reward, done, info = env.step(np.array([0], dtype=np.float64))
        action = 0.0
        axis = 0.0

        # Start off in the manual setting
        game_state = STATES["manual"]

        while True:

            # Get the actions from the xbox controller =========================
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    if event.axis == AXIS["left-thumb-x"]:
                        axis = joystick.get_axis(AXIS["left-thumb-x"])
                    elif event.axis == AXIS["right-thumb-x"]:
                        axis = joystick.get_axis(AXIS["right-thumb-x"])

                if event.type == pygame.JOYBUTTONDOWN:
                    if joystick.get_button(BUTTON["Y"]):  # brain balance mode
                        if game_state == STATES["manual"]:
                            game_state = STATES["brain"]
                            env.set_led_state("brain")

                        elif game_state == STATES["brain"]:
                            game_state = STATES["manual"]
                            env.set_led_state("manual")

                        elif game_state == STATES["rl"]:
                            game_state = STATES["brain"]
                            env.set_led_state("brain")

                        elif game_state == STATES["flip"]:
                            game_state = STATES["brain"]
                            env.set_led_state("brain")

                    elif joystick.get_button(BUTTON["X"]):  # RL Mode
                        if game_state == STATES["manual"]:
                            game_state = STATES["rl"]
                            env.set_led_state("rl")

                        elif game_state == STATES["brain"]:
                            game_state = STATES["rl"]
                            env.set_led_state("rl")

                        elif game_state == STATES["rl"]:
                            game_state = STATES["manual"]
                            env.set_led_state("manual")

                        elif game_state == STATES["flip"]:
                            game_state = STATES["rl"]
                            env.set_led_state("rl")

                    elif joystick.get_button(BUTTON["A"]):  # Full classical flip up
                        if game_state == STATES["manual"]:
                            game_state = STATES["flip"]
                            env.set_led_state("flip")

                        elif game_state == STATES["brain"]:
                            game_state = STATES["flip"]
                            env.set_led_state("flip")

                        elif game_state == STATES["rl"]:
                            game_state = STATES["flip"]
                            env.set_led_state("flip")

                        elif game_state == STATES["flip"]:
                            game_state = STATES["manual"]
                            env.set_led_state("manual")

                    elif joystick.get_button(BUTTON["B"]):  # Switch to manual mode
                        game_state = STATES["manual"]
                        env.set_led_state("manual")

            # Do an action depending on your state =============================
            if game_state == STATES["manual"]:
                action = -3.0 * axis
            elif game_state == STATES["brain"]:
                action = brain_controller(state)
                if abs(state[1]) < (30 * np.pi / 180):
                    action = pd_control_policy(state)
            elif game_state == STATES["rl"]:
                action = rl_controller(state)
            elif game_state == STATES["flip"]:
                action = flip_and_hold_policy(state)

            # Run the action in the environment
            state, reward, done, info = env.step(action)
            if use_simulator:
                env.render()
    except:
        env.close()
        pass


def main():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--use_simulator", action="store_true")
    parser.add_argument("-p", "--port", default=5000, type=int)
    parser.add_argument("-f", "--frequency", default=250, type=int)
    args, _ = parser.parse_known_args()
    run(use_simulator=args.use_simulator, port=args.port, frequency=args.frequency)


if __name__ == "__main__":
    main()

