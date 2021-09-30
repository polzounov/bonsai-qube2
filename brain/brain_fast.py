import tflite_runtime.interpreter as tflite
import numpy as np
import argparse
import time


interpreter = tflite.Interpreter(model_path="saved_model.tflite")
# Figure out the index out the `input` tensor which is what we need to update
# with our current state
input_tensor_index = interpreter.get_input_details()[1]["index"]
output_tensor_index = interpreter.get_output_details()[1]["index"]
interpreter.allocate_tensors()

print("input_tensor:", interpreter.get_tensor(input_tensor_index))
print("output_tensor:", interpreter.get_tensor(output_tensor_index))


# def brain_fast_controller(state):
def b(state):
    # alpha, theta, alpha_dot, theta_dot = state
    # Must be (1,4)
    # state_tensor = np.array([alpha, theta, alpha_dot, theta_dot], dtype=np.float32)
    state_tensor = np.asarray(state, dtype=np.float32)
    state_tensor = state_tensor.reshape(1, -1)
    interpreter.set_tensor(input_tensor_index, state_tensor)
    interpreter.invoke()
    action = interpreter.get_tensor(output_tensor_index)
    return action


def one_hot(x, shape=interpreter.get_tensor(input_tensor_index).shape):
    oh = np.zeros(shape)
    oh[:, x] = 1.0
    return oh


# n = 1000
# ts = np.zeros((n,))
# rs = np.random.randn(n, 4)
# for i in range(n):
#     t = time.time()
#     b(rs[i, :])
#     ts[i] = time.time() - t

# b(one_hot(0))


def test_rl_controller(use_simulator):
    from gym_brt.envs import QubeSwingupEnv

    with QubeSwingupEnv(use_simulator=use_simulator, frequency=250) as env:
        while True:
            state = env.reset()
            state, _, done, _ = env.step(np.array([0]))

            while not done:
                action = rl_controller(state)
                state, _, done, _ = env.step(action)
                if use_simulator:
                    env.render()


if __name__ == "__main__":
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--use_simulator", action="store_true")
    args, _ = parser.parse_known_args()
    test_rl_controller(args.use_simulator)
