#
# Imports
#
import numpy as np


#
# Activation functions
#
def sgm(x, derivative=False):
    if not derivative:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        y = sgm(x)
        return y * (1.0 - y)


def linear(x, derivative=False):
    if not derivative:
        return x
    else:
        return 1.0


def gaussian(x, derivative=False):
    if not derivative:
        return np.exp(-x ** 2)
    else:
        return -2 * x * np.exp(-x ** 2)


def tanh(x, derivative=False):
    if not derivative:
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x) ** 2


def rectifier(x, derivative=False):
    y = x.copy()
    if not derivative:
        y[y < 0] = 0
        return y
    else:
        y[y < 0] = 0
        y[y > 0] = 1
        return y


#
# Classes
#
class BackPropagationNetwork:
    """A back-propagation network"""

    #
    # Class methods
    #
    def __init__(self, layer_size, layer_functions=None):
        """Initialize the network"""

        self.layer_count = 0
        self.shape = None
        self.weights = []
        self.a_funcs = []

        # Layer info
        self.layer_count = len(layer_size) - 1

        if layer_functions is None:
            l_funcs = []
            for i in range(self.layer_count):
                if i == self.layer_count - 1:
                    l_funcs.append(linear)
                else:
                    l_funcs.append(sgm)
        else:
            if len(layer_size) != len(layer_functions):
                raise ValueError("Incompatible list of activation functions.")
            elif layer_functions[0] is not None:
                raise ValueError("Input layer cannot have an activation function.")
            else:
                l_funcs = layer_functions[1:]

        self.a_funcs = l_funcs

        # Data from last Run
        self._layer_input = []
        self._layer_output = []
        self._previous_weight_delta = []

        # Create the weight arrays
        for (l1, l2) in zip(layer_size[:-1], layer_size[1:]):
            self.weights.append(np.random.normal(scale=0.1, size=(l2, l1 + 1)))
            self._previous_weight_delta.append(np.zeros((l2, l1 + 1)))

    #
    # Run method
    #
    def Run(self, input):
        """Run the network based on the input data"""

        n_cases = input.shape[0]

        # Clear out the previous intermediate value lists
        self._layer_input = []
        self._layer_output = []

        # Run it!
        for index in range(self.layer_count):
            # Determine layer input
            if index == 0:
                layer_input = np.dot(self.weights[0], (np.vstack([input.T, np.ones([1, n_cases])])))
            else:
                layer_input = np.dot(self.weights[index], (np.vstack([self._layer_output[-1], np.ones([1, n_cases])])))

            self._layer_input.append(layer_input)
            self._layer_output.append(self.a_funcs[index](layer_input))

        return self._layer_output[-1].T

    #
    # TrainEpoch method
    #
    def TrainEpoch(self, input, target, training_rate=0.2, momentum=0.0):
        """This method trains the network for one epoch"""

        delta = []
        n_cases = input.shape[0]

        # First run the network
        self.Run(input)

        # Calculate the  deltas
        for index in reversed(range(self.layer_count)):
            if index == self.layer_count - 1:
                # Compare to the target values
                output_delta = self._layer_output[index] - target.T
                error = np.sum(output_delta ** 2)
                delta.append(output_delta * self.a_funcs[index](self._layer_input[index], True))
            else:
                # Compare to the following layer's delta
                delta_pullback = np.dot(self.weights[index + 1].T, (delta[-1]))
                delta.append(delta_pullback[:-1, :] * self.a_funcs[index](self._layer_input[index], True))

        # Compute weight deltas
        for index in range(self.layer_count):
            delta_index = self.layer_count - 1 - index

            if index == 0:
                layer_output = np.vstack([input.T, np.ones([1, n_cases])])
            else:
                layer_output = np.vstack(
                    [self._layer_output[index - 1], np.ones([1, self._layer_output[index - 1].shape[1]])])

            curr_weight_delta = np.sum(
                layer_output[None, :, :].transpose(2, 0, 1) *
                delta[delta_index][None, :, :].transpose(2, 1, 0), axis=0)

            weight_delta = training_rate * curr_weight_delta + momentum * self._previous_weight_delta[index]

            self.weights[index] -= weight_delta

            self._previous_weight_delta[index] = weight_delta

        return error

#
# If run as a script, create a test object
#
if __name__ == "__main__":

    lvInput = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    lvTarget = np.array([[0.0], [0.0], [1.0], [1.0]])
    lFuncs = [None, sgm, linear]

    bpn = BackPropagationNetwork((2, 2, 1), lFuncs)

    lnMax = 50000
    lnErr = 1e-6
    for i in range(lnMax + 1):
        err = bpn.TrainEpoch(lvInput, lvTarget, momentum=0.5)
        if i % 5000 == 0 and i > 0:
            print("Iteration {0:6d}K - Error: {1:0.6f}".format(int(i / 1000), err))
        if err <= lnErr:
            print("Desired error reached. Iter: {0}".format(i))
            break

    # Display output
    lvOutput = bpn.Run(lvInput)
    for i in range(lvInput.shape[0]):
        print("Input: {0} Output: {1}".format(lvInput[i], lvOutput[i]))

        
