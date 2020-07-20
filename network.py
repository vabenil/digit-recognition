from numpy import array, random, dot, exp, linalg, zeros
from typing import List


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


# Derivative of the sigomid function.
# x is the output of the sigmoid function. You can see it as x = sigmoid(y).
def _dvsigmoid(x: array) -> array:
    return x * (1 - x)


# Derivative of the sigmoid function
def dvsigmoid(x: array) -> array:
    return _dvsigmoid(sigmoid(x))


# Error or cost
def cost(x: array, y: array) -> array:
    return (linalg.norm(y - x) ** 2) / 2


def dvcost(x: array, y: array) -> array:
    return (x - y)


class Network:
    def __init__(self, dims: List[int]):
        random.seed(5)

        self.n_layers = len(dims)
        self.dims = dims
        # randomly initialize biases
        self.B = [2 * random.random((j, 1)) - 1 for j in dims[1:]]
        # randomly initialize weights
        self.W = [
            2 * random.random((j, i)) - 1
            for j, i in zip(dims[1:], dims[:-1])
        ]
        self.lr = 1

    # Return the ouput of the network given x input
    def feedforward(self, x: array) -> List[array]:
        L = [0] * self.n_layers
        L[0] = x
        for i in range(self.n_layers - 1):
            w, b, a = self.W[i], self.B[i], L[i]
            L[i + 1] = sigmoid(dot(w, a) + b)
        return L

    def backprop(self, tx: array, ty: array) -> tuple:
        # Initialize empty lists of n_layers - 1 size
        db = [0] * (self.n_layers - 1)
        dw = [0] * (self.n_layers - 1)

        # Output of each layer
        L = self.feedforward(tx)

        # Direction to move to reduce the cost times
        # the rate of change of the sigmoid function at a certain layer
        l_delta = dvcost(L[-1], ty) * _dvsigmoid(L[-1])

        db[-1] = l_delta
        dw[-1] = dot(l_delta, L[-2].T)

        # Same as - for i in reversed(range(1, self.num_layers)):
        for i in range(self.n_layers - 2, 0, -1):
            # How sensitive the function is to previous weights
            l_delta = dot(self.W[i].T, l_delta) * _dvsigmoid(L[i])

            db[i - 1] = l_delta
            dw[i - 1] = dot(l_delta, L[i - 1].T)

        return dw, db

    """
    Update weights and biases given a batch
    """
    def update_batch(self, batch, batch_size) -> None:
        sdW = [zeros(w.shape) for w in self.W]
        sdB = [zeros(b.shape) for b in self.B]

        """
        sdWB = array([self.backprop(x, y) for x, y in batch])
        """
        # Sum of all the delta weights and biases calculated from the batch
        for tx, ty in batch:
            dW, dB = self.backprop(tx, ty)

            for i in range(self.n_layers - 1):
                sdW[i] += dW[i]
                sdB[i] += dB[i]

        temp_calc = float(self.lr) / float(batch_size)
        # basically -learning rate * mean(delta weight) or
        # W[i] = W[i] - (learning_rate * sum_delta_weights) / n
        for i in range(self.n_layers - 1):
            self.W[i] += -temp_calc * sdW[i]
            self.B[i] += -temp_calc * sdB[i]

    """
    Train the neural network using mini-batch stochastic gradient descent.
    training_data -> List of tuples (x, y)
        x being the training inputs and
        y being the expect outputs
    """
    def SGD(self,
            tdata,
            epochs: int,
            batch_size: int,
            batch_n: int,
            learning_rate: float):
        self.lr = learning_rate

        for j in range(epochs):
            random.shuffle(tdata)

            # array of batch_n x batch_size shape
            batches = array([
                tdata[k:k+batch_size]
                for k in range(0, batch_n * batch_size, batch_size)
             ])

            for batch in batches:
                self.update_batch(batch, batch_size)

    def accuracy(self, test_data):
        n = len(test_data)
        correct = 0
        for i in range(n):
            tx, ty = test_data[i]

            a = self.feedforward(tx)[-1]
            correct += (a.argmax() == ty.argmax()).all()

        return correct / n


if __name__ == '__main__':
    net_dims = [3, 4, 1]
    net = Network(net_dims)

    tX = [array([[0], [0], [1]]),
          array([[0], [1], [1]]),
          array([[1], [0], [1]]),
          array([[0], [1], [0]]),
          array([[1], [1], [1]])]
    tY = [
        array([[0]]),
        array([[1]]),
        array([[1]]),
        array([[1]]),
        array([[0]])
    ]

    net.SGD(list(zip(tX, tY)), 1000, 5, 1, 6)

    x = array([[1], [0], [1]])
    y = net.feedforward(x)[-1]
    print(y)
