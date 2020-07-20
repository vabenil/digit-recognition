"""
Train model and saves weights and biases to "temp" file
"""
from mnist_loader import load_data_wrapper
from network import Network
import pickle


if __name__ == '__main__':
    tdata, vdata, test_data = load_data_wrapper()

    net = Network([784, 50, 10])

    # Train
    net.SGD(tdata, 10, 10, 300, 4)

    # save weights and biases
    data = [net.W, net.B]
    with open("temp", "wb") as f:
        pickle.dump(data, f)
