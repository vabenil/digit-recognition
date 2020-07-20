import pickle
from mnist_loader import load_data_wrapper
from network import Network


if __name__ == '__main__':
    # Load weights and biases
    with open("temp", "rb") as f:
        W, B = pickle.load(f)

    net = Network([784, 50, 10])
    # Use the weights and biases from the previously trained network
    net.W = W
    net.B = B

    test_data = load_data_wrapper()[2]

    accuracy = net.accuracy(test_data)
    print("Nework accuracy is: %f%%" % (accuracy * 100))
