"""
Interactively test the network with training examples
"""
import pickle
from numpy import array, ubyte
from matplotlib import pyplot as plt
from network import Network
from mnist_loader import load_data_wrapper


def show_digit(digit):
    idata = array([
        [[255, 255, 255, 255] for j in range(28)] for i in range(28)
    ], dtype=ubyte)

    x, y = 0, 0
    for i in range(784):
        pixel = ubyte(255 * (1.0 - digit[i]))
        idata[y, x][0] = pixel
        idata[y, x][1] = pixel
        idata[y, x][2] = pixel
        idata[y, x][3] = ubyte(255)

        y += 1 * (x == 27 and y != 27)
        x = (x + 1) * (x < 28 - 1)

    plt.imshow(idata)
    plt.show()


if __name__ == '__main__':
    with open("temp", "rb") as f:
        W, B = pickle.load(f)

    net = Network([784, 50, 10])
    net.W = W
    net.B = B

    tdata = load_data_wrapper()[2]

    while True:
        i = int(input("Training example: "))
        if i < 0:
            exit()

        tx, ty = tdata[i]

        x = net.feedforward(tx)[-1]

        a = x.argmax()
        y = ty.argmax()

        print("Network guess: %d, confidense: %f" % (a, x[a]))
        print("Actual answer: %d" % y)

        show_digit(tx)
