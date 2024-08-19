# import nn
import numpy as np
from svm import SVM


def main():
    training_data = np.genfromtxt("mnist_train.csv", dtype=int, delimiter=",", skip_header=1)
    testing_data = np.genfromtxt("mnist_test.csv", dtype=int, delimiter=",", skip_header=1)


    # model = nn.MNISTNeural(learning_rate=0.01, batches=64, epochs=20)
    # model.run()

    svm = SVM(num_classes=10, kernel="linear")
    svm.train(training_data)
    svm.test(testing_data)

    print(svm.accuracy)
    print(svm)


if __name__ == "__main__":
    main()
