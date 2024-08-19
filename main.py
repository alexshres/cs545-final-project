import nn
import numpy as np
from svm import SVM


def main():
    training_data = np.genfromtxt("mnist_train.csv", dtype=int, delimiter=",", skip_header=1)
    testing_data = np.genfromtxt("mnist_test.csv", dtype=int, delimiter=",", skip_header=1)


    model = nn.MNISTNeural(learning_rate=0.01, batches=64, epochs=20)
    model.run()

    learning_rates = [0.01, 0.001, 0.0001]
   
    accuracies = []
    for lr in learning_rates:
        model = nn.MNISTNeural(learning_rate=lr, batches=64, epochs=20)
        metrics = model.train_and_validate()


        # Grabbing final accuracy at end of epoch and adding it to accuracies list
        accuracies.append(metrics[-1][-1])
        
        model.predict()

    for i, lr in enumerate(learning_rates):
        print(f"{lr=}\taccuracy={accuracies[i]}")
    

    svm = SVM(num_classes=10, kernel="linear")
    svm.train(training_data)
    svm.test(testing_data, limit=100)

    print(svm.accuracy)
    print(svm)

if __name__ == "__main__":
    main()
