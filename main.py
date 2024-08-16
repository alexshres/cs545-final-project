# import nn
from svm import SVM
from kernel_functions import kernel_and

def main():
    # model = nn.MNISTNeural(learning_rate=0.01, batches=64, epochs=20)
    # model.run()

    svm = SVM((1, 1), kernel_and)




if __name__ == "__main__":
    main()
