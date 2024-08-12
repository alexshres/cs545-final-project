import nn

def main():
    model = nn.MNISTNeural(learning_rate=0.01, batches=64, epochs=20)
    model.run()



if __name__ == "__main__":
    main()
