import nn

def main():
    learning_rates = [0.001] #[0.1, 0.01, 0.001]
   
    accuracies = []
    for lr in learning_rates:
        model = nn.MNISTNeural(learning_rate=lr, batches=64, epochs=20)
        metrics = model.run()

        # Grabbing final accuracy at end of epoch and adding it to accuracies list
        accuracies.append(metrics[-1][-1])

    for i, lr in enumerate(learning_rates):
        print(f"{lr=}\taccuracy={accuracies[i]}")

if __name__ == "__main__":
    main()
