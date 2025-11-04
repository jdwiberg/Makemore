from makemore import Makemore
import matplotlib.pyplot as plt

def main():
    
    print("Initializing Makemore...\n")
    model = Makemore('../data/names.txt', steps=100_000, block_size=4, batch_size=32, lr_start=0.1, lr_end=0.1, lookup_dim=10, hidden_dim=200)
    model.make_splits(train_n=0.8)

    print("Training Makemore...\n")

    stepi = []
    lossi = []
    
    for i in range(model.steps):
        model.forward_pass()
        model.backward_pass()
        model.update()

        stepi.append(i + 1)
        lossi.append(model.tr_loss.log10().item())

    plt.plot(stepi, lossi)


    print(f"Training completed in {i + 1} iterations with final loss: {model.tr_loss.item():.4f}\n")

    n = 5
    print(f"-------- Generated {n} names --------")
    for _ in range(n):
        print(model.sample())
    print("---------------------------------")

    plt.show()

if __name__ == "__main__":
    main()