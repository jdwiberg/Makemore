from makemore import Makemore

def main():
    
    print("Initializing Makemore...\n")
    model = Makemore('../data/names.txt', block_size=3, learning_rate=0.1)
    model.make_splits(train_n=0.8)

    print("Training Makemore...\n")
    i = 0
    while model.tr_loss > 2.2:
        model.forward_pass()
        model.backward_pass()
        model.update()
        i += 1

    print(f"Training completed in {i} iterations with final loss: {model.tr_loss.item():.4f}\n")

    i = 5
    print(f"-------- Generated {i} names --------")
    for _ in range(i):
        print(model.sample())
    print("---------------------------------")

if __name__ == "__main__":
    main()