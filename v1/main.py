from makemore import MakeMore

def main():

    print("Initializing Makemore...\n")
    model = MakeMore('names.txt')
    model.create_data()

    print("Training Makemore...\n")
    i = 0
    while model.loss > 2.4999:
        model.forward_pass()
        model.backward_pass()
        model.update()
        i += 1

    print(f"Training completed in {i} iterations with final loss: {model.loss.item():.4f}\n")

    i = 5
    print(f"-------- Generated {i} names --------")
    for _ in range(i):
        print(model.make())
    print("---------------------------------")
    

if __name__ == "__main__":
    main()
