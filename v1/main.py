from makemore import MakeMore
import sys

def main():

    print("Initializing Makemore...\n")
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_names_file>")
        return
    model = MakeMore(sys.argv[1], )
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
