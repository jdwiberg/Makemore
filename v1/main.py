from makemore import MakeMore

def main():

    model = MakeMore('../data/names.txt')
    model.create_data()

    while model.loss > 2.4999:
        model.forward_pass()
        print(model.loss.item())

        model.backward_pass()
        model.update()

    for _ in range(5):
        print(model.make())
    

if __name__ == "__main__":
    main()
