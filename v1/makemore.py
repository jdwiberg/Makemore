import torch as t
import torch.nn.functional as F


class MakeMore():

    # Important dictionaries
    atoi = {chr(i+96): i for i in range(1, 27)}
    atoi['.'] = 0
    itoa = {v: k for k, v in atoi.items()}

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.names = open(self.filepath, 'r').read().splitlines()

        self.xs: t.LongTensor = None
        self.num = 0
        self.ys: t.LongTensor = None

        self.g = t.Generator().manual_seed(83039849392)
        self.W: t.LongTensor = t.randn((27,27), generator=self.g, requires_grad=True)

        self.loss: float = 100
        self.learning_rate: float = 10.0
        
    def create_data(self):
        xs, ys = [], []
        for name in self.names:
            chs = '.' + name + '.'
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.atoi[ch1]
                ix2 = self.atoi[ch2]
                xs.append(ix1)
                ys.append(ix2)

        self.xs = t.tensor(xs)
        self.num = len(xs)
        self.ys = t.tensor(ys)
    
    def forward_pass(self):
        xenc = F.one_hot(self.xs, num_classes=27).float()

        logits = xenc @ self.W
        count = logits.exp()
        probs = count / count.sum(1, keepdims=True)

        data_loss = -probs[t.arange(self.num), self.ys].log().mean()
        regularization_loss = 0.01 * (self.W**2).mean()
        self.loss = data_loss + regularization_loss


    def backward_pass(self):
        self.W.grad = None
        self.loss.backward()

    def update(self):
        self.W.data += -self.learning_rate * self.W.grad

    def make(self):
        ix = 0
        out = []

        while True:
            xenc = F.one_hot(self.xs, num_classes=27).float()
            logits = xenc @ self.W
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)

            ix = t.multinomial(probs[ix], num_samples=1, replacement=True, generator=self.g).item()
            out.append(self.itoa[ix])

            if ix == 0:
                break

        return ''.join(out)

    


    



    

