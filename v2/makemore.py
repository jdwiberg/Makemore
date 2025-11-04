import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random


class Makemore():

    # Important dictionaries
    stoi = {chr(i+96): i for i in range(1, 27)}
    stoi['.'] = 0
    itos = {v: k for k, v in stoi.items()}

    def __init__(self, filepath: str, block_size: int, learning_rate: float):
        self.names = open(filepath, 'r').read().splitlines()

        self.X_tr: torch.LongTensor = None
        self.Y_tr: torch.LongTensor = None
        
        self.X_dev: torch.LongTensor = None
        self.Y_dev: torch.LongTensor = None

        self.X_test: torch.LongTensor = None
        self.Y_test: torch.LongTensor = None

        self.g = torch.Generator().manual_seed(2147483647)
        self.C = torch.randn((27, 10), generator=self.g)
        self.W1 = torch.randn((30, 200), generator=self.g)
        self.b1 = torch.randn(200, generator=self.g)
        # h = 32 x 100
        self.W2 = torch.randn((200, 27), generator=self.g)
        self.b2 = torch.randn(27, generator=self.g)
        self.params = [self.C, self.W1, self.b1, self.W2, self.b2]
        self.param_count = sum(p.nelement() for p in self.params)

        for p in self.params:
            p.requires_grad = True
        

        self.tr_loss: float = 100
        self.dev_loss: float = 100
        self.test_loss: float = 100

        # Hyper Parameters
        self.block_size = block_size  # how many characters do we take to predict the next
        self.learning_rate: float = learning_rate
        self.batch_size: int = 32

    def create_data(self, names):
        X, Y = [], []
        for n in names:
            context = [0] * self.block_size #HYPER PARAM
            for ch in n + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                # print(''.join(itos[i] for i in context), '--->', itos[ix])
                context = context[1:] + [ix]
        return torch.tensor(X), torch.tensor(Y)
    
    def make_splits(self, train_n : float):
        random.shuffle(self.names)
        n1 = int(train_n * len(self.names))
        n2 = int(((1 - train_n) / 2) * len(self.names)) + n1

        self.X_tr, self.Y_tr = self.create_data(self.names[:n1])
        self.X_dev, self.Y_dev = self.create_data(self.names[n1:n2])
        self.X_test, self.Y_test = self.create_data(self.names[n2:])

    def forward_pass(self):
        # minibatch construction
        ix = torch.randint(0, self.X_tr.shape[0], (self.batch_size,))
        
        emb = self.C[self.X_tr[ix]]
        h = torch.tanh(emb.view(-1, 30) @ self.W1 + self.b1) # first layer
        logits = h @ self.W2 + self.b2 # second / final layer

        # counts = logits.exp()
        # probs = counts / counts.sum()
        # loss = -probs[torch.arange(32), Y].log().mean()
            # ==
        self.tr_loss = F.cross_entropy(logits, self.Y_tr[ix])

    def backward_pass(self):
        for p in self.params:
            p.grad = None
        self.tr_loss.backward()

    def update(self):
        for p in self.params:
            p.data += -self.learning_rate * p.grad
    
    def get_loss(self, X, Y):
        emb = self.C[X]
        h = torch.tanh(emb.view(-1, 30) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        self.dev_loss = F.cross_entropy(logits, Y)
        return self.dev_loss.item()
    
    def sample(self):
        ix = 0
        out = []
        context = [0] * self.block_size

        while True:
            emb = self.C[torch.tensor([context])]
            h = torch.tanh(emb.view(-1, 30) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)

            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=self.g).item()
            out.append(self.itos[ix])

            context = context[1:] + [ix]

            if ix == 0:
                break

        return ''.join(out)
    
