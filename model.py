import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):

    def __init__(self,S,D):
        super().__init__()
        #TODO: add multi-heads
        self.S = S
        self.D = D

        self.Wq = torch.randn(self.S,self.D)
        self.Wk = torch.randn(self.S,self.D)
        self.Wv = torch.randn(self.S,self.D)

    def forward(self,x):

        B,S,D = x.size()

        q = self.Wq.unsqueeze(0)*x
        k = self.Wk.unsqueeze(0)*x
        v = self.Wv.unsqueeze(0)*x

        attn = (q@k.transpose(1,2))@v
        #TODO: add causal masking
        return attn

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.w1 = nn.Linear(in_dim, hid_dim)
        self.relu = nn.ReLU()
        self.w2 = nn.Linear(hid_dim, in_dim)

    def forward(self,x):
        return self.w2(self.relu(self.w1(x)))



if __name__ == '__main__':

    B,S,D = 4,10,5
    x = torch.randn(B,S,D)
    attn = CausalSelfAttention(S,D)
    mlp = MLP()
    att = attn(x)
    print(f"shape of x: {x.shape} attn: {att.shape}")

