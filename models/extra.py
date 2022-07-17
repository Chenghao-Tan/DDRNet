import torch


class pre(torch.nn.Module):
    def __init__(self):
        super(pre, self).__init__()

    def forward(self, x):
        if not self.training:
            with torch.no_grad():
                return (x - x.min()) / (x.max() - x.min())
        else:
            return x


class post(torch.nn.Module):
    def __init__(self, n_classes):
        super(post, self).__init__()
        self.n_classes = n_classes

    def forward(self, x):
        if not self.training:
            with torch.no_grad():
                if self.n_classes == 1:
                    out = torch.sigmoid(x)
                else:
                    out = torch.nn.functional.softmax(x, dim=1)  # type: ignore
                return out
        else:
            return x
