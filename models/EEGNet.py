# from https://github.com/amrzhd/EEGNet/blob/main/EEGNet.py#L36
import torch
import torch.nn as nn

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class EEGNet(nn.Module):  # EEGNet-8,2
    def __init__(self, chans, classes, time_points, f1=8, d=2, pk1=4, pk2=8, dp=0.5, max_norm1=1, max_norm2=0.25):
        super(EEGNet, self).__init__()
        # Calculating FC input features
        f2 = f1 * d
        self.linear_size = (time_points // (pk1 * pk2)) * f2
        # Temporal Filters
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 64), padding='same', bias=False),
            nn.BatchNorm2d(f1),
        )
        # Spatial Filters
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),  # Depthwise Conv
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1), stride=pk1),
            nn.Dropout(dp)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding='same'),  # Separable Conv
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),  # Pointwise Conv
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2), stride=pk2),
            nn.Dropout(dp)
        )
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(self.linear_size, classes)
        self.fc = LinearWithConstraint(self.linear_size, classes, max_norm=1)

        # Apply max_norm constraint to the depthwise layer in block2
        self._apply_max_norm(self.block2[0], max_norm1)

        # Apply max_norm constraint to the linear layer
        self._apply_max_norm(self.fc, max_norm2)

    def _apply_max_norm(self, layer, max_norm):
        for name, param in layer.named_parameters():
            if 'weight' in name:
                param.data = torch.renorm(param.data, p=2, dim=0, maxnorm=max_norm)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
        x = self.block1(x.unsqueeze(dim=1))
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    n = EEGNet(chans=22, classes=4, time_points=800).to('cpu')
    x = torch.randn(1, 22, 800).to('cpu')
    out = n(x)
    print(out.shape)
    summary(n, [(22, 800)])
