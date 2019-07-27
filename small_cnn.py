from torch import nn


class SmallCNN(nn.Module):
    def __init__(self,pretrained):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=3)
        self.max_pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.relu(x)

        return x
