import torch
import torchvision


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.effi = torchvision.models.efficientnet_b7(pretrained=True)
        self.fc = torch.nn.Linear(1000, 27)

    def forward(self, x):
        x = self.effi(x)
        x = self.fc(x)

        return x
