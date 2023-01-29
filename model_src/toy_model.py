from torch import nn
from torch.nn import functional as F


class toy_model(nn.Module):
    def __init__(self, first_channels=28*28, hidden_states=128, num_classes=10):
        super(toy_model, self).__init__()
        self.layer_1 = nn.Linear(first_channels, hidden_states)
        self.layer_2 = nn.Linear(hidden_states, hidden_states)
        self.layer_3 = nn.Linear(hidden_states, num_classes)

    def forward(self, _input):
        bs, ch, h, w = _input.shape
        _input = _input.view(bs, -1)
        output = F.relu(self.layer_1(_input))
        output = F.relu(self.layer_2(output))
        output = F.relu(self.layer_3(output))
        return output
