import torch.nn as nn
import torch.nn.functional as F


class DQN_CNN_Model(nn.Module):
    def __init__(self, env_inputs, n_actions):
        super(DQN_CNN_Model, self).__init__()

        # Obtener el número de canales de entrada
        in_channels = env_inputs

        # Capa convolucional 1
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()

        # Capa convolucional 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()

        # Capa fully-connected
        self.fc = nn.Linear(32 * 9 * 9, 256)

        # Capa de salida
        self.output = nn.Linear(256, n_actions)

    def forward(self, env_input):
        x = self.relu1(self.conv1(env_input))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)  # Aplanar la salida de la convolución
        x = F.relu(self.fc(x))
        output = self.output(x)
        return output
