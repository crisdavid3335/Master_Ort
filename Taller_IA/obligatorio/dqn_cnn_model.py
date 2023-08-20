import torch.nn as nn
import torch.nn.functional as F


class DQN_CNN_Model(nn.Module):
    def __init__(self, env_inputs, n_actions):
        """
        Inicializa una instancia del modelo de red neuronal convolucional para DQN.

        Args:
            env_inputs (tuple): Tupla que indica las dimensiones de entrada del entorno.
            n_actions (int): Número de acciones posibles en el entorno.
        """
        super(DQN_CNN_Model, self).__init__()

        # Obtener el número de canales de entrada
        in_channels = env_inputs[0]

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
        """
        Realiza la propagación hacia adelante a través del modelo.

        Args:
            env_input (torch.Tensor): Tensor de entrada del entorno.

        Returns:
            torch.Tensor: Tensor de salida que representa las acciones estimadas.
        """
        x = self.relu1(self.conv1(env_input))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)  # Aplanar la salida de la convolución
        x = F.relu(self.fc(x))
        return self.output(x)
