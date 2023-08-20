import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
from abstract_agent import Agent
import random


class DQNAgent(Agent):
    def __init__(
        self,
        gym_env,
        model,
        obs_processing_func,
        memory_buffer_size,
        batch_size,
        learning_rate,
        gamma,
        epsilon_i,
        epsilon_f,
        epsilon_anneal_time,
        epsilon_decay,
        episode_block,
        save_every,
        path_pesos,
    ):
        """
        Inicializa un agente DQN.

        """
        super().__init__(
            gym_env,
            obs_processing_func,
            memory_buffer_size,
            batch_size,
            learning_rate,
            gamma,
            epsilon_i,
            epsilon_f,
            epsilon_anneal_time,
            epsilon_decay,
            episode_block,
            save_every,
        )

        # Asignar el modelo al agente (y enviarlo al dispositivo adecuado)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_pesos = path_pesos
        self.policy_net = model.to(self.device)
        self.policy_net.load_state_dict(
            torch.load(self.path_pesos)
        )  # <-- Se carga el peso de los entrenamientos anteriores
        self.save_every = save_every

        # Asignar una función de costo (MSE) (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss().to(self.device)

        # Asignar un optimizador (Adam)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.gamma = gamma

    def select_action(self, state, current_steps, train=True):
        """
        Selecciona una acción epsilon-greedy si se está entrenando y completamente greedy en otro caso.

        Args:
            state (torch.Tensor): Estado actual.
            current_steps (int): Número de pasos actuales.
            train (bool): Indica si el modelo está siendo entrenado. Por defecto: True.

        Returns:
            torch.Tensor: Acción seleccionada.
        """
        if train:
            self.epsilon = self.compute_epsilon(current_steps)
            if random.random() > self.epsilon:
                # Seleccionar la acción greedy utilizando el modelo de política
                return self.policy_net(state).max(1)[1].view(1, 1)
            else:
                # Seleccionar una acción aleatoria del espacio de acciones del entorno
                return torch.tensor(
                    [[self.env.action_space.sample()]],
                    device=self.device,
                    dtype=torch.long,
                )
        else:
            # Seleccionar la acción greedy utilizando el modelo de política sin tener en cuenta la exploración epsilon
            return self.policy_net(state).max(1)[1].view(1, 1)

    def save_weights(self):
        """
        Guarda los pesos del modelo en un archivo.

        """
        torch.save(self.policy_net.state_dict(), self.path_pesos)

    def update_weights(self):
        """
        Actualiza los pesos del modelo utilizando el algoritmo de entrenamiento DQN.

        """
        if len(self.memory) > self.batch_size:
            # Resetear gradientes
            self.optimizer.zero_grad()

            # Obtener un minibatch de la memoria. Resultando en tensores de estados, acciones, recompensas, flags de terminación y siguientes estados.
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))

            # Enviar los tensores al dispositivo correspondiente.
            state_batch = torch.cat(batch.state).to(self.device)
            action_batch = torch.cat(batch.action).to(self.device)
            reward_batch = torch.cat(batch.reward).unsqueeze(dim=1).to(self.device)
            # Dones debería ser 0 y 1; no True y False. Pueden usar .float() en un tensor para convertirlo
            dones = torch.tensor(
                batch.done, device=self.device, dtype=torch.float
            ).unsqueeze(dim=1)
            next_states = torch.cat(batch.next_state).to(self.device)

            # Obtener el valor estado-acción (Q) de acuerdo a la policy net para todos los elementos (estados) del minibatch.
            q_actual = self.policy_net(state_batch)

            # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este cálculo.
            # Si el estado siguiente es terminal (done) este valor debería ser 0.
            q_siguiente = self.policy_net(next_states).detach()
            max_q_siguiente = torch.max(q_siguiente, dim=1)[0].unsqueeze(dim=1)

            # Calcular el target de DQN de acuerdo a la Ecuación (3) del paper.
            Y = reward_batch + self.gamma * ((1 - dones) * max_q_siguiente)
            X = q_actual.gather(1, action_batch)

            # Calcular el costo y actualizar los pesos.
            # En PyTorch, la función de costo se llama con (predicciones, objetivos) en ese orden.
            loss = self.loss_function(X.squeeze(), Y.squeeze())
            loss.backward()
            self.optimizer.step()
