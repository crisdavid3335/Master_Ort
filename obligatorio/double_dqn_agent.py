import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
from abstract_agent import Agent
import random


class DoubleDQNAgent(Agent):
    def __init__(
        self,
        gym_env,
        model_a,
        model_b,
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
        sync_target=100,
    ):
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
        )

        # Asignar los modelos al agente (y enviarlos al dispositivo adecuado)
        self.q_a = model_a
        self.q_b = model_b
        self.env = gym_env
        # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss()

        # Asignar un optimizador para cada modelo (Adam)
        self.optimizer_A = torch.optim.Adam(
            self.q_a.parameters(), lr=self.learning_rate
        )
        self.optimizer_B = torch.optim.Adam(
            self.q_b.parameters(), lr=self.learning_rate
        )

    def select_action(self, state, current_steps, train=True):
        # Implementar. Seleccionando acciones epsilongreedy-mente (sobre Q_a + Q_b)
        # si estamos entranando y completamente greedy en otro caso.
        if train:
            self.epsilon = self.compute_epsilon(current_steps)
            if random.random() < self.epsilon:
                action = torch.randint(0, self.env.action_space.n, (1,)).item()
            else:
                action = torch.argmax(self.q_a + self.q_b).item()
        else:
            action = torch.argmax(self.q_a + self.q_b).item()

    def update_weights(self):
        if len(self.memory) > self.batch_size:
            self.minibatch = self.memory.sample(self.batch_size)
            self.state_batch = torch.cat(
                [
                    torch.from_numpy(np.array(s1)).unsqueeze(0).to(self.device)
                    for (s1, a, r, d, s2) in self.minibatch
                ]
            ).to(self.device)
            self.action_batch = torch.Tensor(
                [a for (s1, a, r, d, s2) in self.minibatch]
            ).to(self.device)
            self.reward_batch = torch.Tensor(
                [r for (s1, a, r, d, s2) in self.minibatch]
            ).to(self.device)
            self.done_batch = torch.Tensor(
                [d for (s1, a, r, d, s2) in self.minibatch]
            ).to(self.device)
            self.next_state_batch = torch.cat(
                [
                    torch.from_numpy(np.array(s2)).unsqueeze(0).to(self.device)
                    for (s1, a, r, d, s2) in self.minibatch
                ]
            ).to(self.device)
            # Actualizar al azar Q_a o Q_b usando el otro para calcular el valor de los siguientes estados.
            Q1 = self.q_a(self.state_batch)
            with torch.no_grad():
                Q2 = self.q_b(self.state_batch)
            # Para el Q elegido:
            # Obetener el valor estado-accion (Q) de acuerdo al Q seleccionado.
            Y = self.reward_batch + self.gamma * (
                (1 - self.done_batch) * torch.max(Q2, dim=1)[0]
            )
            X = Q1.gather(
                dim=1, index=self.action_batch.long().unsqueeze(dim=1)
            ).squeeze()

            # Compute el target de DQN de acuerdo a la Ecuacion (3) del paper.
            loss = self.loss_function(X, Y.detach())
            self.optimizer_A.zero_grad()
            self.optimizer_A.step()
