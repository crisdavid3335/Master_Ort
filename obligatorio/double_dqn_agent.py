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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Asignar los modelos al agente (y enviarlos al dispositivo adecuado)
        self.modelo_a = model_a
        self.modelo_b = model_b
        self.env = gym_env
        # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss()

        # Asignar un optimizador para cada modelo (Adam)
        self.optimizer_A = torch.optim.Adam(
            self.modelo_a.parameters(), lr=self.learning_rate
        )
        self.optimizer_B = torch.optim.Adam(
            self.modelo_b.parameters(), lr=self.learning_rate
        )
        self.obs_processing_func = obs_processing_func

    def act_s(self):
        pass

    def load_best_model(self, current_episode_reward=0):
        self.best_reward = current_episode_reward
        self.best_model_params_a = self.modelo_a.state_dict()
        self.best_model_params_b = self.modelo_b.state_dict()
        if current_episode_reward == 0:
            self.modelo_a.load_state_dict(self.best_model_params_a)
            self.modelo_b.load_state_dict(self.best_model_params_b)

    def select_action(self, state, current_steps, train=True):
        # Implementar. Seleccionando acciones epsilongreedy-mente (sobre Q_a + Q_b)
        # si estamos entranando y completamente greedy en otro caso.
        if train:
            self.epsilon = self.compute_epsilon(current_steps)
            if random.random() < self.epsilon:
                return torch.randint(0, self.env.action_space.n, (1,)).item()
            else:
                return torch.argmax(
                    (self.modelo_a(state) + self.modelo_b(state))
                ).item()
        else:
            return torch.argmax((self.modelo_a(state) + self.modelo_b(state))).item()

    def update_weights(self):
        if len(self.memory) > self.batch_size:
            self.minibatch = self.memory.sample(self.batch_size)
            self.state_batch = torch.cat(
                [s1 for (s1, a, r, d, s2) in self.minibatch]
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
                [s2 for (s1, a, r, d, s2) in self.minibatch]
            ).to(self.device)

            # Actualizar al azar Q_a o Q_b usando el otro para calcular el valor de los siguientes estados.
            if random.randrange(0, 2) == 1:
                Q1 = self.modelo_a(self.state_batch)
                with torch.no_grad():
                    Q2 = self.modelo_b(self.next_state_batch)

                Y = self.reward_batch + self.gamma * (
                    (1 - self.done_batch) * torch.max(Q2, dim=1)[0]
                )
                X = Q1.gather(
                    dim=1, index=self.action_batch.long().unsqueeze(dim=1)
                ).squeeze()
                loss = self.loss_function(X, Y.detach())
                self.optimizer_A.zero_grad()
                loss.backward()
                self.optimizer_A.step()
            else:
                Q1 = self.modelo_b(self.state_batch)
                with torch.no_grad():
                    Q2 = self.modelo_a(self.next_state_batch)
                # Para el Q elegido:
                # Obetener el valor estado-accion (Q) de acuerdo al Q seleccionado.
                Y = self.reward_batch + self.gamma * (
                    (1 - self.done_batch) * torch.max(Q2, dim=1)[0]
                )
                X = Q1.gather(
                    dim=1, index=self.action_batch.long().unsqueeze(dim=1)
                ).squeeze()
                loss = self.loss_function(X, Y.detach())
                self.optimizer_B.zero_grad()
                loss.backward()
                self.optimizer_B.step()
