import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory
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
        # Asignar el modelo al agente (y enviarlo al dispositivo adecuado)
        self.policy_net = model
        # Resto del código de inicialización

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss().to(self.device)

        # Asignar un optimizador (Adam)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=learning_rate
        )
        self.obs_processing_func = obs_processing_func

    def act_s(self):
        pass

    def save_every_f(self, save_every):
        pass

    def load_best_model(self, current_episode_reward=0):
        self.best_reward = current_episode_reward
        self.best_model_params = self.policy_net.state_dict()
        if current_episode_reward == 0:
            self.policy_net.load_state_dict(self.best_model_params)

    def select_action(self, state, current_steps, train=True):
        # Implementar. Seleccionando acciones epsilongreedy-mente si estamos entranando y completamente greedy en otro caso.
        if train:
            self.epsilon = self.compute_epsilon(current_steps)
            if random.random() < self.epsilon:
                return torch.randint(0, self.env.action_space.n, (1,)).item()
            else:
                return torch.argmax(self.policy_net(state)).item()
        else:
            return torch.argmax(self.policy_net(state)).item()

    def update_weights(self):
        if len(self.memory) > self.batch_size:
            self.minibatch = self.memory.sample(self.batch_size)
            self.state_batch = torch.cat(
                [
                    torch.from_numpy(np.array(s1.cpu())).unsqueeze(0).to(self.device)
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
                    torch.from_numpy(np.array(s2.cpu())).unsqueeze(0).to(self.device)
                    for (s1, a, r, d, s2) in self.minibatch
                ]
            ).to(self.device)

            Q1 = self.policy_net(self.state_batch).to(self.device)
            with torch.no_grad():
                Q2 = self.policy_net(self.next_state_batch).to(self.device)

            Y = self.reward_batch + self.gamma * (
                (1 - self.done_batch) * torch.max(Q2, dim=1)[0]
            ).to(self.device)
            X = (
                Q1.gather(dim=1, index=self.action_batch.long().unsqueeze(dim=1))
                .squeeze()
                .to(self.device)
            )
            loss = self.loss_fn(X, Y.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
