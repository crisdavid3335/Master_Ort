import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
from abstract_agent import Agent
import random

class Agente_libro(Agent):
    def __init__(self, gym_env, model_a, model_b, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, save_every, path_pesos, sync_target =  500):
        
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, save_every, path_pesos)              
        
        # Dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Asignar los modelos al agente (y enviarlos al dispositivo adecuado)
        self.policy_net = model_a.to(self.device)
        self.policy_net.load_state_dict(torch.load(self.path_pesos))

        self.q_b = model_b.to(self.device)
        self.sync_target = sync_target

        # Solo es para que guarde
        self.save_every = save_every
        self.path_pesos = path_pesos

        # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss().to(self.device)
        

        # Asignar un optimizador para cada modelo (Adam)
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.gamma = gamma

    def select_action(self, state, current_steps, train=True):
      if train:
        self.epsilon = self.compute_epsilon(current_steps)

        if current_steps % self.sync_target == 0:
            self.q_b.load_state_dict(self.policy_net.state_dict())
        
        if random.random() > self.epsilon:
          with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
      else:
        with torch.no_grad():
          return self.policy_net(state).max(1)[1].view(1, 1)


    def update_weights(self):
        if len(self.memory) > self.batch_size:
          transitions = self.memory.sample(self.batch_size)
          batch = Transition(*zip(*transitions))
          state_batch = torch.cat(batch.state).to(self.device)
          action_batch = torch.cat(batch.action).to(self.device)
          reward_batch = torch.cat(batch.reward).unsqueeze(dim=1).to(self.device)
          dones = torch.tensor(batch.done, device=self.device, dtype=torch.float).unsqueeze(dim=1)
          next_states = torch.cat(batch.next_state).to(self.device)
            
          q_actual = self.policy_net(state_batch)
          with torch.no_grad():
            q_siguiente = self.q_b(next_states)

          max_q_siguiente = torch.max(q_siguiente, dim=1)[0].unsqueeze(dim=1)
          Y = reward_batch + self.gamma  * ((1 -dones) * max_q_siguiente)
          X = q_actual.gather(1, action_batch)

          self.optimizer.zero_grad()
          loss = self.loss_function(X, Y)
          loss.backward()
          self.optimizer.step()


