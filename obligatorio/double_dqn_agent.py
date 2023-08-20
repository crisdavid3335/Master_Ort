import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import numpy as np
from abstract_agent import Agent
import random

class DoubleDQNAgent(Agent):
    def __init__(self, gym_env, model_a, model_b, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, save_every, path_pesos_a, path_pesos_b , sync_target =  100):
        
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_time, epsilon_decay, episode_block, save_every)              
        
        # Dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Solo es para que guarde
        self.save_every = save_every
        self.path_pesos_a = path_pesos_a 
        self.path_pesos_b = path_pesos_b

        # Asignar los modelos al agente (y enviarlos al dispositivo adecuado)
        self.q_a = model_a.to(self.device)
        self.q_a.load_state_dict(torch.load(self.path_pesos_a))
    
        self.q_b = model_b.to(self.device)
        self.q_b.load_state_dict(torch.load(self.path_pesos_b))

        # Asignar una función de costo (MSE)  (y enviarla al dispositivo adecuado)
        self.loss_function = nn.MSELoss().to(self.device)
        
        # Asignar un optimizador para cada modelo (Adam)
        self.optimizer_A = torch.optim.Adam(self.q_a.parameters(), lr=learning_rate)
        self.optimizer_B = torch.optim.Adam(self.q_b.parameters(), lr=learning_rate)
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
                return (self.q_a(state)+self.q_b(state)).max(1)[1].view(1, 1)
            else:
                # Seleccionar una acción aleatoria del espacio de acciones del entorno
                return torch.tensor(
                    [[self.env.action_space.sample()]],
                    device=self.device,
                    dtype=torch.long,
                )
        else:
            # Seleccionar la acción greedy utilizando el modelo de política sin tener en cuenta la exploración epsilon
            return (self.q_a(state)+self.q_b(state)).max(1)[1].view(1, 1)

    def save_weights(self):
        """
        Guarda los pesos de los modelos en dos archivos.

        """
        torch.save(self.q_a.state_dict(), self.path_pesos_a)
        torch.save(self.q_b.state_dict(), self.path_pesos_b)

    def update_weights(self):
        """
        Actualiza los pesos del modelo utilizando el algoritmo de entrenamiento doble DQN.

        """
        if len(self.memory) > self.batch_size:
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

            # Actualizar al azar Q_a o Q_b usando el otro para calcular el valor de los siguientes estados.
            if random.random() > 0.5:
              # Resetear gradientes
              self.optimizer_A.zero_grad()

              # Obtener el valor estado-acción (Q) de acuerdo a la policy net para todos los elementos (estados) del minibatch.
              q_actual = self.q_a(state_batch)

              # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este cálculo.
              # Si el estado siguiente es terminal (done) este valor debería ser 0.
              q_siguiente = self.q_b(next_states).detach()
              max_q_siguiente = torch.max(q_siguiente, dim=1)[0].unsqueeze(dim=1)

              # Calcular el target de DQN de acuerdo a la Ecuación (3) del paper.
              Y = reward_batch + self.gamma * ((1 - dones) * max_q_siguiente)
              X = q_actual.gather(1, action_batch)

              # Calcular el costo y actualizar los pesos.
              # En PyTorch, la función de costo se llama con (predicciones, objetivos) en ese orden.
              loss = self.loss_function(X.squeeze(), Y.squeeze())
              loss.backward()
              self.optimizer_A.step()
            
            else:
              # Resetear gradientes
              self.optimizer_B.zero_grad()

              # Obtener el valor estado-acción (Q) de acuerdo a la policy net para todos los elementos (estados) del minibatch.
              q_actual = self.q_b(state_batch)

              # Obtener max a' Q para los siguientes estados (del minibatch). Es importante hacer .detach() al resultado de este cálculo.
              # Si el estado siguiente es terminal (done) este valor debería ser 0.
              q_siguiente = self.q_a(next_states).detach()
              max_q_siguiente = torch.max(q_siguiente, dim=1)[0].unsqueeze(dim=1)

              # Calcular el target de DQN de acuerdo a la Ecuación (3) del paper.
              Y = reward_batch + self.gamma * ((1 - dones) * max_q_siguiente)
              X = q_actual.gather(1, action_batch)

              # Calcular el costo y actualizar los pesos.
              # En PyTorch, la función de costo se llama con (predicciones, objetivos) en ese orden.
              loss = self.loss_function(X.squeeze(), Y.squeeze())
              loss.backward()
              self.optimizer_B.step()




