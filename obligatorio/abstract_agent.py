import torch
from torch import optim
import torch.nn as nn
from replay_memory import ReplayMemory
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from mario_utils import show_video, wrap_env
import random
import numpy as np


class Agent:
    # class Agent(ABC):
    def __init__(
        self,
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
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Funcion phi para procesar los estados.
        self.model = obs_processing_func

        # Asignarle memoria al agente
        self.memory = ReplayMemory(memory_buffer_size)
        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time
        self.epsilon_decay = epsilon_decay
        self.episode_block = episode_block

        self.total_steps = 0

    def train(
        self,
        number_episodes=50000,
        max_steps_episode=10000,
        max_steps=1000000,
        writer_name="default_writer_name",
    ):
        rewards = []
        total_steps = 0
        writer = SummaryWriter(comment="-" + writer_name)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        losses = []

        for ep in tqdm(range(number_episodes), unit=" episodes"):
            if total_steps > max_steps:
                break

            # Observar estado inicial como indica el algoritmo
            state = self.env.reset()
            current_episode_reward = 0.0

            for s in range(max_steps):
                self.qval_ = self.model(torch.from_numpy(np.array(state)).unsqueeze(0))
                # Seleccionar accion usando una política epsilon-greedy.
                self.epsilon = self.compute_epsilon(s)
                if random.random() < self.epsilon:
                    action = torch.randint(0, self.env.action_space.n, (1,)).item()
                else:
                    action = torch.argmax(self.qval_).item()
                # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
                next_state, reward, done, info = self.env.step(action)
                current_episode_reward += reward
                total_steps += 1

                # Guardar la transicion en la memoria
                self.memory.add(state, action, reward, done, next_state)

                # Actualizar el estado
                state = next_state
                # Actualizar el modelo
                if len(self.memory) > self.batch_size:
                    self.minibatch = self.memory.sample(self.batch_size)
                    self.state_batch = torch.cat(
                        [
                            torch.from_numpy(np.array(s1)).unsqueeze(0)
                            for (s1, a, r, d, s2) in self.minibatch
                        ]
                    )
                    self.action_batch = torch.Tensor(
                        [a for (s1, a, r, d, s2) in self.minibatch]
                    )
                    self.reward_batch = torch.Tensor(
                        [r for (s1, a, r, d, s2) in self.minibatch]
                    )
                    self.done_batch = torch.Tensor(
                        [d for (s1, a, r, d, s2) in self.minibatch]
                    )
                    self.next_state_batch = torch.cat(
                        [
                            torch.from_numpy(np.array(s2)).unsqueeze(0)
                            for (s1, a, r, d, s2) in self.minibatch
                        ]
                    )

                    Q1 = self.model(self.state_batch)
                    with torch.no_grad():
                        Q2 = self.model(self.next_state_batch)

                    Y = self.reward_batch + self.gamma * (
                        (1 - self.done_batch) * torch.max(Q2, dim=1)[0]
                    )
                    X = Q1.gather(
                        dim=1, index=self.action_batch.long().unsqueeze(dim=1)
                    ).squeeze()
                    loss = self.loss_fn(X, Y.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    losses.append(loss.item())
                    optimizer.step()

                if done:
                    break
                if s == max_steps_episode:
                    break

            rewards.append(current_episode_reward)
            mean_reward = np.mean(rewards[-100:])
            writer.add_scalar("epsilon", self.epsilon, total_steps)
            writer.add_scalar("reward_100", mean_reward, total_steps)
            writer.add_scalar("reward", current_episode_reward, total_steps)

            # Report on the traning rewards every EPISODE BLOCK episodes
            if ep % self.episode_block == 0:
                print(
                    f"Episode {ep} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}"
                )

        print(
            f"Episode {ep + 1} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon} total steps {total_steps}"
        )

        torch.save(
            self.model.state_dict(),
            "/mnt/c/Users/crisd/OneDrive/Escritorio/python/mario/letra/obligatorio/modelo_pesos.pth",
        )
        writer.close()

        return rewards

    def compute_epsilon(self, steps_so_far):
        if steps_so_far < self.epsilon_anneal:
            # Utiliza una función no lineal para acelerar la disminución de epsilon
            epsilon_ratio = 1 - (steps_so_far / self.epsilon_anneal)
            return self.epsilon_i * epsilon_ratio + self.epsilon_f * (1 - epsilon_ratio)
        else:
            return self.epsilon_f

    def record_test_episode(self, env):
        done = False

        env = wrap_env(env)
        state = env.reset()
        # Observar estado inicial como indica el algoritmo

        while not done:
            # env.render()  # Queremos hacer render para obtener un video al final.
            self.qval_ = self.model(torch.from_numpy(np.array(state)).unsqueeze(0))
            action = torch.argmax(self.qval_).item()
            state, reward, done, info = env.step(action)
            if done:
                break

            # Actualizar el estado

        env.close()
        show_video()


#    @abstractmethod
#    def select_action(self, state, current_steps, train=True):
#        pass
#
#    @abstractmethod
#    def update_weights(self):
#        pass
