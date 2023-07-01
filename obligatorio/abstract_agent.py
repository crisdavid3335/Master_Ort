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


class Agent(ABC):
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
        self.obs_processing_func = obs_processing_func

        # Asignarle memoria al agente
        self.memory = ReplayMemory(memory_buffer_size)
        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time
        self.epsilon_decay = epsilon_decay
        self.episode_block = episode_block

    def train(
        self,
        number_episodes=50000,
        max_steps_for_episode=10000,
        max_steps=1000000,
        writer_name="default_writer_name",
        act_s_freq=700,
    ):
        rewards = []
        total_steps = 0
        writer = SummaryWriter(comment="-" + writer_name)

        for ep in tqdm(range(number_episodes), unit=" episodes"):
            if total_steps > max_steps:
                break

            # Observar estado inicial como indica el algoritmo
            state = self.env.reset()
            current_episode_reward = 0.0
            state = self.obs_processing_func(state)
            for s in range(max_steps_for_episode):
                # Seleccionar accion usando una política epsilon-greedy.
                action = self.select_action(state, ep)
                # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
                next_state, reward, done, info = self.env.step(action)
                next_state = self.obs_processing_func(next_state)
                current_episode_reward += reward
                total_steps += 1

                # Guardar la transicion en la memoria
                self.memory.add(state, action, reward, done, next_state)

                # Actualizar el estado
                state = next_state
                # Actualizar el modelo
                self.update_weights()

                if s % act_s_freq == 0:
                    self.act_s()

                if done:
                    break

            rewards.append(current_episode_reward)

            if current_episode_reward > max(rewards):
                self.load_best_model(current_episode_reward)

            mean_reward = np.mean(rewards[-100:])
            writer.add_scalar("epsilon", self.epsilon, total_steps)
            writer.add_scalar("reward_100", mean_reward, total_steps)
            writer.add_scalar("reward", current_episode_reward, total_steps)

            # Report on the traning rewards every EPISODE BLOCK episodes
            if ep % self.episode_block == 0:
                print(
                    f"Episode {ep} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:]):.3f} epsilon {self.epsilon:.2f} total steps {total_steps}"
                )

            print(
                f"Episode {ep + 1} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:]):.3f} epsilon {self.epsilon:.2f} total steps {total_steps}"
            )
        writer.close()
        self.load_best_model()
        return rewards

    def compute_epsilon(self, steps_so_far):
        eps = self.epsilon_i - (self.epsilon_i - self.epsilon_f) * min(
            1, steps_so_far / self.epsilon_anneal
        )
        return eps

    def record_test_episode(self, env):
        done = False
        state = env.reset()
        # Observar estado inicial como indica el algoritmo
        state = self.obs_processing_func(state)
        while not done:
            # env.render()  # Queremos hacer render para obtener un video al final.
            action = self.select_action(state, 0, False)
            state, reward, done, info = env.step(action)
            # Actualizar el estado
            state = self.obs_processing_func(state)
            if done:
                break

        env.close()
        show_video()

    @abstractmethod
    def select_action(self, state, current_steps, train=True):
        pass

    @abstractmethod
    def update_weights(self):
        pass

    @abstractmethod
    def act_s(self):
        pass

    @abstractmethod
    def load_best_model(self):
        pass
