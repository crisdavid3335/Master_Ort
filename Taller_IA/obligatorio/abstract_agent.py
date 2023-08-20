import torch
import torch.nn as nn
from replay_memory import ReplayMemory, Transition
from abc import ABC, abstractmethod
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from mario_utils import show_video
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
        save_every,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Funcion phi para procesar los estados.
        self.state_processing_function = obs_processing_func

        # Asignarle memoria al agente
        self.memory = ReplayMemory(memory_buffer_size)

        self.env = gym_env

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.epsilon_anneal = epsilon_anneal_time
        self.epsilon_decay = epsilon_decay
        self.episode_block = episode_block

        self.total_steps = 0
        self.save_every = save_every
        self.epsilon = self.epsilon_i

    def train(
        self,
        number_episodes=50000,
        max_steps_episode=10000,
        max_steps=1000000,
        writer_name="default_writer_name",
    ):
        """
        Entrena el modelo de Reinforcement Learning mediante el algoritmo DQN.

        Args:
            number_episodes (int): Número total de episodios a entrenar. Por defecto: 50000.
            max_steps_episode (int): Número máximo de pasos por episodio. Por defecto: 10000.
            max_steps (int): Número máximo de pasos totales. Por defecto: 1000000.
            writer_name (str): Nombre del escritor de TensorBoard. Por defecto: "default_writer_name".

        Returns:
            list: Lista de recompensas obtenidas en cada episodio.
        """
        rewards = []  # Lista para almacenar las recompensas obtenidas en cada episodio
        total_steps = 0  # Contador de pasos totales
        writer = SummaryWriter(
            comment="-" + writer_name
        )  # Crear objeto SummaryWriter para TensorBoard

        # Bucle para los pasos dentro de un episodio
        for ep in tqdm(range(number_episodes), unit=" episodes"):
            if total_steps > max_steps:
                break

            # Observar estado inicial como indica el algoritmo
            state = self.env.reset()
            state = self.state_processing_function(state)
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # Inicializar la recompensa acumulada para el episodio actual
            current_episode_reward = 0.0

            # Bucle para los pasos dentro de un episodio
            for s in range(max_steps):
                # Seleccionar acción usando una política epsilon-greedy.
                action = self.select_action(state, total_steps)

                # Ejecutar la acción, observar el resultado y procesarlo como indica el algoritmo.
                next_state, r, done, info = self.env.step(action.item())
                reward = torch.tensor([r], device=self.device)
                current_episode_reward += r
                total_steps += 1

                # Controlar que el siguiente estado sea 0 en caso de estado terminal
                next_state = torch.tensor(
                    self.state_processing_function(next_state),
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)

                # Guardar la transición en la memoria
                self.memory.add(state, action, reward, done, next_state)

                # Actualizar el estado
                state = next_state

                # Actualizar el modelo
                self.update_weights()

                # Finalizar el episodio si se alcanza un estado terminal
                if done:
                    break

            # Calcular la media de las últimas 100 recompensas
            rewards.append(current_episode_reward)
            mean_reward = np.mean(rewards[-100:])

            # Registrar los valores en TensorBoard
            writer.add_scalar("epsilon", self.epsilon, total_steps)
            writer.add_scalar("reward_100", mean_reward, total_steps)
            writer.add_scalar("reward", current_episode_reward, total_steps)

            # Guardar los pesos del modelo cada cierto número de pasos
            if total_steps % self.save_every == 0:
                self.save_weights()

            # Mostrar el reporte de recompensas de entrenamiento cada cierto número de episodios
            if ep % self.episode_block == 0:
                print(
                    f"Episode {ep} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon:.2f} total steps {total_steps}"
                )
            # Imprimir el último reporte de recompensas y pasos totales
        print(
            f"Episode {ep + 1} - Avg. Reward over the last {self.episode_block} episodes {np.mean(rewards[-self.episode_block:])} epsilon {self.epsilon:.2f} total steps {total_steps}"
        )
        # Cerrar el objeto SummaryWriter
        writer.close()

        return rewards

    def compute_epsilon(self, steps_so_far):
        """
        Calcula el valor actual de epsilon para la política epsilon-greedy.

        Args:
            steps_so_far (int): Número de pasos realizados hasta ahora.

        Returns:
            float: Valor actual de epsilon.

        """
        # Actualiza el valor de epsilon multiplicándolo por un factor de decaimiento.
        self.epsilon = max(self.epsilon_f, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def record_test_episode(self, env):
        # Ejecuta un episodio de prueba en el entorno y registra el rendimiento del agente.
        done = False

        # Observar estado inicial como indica el algoritmo
        state = env.reset()
        state = self.state_processing_function(state)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )

        while not done:
            env.render()  # Queremos hacer render para obtener un video al final.

            # Seleccione una accion de forma completamente greedy.
            action = self.select_action(state, 0, False)

            # Ejecutar la accion, observar resultado y procesarlo como indica el algoritmo.
            state, r, done, info = env.step(action.item())
            state = self.state_processing_function(state)
            state = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            if done:
                break
            # Actualizar el estado

        env.close()
        show_video()  # Mostrar el video del rendimiento del agente.

    @abstractmethod
    def select_action(self, state, current_steps, train=True):
        pass

    @abstractmethod
    def update_weights(self):
        pass

    @abstractmethod
    def save_weights(self):
        pass
