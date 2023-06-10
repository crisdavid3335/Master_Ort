import random
from collections import deque


class ReplayMemory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
