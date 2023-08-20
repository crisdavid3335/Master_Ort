import random
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state)

class ReplayMemory:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = deque([], maxlen=buffer_size)
        self.position = 0

    def add(self, *args):
      # Implementar.
      self.memory.append(Transition(*args))

    def sample(self, batch_size):
      # Implementar.
      return random.sample(self.memory, batch_size)

    def __len__(self):
      # Implementar.
      return len(self.memory)