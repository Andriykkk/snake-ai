import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np

from config import window_x, window_y
from model import SnakeNet, Trainer

BATCH_SIZE = 32


class Agent:
    def __init__(self, model, lr, gamma, epsilon, memory_size=100000):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=memory_size)
        self.model = model
        self.trainer = Trainer(self.model, self.lr, self.gamma)

    def get_state(self, snake_position, fruit_position, move_direction, snake_body, score):
        head_x, head_y = snake_position
        fruit_x, fruit_y = fruit_position

        sc25 = False
        sc50 = False
        sc100 = False
        sc200 = False
        if score >= 25:
            sc25 = True
        if score >= 50:
            sc50 = True
        if score >= 100:
            sc100 = True
        if score >= 200:
            sc200 = True

        if move_direction == 'UP':
            danger_straight = (head_y - 10 < 0) or ([head_x, head_y - 10] in snake_body)
            danger_right = (head_x + 10 >= window_x) or ([head_x + 10, head_y] in snake_body)
            danger_left = (head_x - 10 < 0) or ([head_x - 10, head_y] in snake_body)
        elif move_direction == 'DOWN':
            danger_straight = (head_y + 10 >= window_y) or ([head_x, head_y + 10] in snake_body)
            danger_right = (head_x - 10 < 0) or ([head_x - 10, head_y] in snake_body)
            danger_left = (head_x + 10 >= window_x) or ([head_x + 10, head_y] in snake_body)
        elif move_direction == 'LEFT':
            danger_straight = (head_x - 10 < 0) or ([head_x - 10, head_y] in snake_body)
            danger_right = (head_y + 10 >= window_y) or ([head_x, head_y + 10] in snake_body)
            danger_left = (head_y - 10 < 0) or ([head_x, head_y - 10] in snake_body)
        elif move_direction == 'RIGHT':
            danger_straight = (head_x + 10 >= window_x) or ([head_x + 10, head_y] in snake_body)
            danger_right = (head_y - 10 < 0) or ([head_x, head_y - 10] in snake_body)
            danger_left = (head_y + 10 >= window_y) or ([head_x, head_y + 10] in snake_body)

        state = [
            danger_straight,
            danger_left,
            danger_right,

            move_direction == 'UP',
            move_direction == 'DOWN',
            move_direction == 'LEFT',
            move_direction == 'RIGHT',

            fruit_x < head_x,
            fruit_x > head_x,
            fruit_y < head_y,
            fruit_y > head_y,

            sc25,
            sc50,
            sc100,
            sc200,
        ]

        return state

    def get_action(self,state, n_games):
        self.epsilon = 80 - n_games
        action = [0,0,0]

        if self.epsilon > random.randint(0,200):
            move = random.randint(0,2)
            action[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            pred = self.model(state_tensor)
            move = torch.argmax(pred)
            action[move] = 1

        return action

    def load_model(self, file_name='model.pth'):
        self.model.load_state_dict(torch.load(file_name))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) < 32:
            mini_sample = self.memory
        else:
            mini_sample = random.sample(self.memory, BATCH_SIZE)

        state, action, reward, next_state, done = zip(*mini_sample)
        self.trainer.train_step(state, action, reward, next_state, done)
