import torch
import torch.nn as nn
import torch.optim as optim


class SnakeNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SnakeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 600)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)



class Trainer:
    def __init__(self, model, lr=0.001, gamma=0.8):
        self.model = model
        self.learning_rate = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done, )

        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
