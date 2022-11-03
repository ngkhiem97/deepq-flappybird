import numpy as np
import learning.model as model
import torch
import learning.env as env
from collections import deque
import random
from tqdm import tqdm

MAX_MEMORY = 50000
N_OBSERVATIONS = 500
DECAY_EPSILON = 0.9999
SATURATED_EPSILON = 0.0001
BATCH_SIZE = 32

class QAgent:
    def __init__(self, env: env.Environment, gamma=0.9, epsilon=0.1):
        super()
        # define environment
        self.env = env

        # define the constants
        self.gamma = gamma
        self.epsilon = epsilon

        # define the random generator
        self.rng = np.random.RandomState(0)

        # define the model
        self.q_model = model.ConvNet(env.action_space)
        self.q_model.cuda()
        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=0.001)

        # define the memory
        self.memory = deque()

        # start agent be doing nothing
        self.state_0 = self.do_nothing()

    def do_nothing(self):
        frame_0, reward_0, done = self.env.step([1, 0])
        state_0 = self.init_state(frame_0)
        return state_0

    def init_state(self, frame):
        return np.stack((frame, frame, frame, frame), axis=0)

    def get_state(self, frame, state):
        return np.vstack((state[:3, :, :], frame[None]))

    def act(self, state):
        if self.rng.rand() < self.epsilon:
            index = np.argmax(self.rng.rand(2))
            action = np.zeros(2)
            action[index] = 1
            return action
        else:
            state = state[np.newaxis, :]
            state = torch.from_numpy(state).float().cuda()
            q_values = self.q_model(state)
            q_values = q_values.cpu().detach().numpy()[0]
            index = np.argmax(q_values)
            action = np.zeros(2)
            action[index] = 1
            return action

    def update_memory(self, state_t, action_t, reward_t, state_t_plus1, done):
        self.memory.append((state_t, action_t, reward_t, state_t_plus1, done))
        if len(self.memory) > MAX_MEMORY:
            self.memory.popleft()

    def learn(self, epochs=100, steps=100):
        state_t = self.state_0
        # observe the environment
        print("Observing the environment...")
        for i in tqdm(range(N_OBSERVATIONS)):
            action_t = self.act(state_t)
            frame_t_plus1, reward_t, done_t = self.env.step(action_t)
            state_t_plus1 = self.get_state(frame_t_plus1, state_t)
            self.update_memory(state_t, action_t, reward_t, state_t_plus1, done_t)
            state_t = state_t_plus1

        print("Learning...")
        # learn from the environment
        for epoch in range(epochs):
            epoch_loss = []
            for step in range(steps):
                # perform an action
                action_t = self.act(state_t)
                frame_t_plus1, reward_t, done_t = self.env.step(action_t)
                self.update_memory(state_t, action_t, reward_t, state_t_plus1, done_t)

                # sample a batch from the memory
                minibatch = random.sample(self.memory, BATCH_SIZE)
                state_t_batch = torch.from_numpy(np.array([x[0] for x in minibatch])).float().cuda()
                action_t_batch = torch.from_numpy(np.array([x[1] for x in minibatch])).float().cuda()
                reward_t_batch = np.array([x[2] for x in minibatch])
                state_t_plus1_batch = torch.from_numpy(np.array([x[3] for x in minibatch])).float().cuda()
                done_t_batch = torch.from_numpy(np.array([x[4] for x in minibatch])).float().cuda()

                # calculate the q values for future states
                q_values_t_plus1_batch = self.q_model(state_t_plus1_batch).cpu().detach().numpy()

                # calculate expected reward
                expected_reward = np.array([])
                for i in range(BATCH_SIZE):
                    if done_t_batch[i]:
                        expected_reward = np.append(expected_reward, reward_t_batch[i])
                    else:
                        expected_reward = np.append(expected_reward, reward_t_batch[i] + self.gamma * np.max(q_values_t_plus1_batch[i]))

                # perform gradient descent
                self.optimizer.zero_grad()
                q_values_t_batch = self.q_model(state_t_batch)
                expected_reward = torch.from_numpy(expected_reward).float().cuda()
                readout_action_t_batch = torch.sum(q_values_t_batch * action_t_batch, dim=1)
                loss = torch.nn.functional.mse_loss(readout_action_t_batch, expected_reward)
                loss.backward()
                self.optimizer.step()

                # record the loss
                epoch_loss.append(loss.cpu().detach().numpy())
                
                # decay epsilon
                self.epsilon = max(SATURATED_EPSILON, self.epsilon * DECAY_EPSILON)
            
            # print the loss
            print("Epoch: {}, Loss: {}".format(epoch, np.mean(epoch_loss)))
            
        