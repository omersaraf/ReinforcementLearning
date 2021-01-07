import argparse
import os
import random
import time
from collections import deque

import gym
import keras
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

# based on: https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/
# Repository: https://github.com/udacity/deep-reinforcement-learning

# ENV_NAME = 'MountainCar-v0'
# ENV_NAME = 'LunarLander-v2'
ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.1
LEARNING_RATE_DECAY = 0.01
MEMORY_SIZE = 10000
BATCH_SIZE = 64
FRAME_RATE = 0.00
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.95

EXPLORATION = EXPLORATION_MAX


class Policy:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

    def act(self, state, exploration):
        raise NotImplementedError

    def remember(self, state, action, reward, state_next, terminal):
        raise NotImplementedError

    def process_state(self, state):
        return state

    def learn(self):
        raise NotImplementedError

    def play(self, exploration, render=True):
        state = self.process_state(self.env.reset())
        if render:
            env.render()
        done = False
        score = 0
        actions = []
        while not done:
            action = self.act(state, exploration)
            actions.append(action)
            next_state, reward, done, _ = self.env.step(action)
            if render:
                time.sleep(FRAME_RATE)
                env.render()
            next_state = self.process_state(next_state)
            self.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward
        if not PLAY_MODE:
            self.learn()
        return score


class DQNAgent(Policy):
    def __init__(self, env):
        super().__init__(env)
        self.exploration_rate = EXPLORATION_MAX
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.should_learn = True

        if os.path.exists(MODEL_PATH):
            self.model = keras.models.load_model(MODEL_PATH)
            print('Using predefined model')
        else:
            self.model = Sequential()
            self.model.add(Dense(24, input_dim=self.observation_space, activation='tanh'))
            self.model.add(Dense(48, activation='tanh'))
            self.model.add(Dense(self.action_space, activation='linear'))
            self.model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE, decay=LEARNING_RATE_DECAY))
            print('Creating new model')
        self.iterations = 1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, exploration_rate):
        if np.random.rand() < exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def learn(self):
        states_batch, q_values_batch = [], []
        mini_batch = random.sample(
            self.memory, min(len(self.memory), BATCH_SIZE))
        for state, action, reward, next_state, done in mini_batch:
            q_values = self.model.predict(state)
            q_values[0][action] = reward if done else reward + GAMMA * np.max(self.model.predict(next_state)[0])
            states_batch.append(state[0])
            q_values_batch.append(q_values[0])

        self.model.fit(np.array(states_batch), np.array(q_values_batch), batch_size=len(states_batch), verbose=0)

        if self.iterations % 10 == 0:
            print(f'Saving model to {MODEL_PATH}')
            self.model.save(MODEL_PATH)

        self.iterations += 1

    def process_state(self, state):
        return np.reshape(state, [1, self.observation_space])


class RandomAgent(Policy):
    def __init__(self, env):
        super().__init__(env)
        self.memory = []

    def act(self, state, exploration):
        return random.randrange(self.action_space)

    def remember(self, state, action, reward, state_next, terminal):
        self.memory.append((state, action, reward, state_next, terminal))

    def learn(self):
        pass

    def stats(self) -> dict:
        return {}


run = 1
scores = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--play-mode', dest='play_mode', default=False, type=bool)
    parser.add_argument('--exploration', dest='exploration', default=1, type=float)
    parser.add_argument('--env', dest='env', default=ENV_NAME, type=str)

    args = parser.parse_args()
    ENV_NAME = args.env
    EXPLORATION = args.exploration
    PLAY_MODE = args.play_mode
    MODEL_PATH = f'{ENV_NAME}_model'

    if PLAY_MODE:
        print('Playing for fun!')
    else:
        print('Learning for fun!')

    print(f'Environment: {ENV_NAME}, exploration: {EXPLORATION}')
    env = gym.make(ENV_NAME)
    policy = DQNAgent(env)
    while True:
        score = policy.play(EXPLORATION)
        scores.append(score)
        print(f"Runs: {run}, exploration: {EXPLORATION}, score: {score} (AVG: {np.mean(scores)})")
        EXPLORATION *= EXPLORATION_DECAY
        EXPLORATION = max(EXPLORATION, EXPLORATION_MIN)
        run += 1
