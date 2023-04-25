import gym
import numpy as np
import torch
from DQN import DQN

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    env.action_space.seed(42)
    d = DQN(4, 2, 128, .99)

    EPOCHS = 1000
    EPS: float = .3
    UPDATE_TIME = 20


    time = 0
    for epoch in range(EPOCHS):
        total_reward = 0
        state = env.reset()
        state = torch.Tensor(state[0])
        while (True):
            if np.random.rand() < EPS:
                action = np.random.randint(2)
            else:
                action = d.GenernateAction(state)

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.Tensor(next_state)
            total_reward += reward

            d.Learn(state, action, next_state, int(reward), done)
            state = next_state

            EPS *= .95
            EPS = min(EPS, .1)

            time += 1
            if(time % UPDATE_TIME == 0):
                d.updateWeights()

            if done:
                print(f"{epoch} Epoch total reward {total_reward}")
                break

    env.close()

if __name__ == "__main__":
    main()

