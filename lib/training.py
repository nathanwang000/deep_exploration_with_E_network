import torch
from itertools import count
from torch.autograd import Variable
from dataset import ReplayMemory, Transition
from setting import *
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, env, selection):
        self.batch_size = 128
        self.gamma = 0.999

        self.memory = ReplayMemory(10000)
        self.model = model
        if use_cuda:
            self.model.cuda()
        self.optimizer = optim.RMSprop(self.model.parameters())

        self.num_episodes = 20
        self.env = env
        self.selection = selection

        # this is cartpole specific, todo: make this part of environment
        self.episode_durations = []

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                    if s is not None]),
                                         volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def plot_durations(self): # todo: make this part of the env
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(self.episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        
    def run(self):
        for i_episode in range(self.num_episodes):
            # Initialize the environment and state
            state = self.env.reset()
            for t in count():
                # Select and perform an action
                Qs = self.model(Variable(state, volatile=True).type(FloatTensor)).\
                                                          data.cpu().numpy().ravel()
                action = self.selection.select_action(Qs)
                # _, reward, done, _ = env.step(action[0, 0])
                # print('action:', action[0, 0])
                next_state, reward, done, _ = self.env.step(action[0, 0])        
                reward = Tensor([reward])
        
                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

        print('Complete')
        self.env.render(close=True)
        self.env.close()
        plt.ioff()
        plt.show()
