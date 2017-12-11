import torch, copy
from itertools import count
from torch.autograd import Variable
from lib.dataset import ReplayMemory, Transition
from lib.setting import *
import matplotlib
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from sklearn.externals import joblib
import os

class Trainer:
    def __init__(self, model, env, selection, lr=1e-4, sarsa=False,
                 run_name='default', plot=False):

        self.plot = plot
        self.run_name = run_name

        self.sarsa = sarsa # if true, update sarsa, false q-learning
        self.batch_size = 128
        self.gamma = 0.999

        self.memory = ReplayMemory(10000)
        self.model = model

        if use_cuda:
            self.model.cuda()
        self.target_model = copy.deepcopy(self.model)

        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        # self.optimizer = optim.RMSprop(self.model.parameters())        
        self.target_update_frequency = 20
        self.qnet_update_frequency = 1

        self.num_episodes = 30
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
                                              batch.next_action)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([b for a, b in zip(batch.next_action,
                                                                      batch.next_state)
                                                    if a is not None]),
                                         volatile=True)    
        # Variable(torch.cat([s for s in batch.next_state
        #                                             if s is not None]),
        #                                  volatile=True)
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        next_action_batch = Variable(torch.cat([a for a, b in zip(batch.next_action,
                                                                  batch.next_state)
                                                if a is not None]))    

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken

        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.batch_size).type(Tensor))

        # use target network for this transformation
        if self.sarsa:
            next_state_values[non_final_mask] = self.target_model(non_final_next_states)\
                                                    .gather(1, next_action_batch)
        else:
            next_state_values[non_final_mask] = self.target_model(non_final_next_states)\
                                                    .max(1)[0]
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

        if self.selection.steps_done % self.qnet_update_frequency == 0:
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        # update target network
        if self.selection.steps_done % self.target_update_frequency == 0:
            self.target_model = copy.deepcopy(self.model)

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
            sarsa = None
            for t in count():
                # Select and perform an action
                Qs = self.model(Variable(state, volatile=True).type(FloatTensor)).\
                                                          data.cpu().numpy().ravel()
                action = self.selection.select_action(Qs)

                if sarsa is not None:
                    # Store the transition in memory                    
                    sarsa.append(action)
                    self.memory.push(*sarsa) 

                if self.plot:
                    self.env.render()
                next_state, reward, done, _ = self.env.step(action[0, 0])
                reward = Tensor([reward])
        
                # Store next batch of trainsition
                sarsa = [state, action, reward, next_state]

                # Move to the next state
                state = next_state

                # Perform one step of the optimization
                self.optimize_model()
                
                if done:
                    # store last transition to memory
                    sarsa.append(None)
                    self.memory.push(*sarsa) 
                    
                    # report result
                    self.episode_durations.append(t + 1)
                    joblib.dump(self.episode_durations,
                                os.path.join(log_path, "dqn_%s.pkl" % self.run_name))
                    # if self.plot: self.plot_durations()
                    break

        self.env.render(close=True)
        self.env.close()
        # if self.plot:
        #     print('Complete')            
        #     plt.ioff()
        #     plt.show()

        
class DoraTrainer:
    def __init__(self, qnet, enet, env, selection, lr=1e-3, run_name="default",
                 plot=False):

        self.plot = plot
        self.run_name = run_name
        self.env = env
        self.selection = selection
        self.lr = lr
        
        # no use of selection and env here, because will override run function
        self.qnet_trainer = Trainer(qnet, env, selection, lr=lr, sarsa=False)
        self.enet_trainer = Trainer(enet, env, selection, lr=lr, sarsa=True)

        self.num_episodes = 30        

    def run(self):
        for i_episode in range(self.num_episodes):
            # Initialize the environment and state
            state = self.env.reset()
            sarsa = None
            for t in count():
                # Select and perform an action
                Qs = self.qnet_trainer.model(Variable(state, volatile=True).\
                                             type(FloatTensor)).\
                                             data.cpu().numpy().ravel()
                Es = self.enet_trainer.model(Variable(state, volatile=True).\
                                             type(FloatTensor)).\
                                             data.cpu().numpy().ravel()
                
                action = self.selection.select_action(Qs, Es, self.lr)

                if sarsa is not None:
                    # Store the transition in memory                    
                    sarsa.append(action)
                    if sarsa[4] is None and sarsa[3] is not None:
                        print("bad")
                    sarsa_dora = copy.deepcopy(sarsa)
                    sarsa_dora[2] = Tensor([0]) # set reward of enet to 0
                    self.qnet_trainer.memory.push(*sarsa)
                    self.enet_trainer.memory.push(*sarsa_dora)     

                if self.plot:
                    self.env.render()
                # reward is 0 for updating evalue
                next_state, reward, done, _ = self.env.step(action[0, 0]) 
                reward = Tensor([reward])
        
                # Store next transition
                sarsa = [state, action, reward, next_state]

                # Move to the next state
                state = next_state

                # Perform one step of the optimization
                self.qnet_trainer.optimize_model()
                self.enet_trainer.optimize_model()

                if done:
                    # store last transition to memory
                    sarsa.append(None)
                    # if sarsa[3] is not None: print('found it')                    
                    
                    sarsa_dora = copy.deepcopy(sarsa)
                    sarsa_dora[2] = Tensor([0])
                    self.qnet_trainer.memory.push(*sarsa)
                    self.enet_trainer.memory.push(*sarsa_dora)     

                    # report result
                    self.qnet_trainer.episode_durations.append(t + 1)
                    joblib.dump(self.qnet_trainer.episode_durations,
                                os.path.join(log_path, "dora_%s.pkl" % self.run_name))
                    # if self.plot:
                    #     self.qnet_trainer.plot_durations()
                    break


        self.env.render(close=True)
        self.env.close()
        # if self.plot:
        #     print('Complete')            
        #     plt.ioff()
        #     plt.show()

        
