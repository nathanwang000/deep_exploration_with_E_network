import torch, copy
from itertools import count
from torch.autograd import Variable
from lib.dataset import ReplayMemory, Transition
from lib.setting import *
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn.externals import joblib
import os

class Trainer:
    def __init__(self, model, env, selection, sarsa=False,
                 run_name='default', plot=False,
                 setting=DefaultSetting(), log_path='logs'):

        print(setting.__dict__)
        self.plot = plot
        self.run_name = run_name
        self.log_path = log_path

        self.sarsa = sarsa # if true, update sarsa, false q-learning
        self.batch_size = setting.batch_size
        self.gamma = setting.gamma_q

        self.memory = ReplayMemory(setting.memory_size)
        self.model = model

        if use_cuda:
            self.model.cuda()
        self.target_model = copy.deepcopy(self.model)

        self.lr = setting.lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        # self.optimizer = optim.RMSprop(self.model.parameters())        
        self.target_update_frequency = setting.target_update_frequency_Q
        self.qnet_update_frequency = setting.qnet_update_frequency

        self.num_episodes = setting.num_episodes
        self.env = env
        self.selection = selection

        self.rewards = []
        self.learning_start = 1000

    def optimize_model(self):

        if len(self.memory) < max(self.batch_size, self.learning_start):
            return
        
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_action)))

        all_zeros = False
        if sum(non_final_mask) == 0: # only when no exploration
            all_zeros = True

            
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.batch_size).type(Tensor))

        if not all_zeros:
            # We don't want to backprop through the expected action values and volatile
            # will save us on temporarily changing the model parameters'
            # requires_grad to False!
            non_final_next_states = Variable(torch.cat([b for a, b in \
                                                        zip(batch.next_action,
                                                            batch.next_state)
                                                        if a is not None]),
                                             volatile=True)
            next_action_batch = Variable(torch.cat([a for a, b in zip(batch.next_action,
                                                                      batch.next_state)
                                                    if a is not None]))    
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
        if self.env.name == 'mountain_car':
            loss =F.mse_loss(state_action_values, expected_state_action_values)
        else:
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        if self.selection.steps_done % self.qnet_update_frequency == 0:
            for param in self.model.parameters():
                param.grad.data.clamp_(-10, 10)
            self.optimizer.step()

        # update target network
        if self.selection.steps_done % self.target_update_frequency == 0:
            self.target_model = copy.deepcopy(self.model)

    def run(self):
        for i_episode in range(self.num_episodes):
            print("episode:", i_episode)
            # Initialize the environment and state
            state = self.env.reset()
            sarsa = None

            rewards = 0
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
                rewards += reward
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
                    self.rewards.append(rewards)
                    import numpy as np
                    import math
                    eps = self.selection.eps_end + (self.selection.eps_start - self.selection.eps_end) * \
                          math.exp(-1. * self.selection.steps_done / self.selection.eps_decay)
                    
                    print(np.mean(self.rewards[-100:]), self.selection.steps_done, eps)
                    joblib.dump(self.rewards,
                                os.path.join(self.log_path, "dqn_%s.pkl" % self.run_name))
                    break

        self.env.render(close=True)
        self.env.close()
        
class DoraTrainer:
    def __init__(self, qnet, enet, env, selection, run_name="default",
                 plot=False, setting=DefaultSetting(), log_path='logs'):

        self.plot = plot
        self.run_name = run_name
        self.env = env
        self.selection = selection
        self.lr = setting.lr
        self.log_path = log_path

        settingQ = setting
        settingE = copy.deepcopy(setting)        
        settingE.target_update_frequency_Q = settingE.target_update_frequency_E
        settingE.qnet_update_frequency = settingE.enet_update_frequency
        settingE.gamma_q = settingE.gamma_e
        
        # no use of selection and env here, because will override run function
        self.qnet_trainer = Trainer(qnet, env, selection, sarsa=False, setting=settingQ)
        self.enet_trainer = Trainer(enet, env, selection, sarsa=True, setting=settingE)

        self.num_episodes = setting.num_episodes

    def run(self):
        for i_episode in range(self.num_episodes):
            # Initialize the environment and state
            state = self.env.reset()
            sarsa = None
            rewards = 0

            print('episode:', i_episode)
            for t in count():
                # Select and perform an action
                Qs = self.qnet_trainer.model(Variable(state, volatile=True).\
                                             type(FloatTensor)).\
                                             data.cpu().numpy().ravel()
                Es = self.enet_trainer.model(Variable(state, volatile=True).\
                                             type(FloatTensor)).\
                                             data.cpu().numpy().ravel()
                
                action = self.selection.select_action(Qs, Es, self.lr)

                # for bridge env
                if self.env.name == 'bridge':
                    if len(self.qnet_trainer.memory) > self.qnet_trainer.batch_size:
                        s_ = int(state[0][0])
                        a_ = action[0][0]
                        Es_a = Es[a_]
                        q_learn = Qs[a_]
                        q_star = self.env.Q_star[s_, a_]
                        if q_star != 0:
                            q_diff = abs((q_learn - q_star) / q_star)
                            EsCounter[(s_,a_)].append([Es_a, q_diff])  

                if sarsa is not None:
                    # Store the transition in memory                    
                    sarsa.append(action)
                    sarsa_dora = copy.deepcopy(sarsa)
                    sarsa_dora[2] = Tensor([0]) # set reward of enet to 0
                    self.qnet_trainer.memory.push(*sarsa)
                    self.enet_trainer.memory.push(*sarsa_dora)     

                if self.plot:
                    self.env.render()
                # reward is 0 for updating evalue
                next_state, reward, done, _ = self.env.step(action[0, 0])
                rewards += reward
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
                    # print("{} updates".format(t))
                    
                    sarsa_dora = copy.deepcopy(sarsa)
                    sarsa_dora[2] = Tensor([0])
                    self.qnet_trainer.memory.push(*sarsa)
                    self.enet_trainer.memory.push(*sarsa_dora)     

                    # report result
                    self.qnet_trainer.rewards.append(rewards)
                    joblib.dump(self.qnet_trainer.rewards,
                                os.path.join(self.log_path,
                                             "dora_%s.pkl" % self.run_name))

                    break


        self.env.render(close=True)
        self.env.close()


        
