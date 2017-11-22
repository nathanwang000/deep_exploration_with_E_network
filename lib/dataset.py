from collections import namedtuple
import random, gym
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch
from lib.setting import *


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class CartPoleVision:
    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.screen_width = 600

    def get_cart_location(self):
        world_width = self.env.x_threshold * 2
        scale = self.screen_width / world_width
        return int(self.env.state[0] * scale + self.screen_width / 2.0)  # MIDDLE OF CART

    def reset(self):
        self.env.reset()
        self.last_screen = self.get_screen()
        self.current_screen = self.get_screen()
        state = self.current_screen - self.last_screen
        return state

    def step(self, action):
        _, reward, done, info = self.env.step(action)

        # Observe new state
        self.last_screen = self.current_screen
        self.current_screen = self.get_screen()
        if not done:
            next_state = self.current_screen - self.last_screen
        else:
            next_state = None
        return next_state, reward, done, info

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

    ########## environment specific helpers #############
    
    def get_screen(self):

        resize = T.Compose([T.ToPILImage(),
                            T.Scale(40, interpolation=Image.CUBIC),
                            T.ToTensor()])
        
        screen = self.env.render(mode='rgb_array').transpose(
            (2, 0, 1))  # transpose into torch order (CHW)
        # Strip off the top and bottom of the screen
        screen = screen[:, 160:320]
        view_width = 320
        cart_location = self.get_cart_location()
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (self.screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescare, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).type(Tensor)
