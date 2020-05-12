#!/usr/bin/env python
# coding: utf-8

# In[15]:


# https://www.endpoint.com/blog/2018/08/29/self-driving-toy-car-using-the-a3c-algorithm


# In[16]:


import numpy as np
import torch
import gym
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.nn import functional as F
import random
import collections
from collections import deque
import multiprocessing as mp


# uniform = [0.25, 0.25, 0.25, 0.25]
# more confident = [0.5, 0.25, 0.15, 0.10]
# over confident = [0.95, 0.01, 0.01, 0.03]
# super over confident = [0.99, 0.003, 0.004, 0.003]
# 
# y(x) = x*log(x, 10)
# 
# entropy(dist) = -sum(map(y, dist))
# 
# entropy (uniform) => 0.6021
# entropy (more confident) => 0.5246
# entropy (over confident) => 0.1068
# entropy (super over confident) => 0.0291

# In[17]:


class Agent(nn.Module):
    def __init__(self, **kwargs):
        super(Agent, self).__init__(**kwargs)

        self.init_args = kwargs

        self.h = torch.zeros(1, 256)

        self.norm1 = nn.BatchNorm2d(4)
        self.norm2 = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(4, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.gru = nn.GRUCell(1152, 256)
        self.policy = nn.Linear(256, 4)
        self.value = nn.Linear(256, 1)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()

        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.conv1.bias, 0.01)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.conv2.bias, 0.01)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.conv3.bias, 0.01)
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.conv4.bias, 0.01)
        nn.init.constant_(self.gru.bias_ih, 0)
        nn.init.constant_(self.gru.bias_hh, 0)
        nn.init.xavier_uniform_(self.policy.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.policy.bias, 0.01)
        nn.init.xavier_uniform_(self.value.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.value.bias, 0.01)

        self.train()

    def reset(self):
        self.h = torch.zeros(1, 256)

    def clone(self, num=1):
        return [ self.clone_one() for _ in range(num) ]

    def clone_one(self):
        return Agent(**self.init_args)

    def forward(self, state):
        state = state.view(1, 4, 96, 96)
        state = self.norm1(state)

        data = self.prelu1(self.conv1(state))
        data = self.prelu2(self.conv2(data))
        data = self.prelu3(self.conv3(data))
        data = self.prelu4(self.conv4(data))

        data = self.norm2(data)
        data = data.view(1, -1)

        h = self.gru(data, self.h)
        self.h = h.detach()

        pre_policy = h.view(-1)

        policy = F.softmax(self.policy(pre_policy))
        value = self.value(pre_policy)

        return policy, value


# In[18]:


class Runner:
    def __init__(self, agent, ix, train = True, **kwargs):
        self.agent = agent
        self.train = train
        self.ix = ix
        self.reset = False
        self.states = []

        # each runner has its own environment:
        self.env = gym.make('CarRacing-v0')

    def get_value(self):
        """
        Returns just the current state's value.
        This is used when approximating the R.
        If the last step was
        not terminal, then we're substituting the "r"
        with V(s) - hence, we need a way to just
        get that V(s) without moving forward yet.
        """
        _input = self.preprocess(self.states)
        _, _, _, value = self.decide(_input)
        return value

    def run_episode(self, yield_every = 10, do_render = False):
        """
        The episode runner written in the generator style.
        This is meant to be used in a "for (...) in run_episode(...):" manner.
        Each value generated is a tuple of:
        step_ix: the current "step" number
        rewards: the list of rewards as received from the environment (without discounting yet)
        values: the list of V(s) values, as predicted by the "critic"
        policies: the list of policies as received from the "actor"
        actions: the list of actions as sampled based on policies
        terminal: whether we're in a "terminal" state
        """
        self.reset = False
        step_ix = 0

        rewards, values, policies, actions = [[], [], [], []]

        self.env.reset()

        # we're going to feed the last 4 frames to the neural network that acts as the "actor-critic" duo. We'll use the "deque" to efficiently drop too old frames always keeping its length at 4:
        states = deque([ ])

        # we're pre-populating the states deque by taking first 4 steps as "full throttle forward":
        while len(states) < 4:
            _, r, _, _ = self.env.step([0.0, 1.0, 0.0])
            state = self.env.render(mode='rgb_array')
            states.append(state)
            logger.info('Init reward ' + str(r) )

        # we need to repeat the following as long as the game is not over yet:
        while True:
            # the frames need to be preprocessed (I'm explaining the reasons later in the article)
            _input = self.preprocess(states)

            # asking the neural network for the policy and value predictions:
            action, action_ix, policy, value = self.decide(_input, step_ix)

            # taking the step and receiving the reward along with info if the game is over:
            _, reward, terminal, _ = self.env.step(action)

            # explicitly rendering the scene (again, this will be explained later)
            state = self.env.render(mode='rgb_array')

            # update the last 4 states deque:
            states.append(state)
            while len(states) > 4:
                states.popleft()

            # if we've been asked to render into the window (e. g. to capture the video):
            if do_render:
                self.env.render()

            self.states = states
            step_ix += 1

            rewards.append(reward)
            values.append(value)
            policies.append(policy)
            actions.append(action_ix)

            # periodically save the state's screenshot along with the numerical values in an easy to read way:
            if self.ix == 2 and step_ix % 200 == 0:
                fname = '.'+os.sep+'screens'+os.sep+'car-racing'+os.sep+'screen-' + str(step_ix) + '-' + str(int(time.time())) + '.jpg'
                im = Image.fromarray(state)
                im.save(fname)
                state.tofile(fname + '.txt', sep=" ")
                _input.numpy().tofile(fname + '.input.txt', sep=" ")

            # if it's game over or we hit the "yield every" value, yield the values from this generator:
            if terminal or step_ix % yield_every == 0:
                yield step_ix, rewards, values, policies, actions, terminal
                rewards, values, policies, actions = [[], [], [], []]

            # following is a very tacky way to allow external using code to mark that it wants us to reset the environment, finishing the episode prematurely. (this would be hugely refactored in the production code but for the sake of playing with the algorithm itself, it's good enough):
            if self.reset:
                self.reset = False
                self.agent.reset()
                states = deque([ ])
                self.states = deque([ ])
                return

            if terminal:
                self.agent.reset()
                states = deque([ ])
                return
    def ask_reset(self):
        self.reset = True

    def preprocess(self, states):
        return torch.stack([ torch.tensor(self.preprocess_one(image_data), dtype=torch.float32) for image_data in states ])

    def preprocess_one(self, image):
        """
        Scales the rendered image and makes it grayscale
        """
        return rescale(rgb2gray(image), (0.24, 0.16), anti_aliasing=False, mode='edge', multichannel=False)

    def choose_action(self, policy, step_ix):
        """
        Chooses an action to take based on the policy and whether we're in the training mode or not. During training, it samples based on the probability values in the policy. During the evaluation, it takes the most probable action in a greedy way.
        """
        policies = [[-0.8, 0.0, 0.0], [0.8, 0.0, 0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.6]]

        if self.train:
            action_ix = np.random.choice(4, 1, p=torch.tensor(policy).detach().numpy())[0]
        else:
            action_ix = np.argmax(torch.tensor(policy).detach().numpy())

        logger.info('Step ' + str(step_ix) + ' Runner ' + str(self.ix) + ' Action ix: ' + str(action_ix) + ' From: ' + str(policy))

        return np.array(policies[action_ix], dtype=np.float32), action_ix

    def decide(self, state, step_ix = 999):
        policy, value = self.agent(state)

        action, action_ix = self.choose_action(policy, step_ix)

        return action, action_ix, policy, value

    def load_state_dict(self, state):
        """
        As we'll have multiple "worker" runners, they will need to be able to sync their agents' weights with the main agent.
        This function loads the weights into this runner's agent.
        """
        self.agent.load_state_dict(state)


# In[19]:


class Trainer:
    def __init__(self, gamma, agent, window = 15, workers = 8, **kwargs):
        super().__init__(**kwargs)

        self.agent = agent
        self.window = window
        self.gamma = gamma
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-4)
        self.workers = workers

        # even though we're loading the weights into worker agents explicitly, I found that still without sharing the weights as following, the algorithm was not converging:
        self.agent.share_memory()

    def fit(self, episodes = 1000):
#            """
#            The higher level method for training the agents.
#            It called into the lower level "train" which orchestrates the process itself.
#            """
        last_update = 0
        updates = dict()

        for ix in range(1, self.workers + 1):
            updates[ ix ] = { 'episode': 0, 'step': 0, 'rewards': deque(), 'losses': deque(), 'points': 0, 'mean_reward': 0, 'mean_loss': 0 }

        for update in self.train(episodes):
            now = time.time()

            # you could do something useful here with the updates dict.
            # I've opted out as I'm using logging anyways and got more value in just watching the log file, grepping for the desired values

            # save the current model's weights every minute:
            if now - last_update > 60:
                torch.save(self.agent.state_dict(), './checkpoints/car-racing/' + str(int(now)) + '-.pytorch')
                last_update = now

    def train(self, episodes = 1000):
        """
        Lower level training orchestration method. Written in the generator style. Intended to be used with "for update in train(...):"
        """

        # create the requested number of background agents and runners:
        worker_agents = self.agent.clone(num = self.workers)
        runners = [ Runner(agent=agent, ix = ix + 1, train = True) for ix, agent in enumerate(worker_agents) ]

        # we're going to communicate the workers' updates via the thread safe queue:
        queue = mp.SimpleQueue()

        # if we've not been given a number of episodes: assume the process is going to be interrupted with the keyboard interrupt once the user (us) decides so:
        if episodes is None:
            print('Starting out an infinite training process')

        # create the actual background processes, making their entry be the train_one method:
        processes = [ mp.Process(target=self.train_one, args=(runners[ix - 1], queue, episodes, ix)) for ix in range(1, self.workers + 1) ]

        # run those processes:
        for process in processes:
            process.start()

        try:
            # what follows is a rather naive implementation of listening to workers updates. it works though for our purposes:
            while any([ process.is_alive() for process in processes ]):
                results = queue.get()
                yield results
        except Exception as e:
            logger.error(str(e))

    def train_one(self, runner, queue, episodes = 1000, ix = 1):
        """
        Orchestrate the training for a single worker runner and agent. This is intended to run in its own background process.
        """

        # possibly naive way of trying to de-correlate the weight updates further (I have no hard evidence to prove if it works, other than my subjective observation):
        time.sleep(ix)

        try:
            # we are going to request the episode be reset whenever our agent scores lower than its max points. the same will happen if the agent scores total of -10 points:
            max_points = 0
            max_eval_points = 0
            min_points = 0
            max_episode = 0

            for episode_ix in itertools.count(start=0, step=1):

                if episodes is not None and episode_ix >= episodes:
                    return

                max_episode_points = 0
                points = 0

                # load up the newest weights every new episode:
                runner.load_state_dict(self.agent.state_dict())

                # every 5 episodes lets evaluate the weights we've learned so far by recording the run of the car using the greedy strategy:
                if ix == 1 and episode_ix % 5 == 0:
                    eval_points = self.record_greedy(episode_ix)

                    if eval_points > max_eval_points:
                        torch.save(runner.agent.state_dict(), './checkpoints/car-racing/' + str(eval_points) + '-eval-points.pytorch')
                        max_eval_points = eval_points

                # each n-step window, compute the gradients and apply
                # also: decide if we shouldn't restart the episode if we don't want to explore too much of the not-useful state space:
                for step, rewards, values, policies, action_ixs, terminal in runner.run_episode(yield_every=self.window):
                    points += sum(rewards)

                    if ix == 1 and points > max_points:
                        torch.save(runner.agent.state_dict(), './checkpoints/car-racing/' + str(points) + '-points.pytorch')
                        max_points = points

                    if ix == 1 and episode_ix > max_episode:
                        torch.save(runner.agent.state_dict(), './checkpoints/car-racing/' + str(episode_ix) + '-episode.pytorch')
                        max_episode = episode_ix

                    if points < -10 or (max_episode_points > min_points and points < min_points):
                        terminal = True
                        max_episode_points = 0
                        point = 0
                        runner.ask_reset()

                    if terminal:
                        logger.info('TERMINAL for ' + str(ix) + ' at step ' + str(step) + ' with total points ' + str(points) + ' max: ' + str(max_episode_points) )

                    # if we're learning, then compute and apply the gradients and load the newest weights:
                    if runner.train:
                        loss = self.apply_gradients(policies, action_ixs, rewards, values, terminal, runner)
                        runner.load_state_dict(self.agent.state_dict())

                    max_episode_points = max(max_episode_points, points)
                    min_points = max(min_points, points)

                    # communicate the gathered values to the main process:
                    queue.put((ix, episode_ix, step, rewards, loss, points, terminal))

        except Exception as e:
            string = traceback.format_exc()
            logger.error(str(e) + ' → ' + string)
            queue.put((ix, -1, -1, [-1], -1, str(e) + '<br />' + string, True))

    def record_greedy(self, episode_ix):
        """
        Records the video of the "greedy" run based on the current weights.
        """
        directory = './videos/car-racing/episode-' + str(episode_ix) + '-' + str(int(time.time()))
        player = Player(agent=self.agent, directory=directory, train=False)
        points = player.play()
        logger.info('Evaluation at episode ' + str(episode_ix) + ': ' + str(points) + ' points (' + directory + ')')
        return points

    def apply_gradients(self, policies, actions, rewards, values, terminal, runner):
        worker_agent = runner.agent
        actions_one_hot = torch.tensor([[ int(i == action) for i in range(4) ] for action in actions], dtype=torch.float32)

        policies = torch.stack(policies)
        values = torch.cat(values)
        values_nograd = torch.zeros_like(values.detach(), requires_grad=False)
        values_nograd.copy_(values)

        discounted_rewards = self.discount_rewards(runner, rewards, values_nograd[-1], terminal)
        advantages = discounted_rewards - values_nograd

        logger.info('Runner ' + str(runner.ix) + 'Rewards: ' + str(rewards))
        logger.info('Runner ' + str(runner.ix) + 'Discounted Rewards: ' + str(discounted_rewards.numpy()))

        log_policies = torch.log(0.00000001 + policies)

        one_log_policies = torch.sum(log_policies * actions_one_hot, dim=1)

        entropy = torch.sum(policies * -log_policies)

        policy_loss = -torch.mean(one_log_policies * advantages)

        value_loss = F.mse_loss(values, discounted_rewards)

        value_loss_nograd = torch.zeros_like(value_loss)
        value_loss_nograd.copy_(value_loss)

        policy_loss_nograd = torch.zeros_like(policy_loss)
        policy_loss_nograd.copy_(policy_loss)

        logger.info('Value Loss: ' + str(float(value_loss_nograd)) + ' Policy Loss: ' + str(float(policy_loss_nograd)))

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        self.agent.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(worker_agent.parameters(), 40)

        # the following step is crucial. at this point, all the info about the gradients reside in the worker agent's memory. We need to "move" those gradients into the main agent's memory:
        self.share_gradients(worker_agent)

        # update the weights with the computed gradients:
        self.optimizer.step()

        worker_agent.zero_grad()
        return float(loss.detach())

    def share_gradients(self, worker_agent):
        for param, shared_param in zip(worker_agent.parameters(), self.agent.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def clip_reward(self, reward):
        """
        Clips the rewards into the <-3, 3> range preventing too big of the gradients variance.
        """
        return max(min(reward, 3), -3)

    def discount_rewards(self, runner, rewards, last_value, terminal):
        discounted_rewards = [0 for _ in rewards]
        loop_rewards = [ self.clip_reward(reward) for reward in rewards ]

        if terminal:
            loop_rewards.append(0)
        else:
            loop_rewards.append(runner.get_value())

        for main_ix in range(len(discounted_rewards) - 1, -1, -1):
            for inside_ix in range(len(loop_rewards) - 1, -1, -1):
                if inside_ix >= main_ix:
                    reward = loop_rewards[inside_ix]
                    discounted_rewards[main_ix] += self.gamma**(inside_ix - main_ix) * reward

        return torch.tensor(discounted_rewards)
    


# In[20]:


class Player(Runner):
    def __init__(self, directory, **kwargs):
        super().__init__(ix=999, **kwargs)

        self.env = Monitor(self.env, directory)

    def play(self):
        points = 0
        for step, rewards, values, policies, actions, terminal in self.run_episode(yield_every = 1, do_render = True):
            points += sum(rewards)
        self.env.close()
        return points


# In[ ]:





# In[21]:


if __name__ == "__main__":
    agent = Agent()

    trainer = Trainer(gamma = 0.99, agent = agent)
    trainer.fit(episodes=None)
    


# In[ ]:




