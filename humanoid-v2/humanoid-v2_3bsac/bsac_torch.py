
'''
Author: Qin Yang
05/06/2022
'''

import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork1, ActorNetwork2, CriticNetwork, ValueNetwork

general_device = T.device("cuda" if T.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[17],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor_1 = ActorNetwork1(alpha, input_dims, n_actions=3,
                    name='actor1', max_action=[0.4])
        self.actor_2 = ActorNetwork2(alpha, input_dims, n_actions=8,
                    name='actor2', max_action=[0.4])
        self.actor_3 = ActorNetwork2(alpha, input_dims, n_actions=6,
                    name='actor3', max_action=[0.4])
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(general_device)
        action_1, _ = self.actor_1.sample_normal(state, reparameterize=False)
        action_2, _ = self.actor_2.sample_normal(state, action_1, reparameterize=False)
        action_3, _ = self.actor_3.sample_normal(state, action_1, reparameterize=False)

        actions = T.cat([action_1, action_3, action_2], dim=1)

        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor_1.save_checkpoint()
        self.actor_2.save_checkpoint()
        self.actor_3.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor_1.load_checkpoint()
        self.actor_2.load_checkpoint()
        self.actor_3.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(general_device)
        done = T.tensor(done).to(general_device)
        state_ = T.tensor(new_state, dtype=T.float).to(general_device)
        state = T.tensor(state, dtype=T.float).to(general_device)
        action = T.tensor(action, dtype=T.float).to(general_device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        action_1, log_probs_1 = self.actor_1.sample_normal(state, reparameterize=False)
        action_2, log_probs_2 = self.actor_2.sample_normal(state, action_1, reparameterize=False)
        action_3, log_probs_3 = self.actor_3.sample_normal(state, action_1, reparameterize=False)
        actions = T.cat((action_1, action_3, action_2), 1).to(general_device)

        log_probs_1 = log_probs_1.view(-1)
        log_probs_2 = log_probs_2.view(-1)
        log_probs_3 = log_probs_3.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target_1 = critic_value - log_probs_1
        value_loss_1 = 0.5 * F.mse_loss(value, value_target_1)
        value_target_2 = critic_value - log_probs_2
        value_loss_2 = 0.5 * F.mse_loss(value, value_target_2)
        value_target_3 = critic_value - log_probs_3
        value_loss_3 = 0.5 * F.mse_loss(value, value_target_3)
        value_loss = (value_loss_1 + value_loss_2 + value_loss_3) / 3
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        action_1, log_probs_1 = self.actor_1.sample_normal(state, reparameterize=True)
        action_2, log_probs_2 = self.actor_2.sample_normal(state, action_1, reparameterize=True)
        action_3, log_probs_3 = self.actor_3.sample_normal(state, action_1, reparameterize=True)
        actions = T.cat((action_1, action_3, action_2), 1).to(general_device)

        log_probs_1 = log_probs_1.view(-1)
        log_probs_2 = log_probs_2.view(-1)
        log_probs_3 = log_probs_3.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        
        actor_loss = (log_probs_1 + log_probs_2 + log_probs_3) / 3 - critic_value

        actor_loss = T.mean(actor_loss)
        self.actor_1.optimizer.zero_grad()
        self.actor_2.optimizer.zero_grad()
        self.actor_3.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_1.optimizer.step()
        self.actor_2.optimizer.step()
        self.actor_3.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss

        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()