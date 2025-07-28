import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, h=None):
        # x: (B, 1, D)
        out, h = self.encoder(x, h)
        value = self.value_head(out[:, -1])  # (B, 1)
        return value.squeeze(-1), h  # (B,), h


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.entropies = []

    def add(self, state, action, reward, log_prob, value, entropy):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)

    def clear(self):
        self.__init__()


class PPOAgent:
    def __init__(self, policy_net, value_net, policy_optimizer, value_optimizer,
                 clip_param=0.2, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5):
        self.policy_net = policy_net
        self.value_net = value_net
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def compute_returns_and_advantages(self, rewards, values, gamma=0.99):
        returns = []
        advs = []
        G = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns).to(values[0].device)
        values = torch.stack(values)
        advantages = returns - values.detach()
        return returns, advantages

    # def update(self, h, buffer: RolloutBuffer):
    #     log_probs_old = torch.stack(buffer.log_probs).detach()
    #     actions = torch.tensor(buffer.actions).to(log_probs_old.device)
    #     states = torch.stack(buffer.states).detach()
    #     values = buffer.values
    #     rewards = buffer.rewards

    #     returns, advantages = self.compute_returns_and_advantages(rewards, values)

    #     # Policy update
    #     new_log_probs = []
    #     entropies = []
    #     for state, action in zip(states, actions):
    #         probs, _, _ = self.policy_net(state.unsqueeze(0), None)
    #         dist = Categorical(probs)
    #         new_log_prob = dist.log_prob(action)
    #         entropy = dist.entropy()
    #         new_log_probs.append(new_log_prob)
    #         entropies.append(entropy)

    #     new_log_probs = torch.stack(new_log_probs)
    #     entropies = torch.stack(entropies)

    #     ratio = torch.exp(new_log_probs - log_probs_old)
    #     surr1 = ratio * advantages
    #     surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
    #     policy_loss = -torch.min(surr1, surr2).mean()
    #     entropy_bonus = entropies.mean()

    #     self.policy_optimizer.zero_grad()
    #     (policy_loss - self.entropy_coef * entropy_bonus).backward()
    #     nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
    #     self.policy_optimizer.step()

    #     # Value update
    #     # value_preds = []
    #     # for state in states:
    #     #     value_pred, h = self.value_net(state.unsqueeze(0), h)
    #     #     value_preds.append(value_pred)
    #     # value_preds = torch.stack(value_preds)
    #     value_preds = torch.stack(values)
    #     value_loss = F.mse_loss(value_preds, returns)

    #     self.value_optimizer.zero_grad()
    #     (value_loss * self.value_coef).backward()
    #     nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
    #     self.value_optimizer.step()

    def update(self, buffer: RolloutBuffer):
        log_probs = torch.stack(buffer.log_probs).detach()
        actions = torch.tensor(buffer.actions).to(log_probs.device)
        states = torch.stack(buffer.states).detach()
        entropys = torch.stack(buffer.entropies).detach()
        values = buffer.values
        rewards = buffer.rewards

        returns, advantages = self.compute_returns_and_advantages(rewards, values)

        # Policy update
        # entropies = torch.stack(entropies)

        policy_loss =  -(log_probs * advantages).mean()
        # entropy_bonus = entropies.mean()

        self.policy_optimizer.zero_grad()
        # (policy_loss - self.entropy_coef * entropy_bonus).backward()
        (policy_loss).backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # Value update
        value_preds = torch.stack(values)
        value_loss = F.mse_loss(value_preds, returns)

        self.value_optimizer.zero_grad()
        (value_loss * self.value_coef).backward()
        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

