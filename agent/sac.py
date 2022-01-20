import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agent import Agent
import utils

import hydra


class SACAgent(Agent):
    """SAC algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount,n_step, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature,name="sac"):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        # self.discount = discount
        self.discount = discount**n_step
        # self.n_step = n_step # n_step experience replay
        # self.n_discount = discount**n_step
        self.critic_tau = critic_tau # soft update: target_net = target_net*(1-tau) + net*tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature # bool, whether alpha is learnable

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature),device=self.device,requires_grad=True)
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.AdamW([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        # set training mode
        self.train()
        # self.critic_target.train()# TODO is this necessary?

        # step for deciding wether to update
        self.actor_update_step = 0 
        self.critic_target_update_step = 0
        
        
    def save(self,path):
        """save model checkpoint"""
        torch.save({
            "log_alpha":self.log_alpha,
            "critic":self.critic.state_dict(),
            "critic_target":self.critic_target.state_dict(),
            "actor":self.actor.state_dict(),
            "actor_optimizer":self.actor_optimizer.state_dict(),
            "critic_optimizer":self.critic_optimizer.state_dict(),
            "log_alpha_optimizer":self.log_alpha_optimizer.state_dict()
        },path)
        
    def load(self,path):
        """load saved model"""
        chpt  = torch.load(path,map_location=self.device) # loaded checkpoint
        self.log_alpha = chpt["log_alpha"]
        self.critic.load_state_dict(chpt["critic"])
        self.critic_target.load_state_dict(chpt["critic_target"])
        self.actor.load_state_dict(chpt["actor"])
        self.actor_optimizer.load_state_dict(chpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(chpt["critic_optimizer"])
        self.log_alpha_optimizer.load_state_dict(chpt["log_alpha_optimizer"])
        # print("\n\nactor:\n",self.actor)
        # print("\n\ncritic:\n",self.critic)
        
    def train(self, training=True):
        """set training mode for pytorch.nn.Module"""
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self): # entropy multiplier
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.tensor(obs, dtype=torch.float, device=self.device).unsqueeze(0) 
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        # assert action.ndim == 2 and action.shape[0] == 1 #  TODO:Check this
        # return utils.toNumpy(action[0])
        return action[0].detach().cpu().numpy()

    def updateCritic(self, obs, action, reward, next_obs, not_done, logger,
                      step):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True) # sum log_prob -> multiply prob
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step) # TODO

    def updateActorAndAlpha(self, obs, logger, step):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step) #TODO

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.updateCritic(obs, action, reward, next_obs, not_done,
                           logger, step)

        # if step % self.actor_update_frequency == 0:
        if self.actor_update_step % self.actor_update_frequency == 0:            
            self.updateActorAndAlpha(obs, logger, step)

        # if step % self.critic_target_update_frequency == 0:
        if self.critic_target_update_step % self.critic_target_update_frequency == 0:
            utils.softUpdateParams(self.critic, self.critic_target,self.critic_tau)

        # counter ++
        self.actor_update_step+=1
        self.critic_target_update_step+=1