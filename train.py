#!/usr/bin/env python3
# to run: python train.py env=cheetah_run
import numpy as np
import torch
import os
import sys
import time

sys.path.append(os.path.realpath(os.path.dirname(__file__)))
from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
from utils import setSeedEverywhere, evalMode
# import utils

import hydra

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        setSeedEverywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        # self.env = utils.makeEnv(cfg)
        self.env = hydra.utils.call(cfg.env)

        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        cfg.agent.n_step = cfg.replay_buffer.n_step # n-step experience replay
        self.agent = hydra.utils.instantiate(cfg.agent,_recursive_=False)

        self.replay_buffer = ReplayBuffer(
            capacity=cfg.replay_buffer.capacity,
            obs_shape = self.env.observation_space.shape,
            action_shape = self.env.action_space.shape,
            obs_dtype = self.env.observation_space.dtype,
            action_dtype = self.env.action_space.dtype,
            n_step = cfg.replay_buffer.n_step, # n-step experience replay
            discount=cfg.agent.discount, # per step discount
            device = self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with evalMode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.dump(self.step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        num_train_steps = self.cfg.num_train_steps # total training steps
        num_seed_steps = self.cfg.num_seed_steps # steps prior to training
        env = self.env
        while self.step < num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > num_seed_steps))
                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                self.logger.log('train/episode_reward', episode_reward,self.step)
                self.logger.log('train/episode', episode, self.step)

                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                
                self.agent.reset()
                obs = env.reset()
                self.replay_buffer.onEpisodeEnd()

            # sample action for data collection
            if self.step < num_seed_steps:
                action = env.action_space.sample()
            else:
                with evalMode(self.agent):
                    action = self.agent.act(obs, sample=True)
            # run training update
            if self.step >= num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step) 

            next_obs, reward, done, _ = env.step(action)

            max_episode_step_reached = (episode_step + 1 == env._max_episode_steps)
            not_done = True if max_episode_step_reached else (not done) # allow infinite bootstrap
            done = done or max_episode_step_reached # signals episode ended
            self.replay_buffer.add(obs, action, reward, next_obs, not_done)
            
            obs = next_obs
            episode_step += 1
            self.step += 1
            episode_reward += reward


@hydra.main(config_path="config",config_name='train')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
