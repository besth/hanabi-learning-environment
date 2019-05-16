# coding=utf-8
##
# By Fan
##

"""Implementation of a TOM agent adapted to the multiplayer setting."""

from qmdp_agent import QMDPAgent
import torch

# --------------------------------------- Rainbow vvv

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from third_party.dopamine import checkpointer
import dqn_agent
import rainbow_agent
import gin.tf
import numpy as np
import prioritized_replay_memory
import tensorflow as tf

slim = tf.contrib.slim
# --------------------------------------- Rainbow ^^^

import pdb

class ResidualNet(torch.nn.Module):
  '''
  the state action value residual neural net

  same structure as a dqn net
  '''

  def __init__(self,
               observation_size,
               num_actions,
               fc_size
               ):
    super(ResidualNet, self).__init__()

    super(ResidualNet, self).add_module('in',
        self.action_fc = torch.nn.Linear(observation_size, fc_size)

  def forward(self,
              ):
    '''
    observation
    - board
    - discards

    ??
    - hands
      - other's hands
      - our hint mask

    ??
    - hands
      - filled-in "full obs" hands?
    '''

    # prob mask
    # (mask for which obs vector components this probability helps sample)
    pass


class ExpectationNet(torch.nn.Module):
  '''
  the torch module holding the kernel module and value residual neral net
  handles computation of expectation using those two (via importance sampling)
  '''

  def __init__(self,
               observation_size,
               num_actions,
               num_players
               ):
    super(ExpectationNet, self).__init__()

    # self.kernel_net =
    #def __init__(self, obs_shape, action_shape, belief_size, half_fc_size=32, fc_layer_num=3):

    #self.kernel_net =  Dummy kernel

    # self.value_residual = ResidualNet()
    # using rainbow?

  def forward(self, obs, action):
    # do sampling
    pass

  def expected_action_value(self, prev_belief, obs, prev_a):
    '''
    computes expectation by importance weighted sampling
    '''
    #kernel = self.kernel_model.forward(obs, force)
    #belief = torch.bmm(prev_belief.unsqueeze(-2), kernel)  # batch_size x 1 x belief_size

    dummy_belief = None
    belief = dummy_belief
    return belief

  def proposal_sample(self):
    pass

  def consistent_card_belief(self):
    pass

  def value_residual(self):
    pass

class ResidualAgent(object): #maybe inherit from dqn agent

  def __init__(self,
               game,
               checkpoint_dir,
               agent_type,
               observation_size,
               num_actions,
               num_players
               ):
    """Initializes the agent and constructs its graph.
    Args:
      num_actions: int, number of actions the agent can take at any state.
      observation_size: int, size of observation vector.
      num_players: int, number of players playing this game.
      #num_atoms: Int, the number of buckets for the value function distribution.
      #vmax: float, maximum return predicted by a value distribution.
      gamma: float, discount factor as commonly used in the RL literature.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of stored transitions before training.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_train: float, final epsilon for training.
      epsilon_eval: float, epsilon during evaluation.
      epsilon_decay_period: int, number of steps for epsilon to decay.
      #learning_rate: float, learning rate for the optimizer.
      optimizer_epsilon: float, epsilon for Adam optimizer.
      tf_device: str, Tensorflow device on which to run computations.
    """
    # TODO: consider adding out of graph (prioritized) replay memory from rainbow
    # TODO: consider double Q learning (from rainbow)
    # TODO: consider n-step learning (from rainbow)
    self.qmdp = QMDPAgent(
                  game = environment.game,
                  checkpoint_dir      = CKPT_DIR,
                  agent_type          = test,
                  observation_size    = obs_stacker.observation_size(),
                  num_actions         = environment.num_moves(),
                  num_players         = environment.players)

  def begin_episode(self, current_player, legal_actions, observation):
    pass

  def step(self, reward, current_player, legal_actions, observation):
    #TODO: check
    self._train_step()

    self.action = self._select_action(observation, legal_actions)

    #self._record_transition(current_player, reward, observation, legal_actions,
    #                        self.action)
    return self.action

  def end_episode(self, final_rewards):
    # DO NOTHING
    return

  def _select_action(self, observation, legal_actions):
    q_a_1 = self.qmdp._expected_action_value(observation)
    # q_a_2 =
    # q_a_3 =

  def _train_step(self):
    if self.eval_mode:
      return

if __name__ == '__main__':
  pass
