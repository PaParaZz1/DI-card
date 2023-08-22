import torch
import numpy as np
import pytest
import random

from ding.torch_utils import is_differentiable,to_device
from ding.utils import squeeze
from easydict import EasyDict

from genshin_card_env import GenshinCardEnv
from obs_encoder import ObservationEncoder
from policy_head import GenshinVAC

B = 8
test_device = 'cuda'
embedding_size = 256

def test_policy_head():
    env = GenshinCardEnv(env_id=None, character_list=None, card_list=None, embedding_num=embedding_size)
    # obs = env.observation_space.sample()
    # print(obs)
    obs_encoder = to_device(ObservationEncoder(env.observation_space, output_size=embedding_size), test_device)
    batch_obs = [env.observation_space.sample() for i in range(B)]
    batch_last_action = to_device([env.action_space.sample(obs=obs).tensor() for obs in batch_obs], test_device)
    batch_obs_tensor = to_device([obs.tensor() for obs in batch_obs], test_device)
    for obs,last_action in zip(batch_obs_tensor, batch_last_action):
        print(obs, last_action)
    encoded_obs = obs_encoder(batch_obs_tensor, batch_last_action)
    assert isinstance(encoded_obs, dict), "encoded_obs should be a dictionary"
    assert encoded_obs['merged_obs'].shape == (B, embedding_size), 'Shape of encoded_obs should be {}'.format((B, embedding_size))
    is_differentiable(encoded_obs['merged_obs'].sum(), obs_encoder)

    policy_head = to_device(GenshinVAC(
        obs_embedding_shape = embedding_size,
        action_space = env.action_space,
        encoded_obs_shape = obs_encoder.obs_merge_input_sizes,
    ), test_device)
    # test computer_actor_critic
    obs_embedding = encoded_obs['merged_obs'].detach()
    encoded_obs_for_head = {key: obs.detach() for key,obs in encoded_obs['obs_for_head'].items()}
    outputs0 = policy_head.forward(
        mode='compute_actor_critic',
        obs_embedding = obs_embedding,
        encoded_obs = encoded_obs_for_head,
        selected_action_type = random.choice(range(5)),
        sample_action_type = 'argmax',
    )
    assert outputs0['value'].shape == (B,), "action_logit should be a dictionary"
    test_output0 = outputs0['value'].sum() + outputs0['logit']['action_type'].sum() + sum([action_arg_logit.sum() for action_arg_logit in outputs0['logit']['action_args'].values()])
    is_differentiable(test_output0, policy_head)
    # test computer_actor
    for p in obs_encoder.parameters():
        p.grad = None
    for p in policy_head.parameters():
        p.grad = None
    outputs1 = policy_head.forward(
        mode='compute_actor',
        obs_embedding = obs_embedding,
        encoded_obs = encoded_obs_for_head,
        selected_action_type = random.choice(range(5)),
        sample_action_type = 'argmax',
    )
    assert isinstance(outputs1['logit'], dict), "outputs['logit'] should be a dictionary"
    test_output1 = outputs1['logit']['action_type'].sum() + sum([action_arg_logit.sum() for action_arg_logit in outputs1['logit']['action_args'].values()])
    is_differentiable(test_output1, policy_head)

    # test computer_critic
    for p in obs_encoder.parameters():
        p.grad = None
    for p in policy_head.parameters():
        p.grad = None
    outputs2 = policy_head.forward(
        mode='compute_critic',
        obs_embedding = obs_embedding,
    )
    assert outputs2['value'].shape == (B,), "action_logit should be a dictionary"
    is_differentiable(outputs2['value'].sum(), policy_head)
    

test_policy_head()