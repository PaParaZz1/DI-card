import torch
import numpy as np
import pytest
import random

from ding.torch_utils import is_differentiable,to_device

from genshin_card_env import GenshinCardEnv
from encoder import ObservationEncoder
from head import GenshinVAC
from card_network import CardNetwork
from fake_data import Data_Generator

B = 8
test_device = 'cuda'
embedding_size = 256

@pytest.mark.unittest
class TestCardNetwork:

    def test_compute_actor_critic(self):
        generator = Data_Generator()
        net = CardNetwork(
            generator.observation_space,
            generator.action_space,
            encoder_output_size=embedding_size,
        ).to(test_device)
        batch_obs, batch_last_action = generator.get_batch_data(batch_size=B, device=test_device)
        outputs = net(
            batch_obs,
            batch_last_action,
            head_mode = 'compute_actor_critic'
            )
        assert outputs['value'].shape == (B, ), "action_logit should be a dictionary"
        test_output = outputs['value'].sum() + outputs['logit']['action_type'].sum() + sum(
                [action_arg_logit.sum() for action_arg_logit in outputs['logit']['action_args'].values()]
            )
        is_differentiable(test_output, net)

    def test_compute_actor(self):
        generator = Data_Generator()
        net = CardNetwork(
            generator.observation_space,
            generator.action_space,
            encoder_output_size=embedding_size,
        ).to(test_device)
        batch_obs, batch_last_action = generator.get_batch_data(batch_size=B, device=test_device)
        outputs = net(
            batch_obs,
            batch_last_action,
            head_mode = 'compute_actor'
            )
        assert isinstance(outputs['logit'], dict), "outputs['logit'] should be a dictionary"
        test_output = outputs['logit']['action_type'].sum() + sum(
                [action_arg_logit.sum() for action_arg_logit in outputs['logit']['action_args'].values()]
            )
        is_differentiable(test_output, [net.encoder, net.head.actor_head])

    def test_compute_critic(self):
        generator = Data_Generator()
        net = CardNetwork(
            generator.observation_space,
            generator.action_space,
            encoder_output_size=embedding_size,
        ).to(test_device)
        batch_obs, batch_last_action = generator.get_batch_data(batch_size=B, device=test_device)
        outputs = net(
            batch_obs,
            batch_last_action,
            head_mode = 'compute_critic'
            )
        assert outputs['value'].shape == (B, ), "action_logit should be a dictionary"
        is_differentiable(outputs['value'].sum(), [net.encoder, net.head.critic_head])

# test_compute_actor_critic()
# test_compute_actor()
# test_compute_critic()