import torch
import pytest
import random

from ding.torch_utils import is_differentiable, to_device

from genshin_card_env import GenshinCardEnv
from encoder import ObservationEncoder
from head import GenshinVAC

B = 8
test_device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_size = 256


@pytest.mark.unittest
class TestHead:

    def test_compute_actor_critic(self):
        env = GenshinCardEnv(env_id=None, character_list=None, card_list=None, embedding_num=embedding_size)
        # obs = env.observation_space.sample()
        # print(obs)
        encoder = ObservationEncoder(env.observation_space, output_size=embedding_size).to(test_device)
        batch_obs = [env.observation_space.sample() for i in range(B)]
        batch_last_action = to_device([env.action_space.sample(obs=obs).tensor() for obs in batch_obs], test_device)
        batch_obs_tensor = to_device([obs.tensor() for obs in batch_obs], test_device)
        for obs, last_action in zip(batch_obs_tensor, batch_last_action):
            print(obs, last_action)
        encoded_obs = encoder(batch_obs_tensor, batch_last_action)
        # test compute_actor_critic
        head = GenshinVAC(
                obs_embedding_shape=embedding_size,
                action_space=env.action_space,
                encoded_obs_shape=encoder.obs_merge_input_sizes,
            ).to(test_device)
        obs_embedding = encoded_obs['merged_obs']
        encoded_obs_for_head = encoded_obs['obs_for_head']
        outputs0 = head.forward(
            mode='compute_actor_critic',
            obs_embedding=obs_embedding,
            encoded_obs=encoded_obs_for_head,
            selected_action_type=random.choice(range(5)),
            sample_action_type='argmax',
        )
        assert outputs0['value'].shape == (B, ), "action_logit should be a dictionary"
        test_output0 = outputs0['value'].sum() + outputs0['logit']['action_type'].sum() + sum(
            [action_arg_logit.sum() for action_arg_logit in outputs0['logit']['action_args'].values()]
        )
        is_differentiable(test_output0, head)

    def test_computer_actor(self):
        env = GenshinCardEnv(env_id=None, character_list=None, card_list=None, embedding_num=embedding_size)
        # obs = env.observation_space.sample()
        # print(obs)
        encoder = ObservationEncoder(env.observation_space, output_size=embedding_size).to(test_device)
        batch_obs = [env.observation_space.sample() for i in range(B)]
        batch_last_action = to_device([env.action_space.sample(obs=obs).tensor() for obs in batch_obs], test_device)
        batch_obs_tensor = to_device([obs.tensor() for obs in batch_obs], test_device)
        for obs, last_action in zip(batch_obs_tensor, batch_last_action):
            print(obs, last_action)
        encoded_obs = encoder(batch_obs_tensor, batch_last_action)
        head = GenshinVAC(
                obs_embedding_shape=embedding_size,
                action_space=env.action_space,
                encoded_obs_shape=encoder.obs_merge_input_sizes,
            ).to(test_device)
        obs_embedding = encoded_obs['merged_obs']
        encoded_obs_for_head = encoded_obs['obs_for_head']
        # test compute_actor
        for p in encoder.parameters():
            p.grad = None
        for p in head.parameters():
            p.grad = None
        outputs1 = head.forward(
            mode='compute_actor',
            obs_embedding=obs_embedding,
            encoded_obs=encoded_obs_for_head,
            selected_action_type=random.choice(range(5)),
            sample_action_type='argmax',
        )
        assert isinstance(outputs1['logit'], dict), "outputs['logit'] should be a dictionary"
        test_output1 = outputs1['logit']['action_type'].sum() + sum(
            [action_arg_logit.sum() for action_arg_logit in outputs1['logit']['action_args'].values()]
        )
        is_differentiable(test_output1, head.actor_head)

    def test_compute_critic(self):
        env = GenshinCardEnv(env_id=None, character_list=None, card_list=None, embedding_num=embedding_size)
        # obs = env.observation_space.sample()
        # print(obs)
        encoder = ObservationEncoder(env.observation_space, output_size=embedding_size).to(test_device)
        batch_obs = [env.observation_space.sample() for i in range(B)]
        batch_last_action = to_device([env.action_space.sample(obs=obs).tensor() for obs in batch_obs], test_device)
        batch_obs_tensor = to_device([obs.tensor() for obs in batch_obs], test_device)
        for obs, last_action in zip(batch_obs_tensor, batch_last_action):
            print(obs, last_action)
        encoded_obs = encoder(batch_obs_tensor, batch_last_action)
        head = GenshinVAC(
                obs_embedding_shape=embedding_size,
                action_space=env.action_space,
                encoded_obs_shape=encoder.obs_merge_input_sizes,
            ).to(test_device)
        obs_embedding = encoded_obs['merged_obs']
        # test compute_critic
        for p in encoder.parameters():
            p.grad = None
        for p in head.parameters():
            p.grad = None
        outputs2 = head.forward(
            mode='compute_critic',
            obs_embedding=obs_embedding,
        )
        assert outputs2['value'].shape == (B, ), "action_logit should be a dictionary"
        is_differentiable(outputs2['value'].sum(), head.critic_head)


# test_head()
