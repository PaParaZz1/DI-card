import torch
import numpy as np
import pytest

from ding.torch_utils import is_differentiable,to_device
from ding.utils import squeeze
from easydict import EasyDict

from genshin_card_env import GenshinCardEnv
from obs_encoder import ObservationEncoder

B = 8
test_device = 'cuda'
embedding_size = 256


@pytest.mark.unittest
class TestObservationEncoder:

    def test_obs_encoder(self):
        env = GenshinCardEnv(env_id=None, character_list=None, card_list=None)
        # obs = env.observation_space.sample()
        # print(obs)
        obs_encoder = to_device(ObservationEncoder(env.observation_space, output_size=embedding_size), test_device)
        batch_obs = [env.observation_space.sample() for i in range(B)]
        batch_last_action = to_device([env.action_space.sample(obs=obs).tensor() for obs in batch_obs], test_device)
        batch_obs_tensor = to_device([obs.tensor() for obs in batch_obs], test_device)
        for obs,last_action in zip(batch_obs_tensor, batch_last_action):
            print(obs, last_action)
        encoded_obs = obs_encoder(batch_obs_tensor, batch_last_action)
        assert encoded_obs.shape == (B, embedding_size), 'shape of encoded_obs wrong'

        is_differentiable(encoded_obs.sum(), obs_encoder)
        
