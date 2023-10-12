import pytest
import torch

from ding.torch_utils import is_differentiable, to_device

from genshin_card_env import GenshinCardEnv
from encoder import ObservationEncoder

B = 8
test_device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_size = 256


@pytest.mark.unittest
class TestObservationEncoder:

    def test_encoder(self):
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
        assert isinstance(encoded_obs, dict), "encoded_obs should be a dictionary"
        assert encoded_obs['merged_obs'].shape == (B, embedding_size), 'Shape of encoded_obs should be {}'.format(
            (B, embedding_size)
        )
        is_differentiable(encoded_obs['merged_obs'].sum(), encoder)
