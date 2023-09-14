import torch
import numpy as np
import random

from ding.torch_utils import to_device

from obs import get_observation_space
from action import get_action_space

class Data_Generator:
    r"""
    Overview:
        data generator to generate fake data.
    """
    def __init__(
        self,
        character_num: int = 3 * 2,  # our side and other side
        embedding_num: int = 32,
        max_skill_num: int = 4,
        max_usable_card_num: int = 10,
        max_usable_dice_num: int = 16,
        max_summoner_num: int = 4 * 2,  # our side and other side
        max_supporter_num: int = 4 * 2,  # our side and other side
    ):
        super(Data_Generator, self).__init__()
        self.observation_space = get_observation_space(
            character_num, embedding_num, max_skill_num, max_usable_card_num, max_usable_dice_num, max_summoner_num,
            max_supporter_num
        )
        self.action_space = get_action_space(character_list=None)

    def get_batch_data(self, batch_size, device='cpu'):
        batch_obs = [self.observation_space.sample() for i in range(batch_size)]
        batch_last_action = to_device([self.action_space.sample(obs=obs).tensor() for obs in batch_obs], device)
        batch_obs_tensor = to_device([obs.tensor() for obs in batch_obs], device)
        return batch_obs_tensor, batch_last_action

