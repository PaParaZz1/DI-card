from typing import Optional, List
import gym
import numpy as np
import treetensor.numpy as tnp
from ding.torch_utils import to_device
from obs import get_observation_space, Character, Card
from action import get_action_space, BasicRuleUtilities
from obs_encoder import ObservationEncoder


class GenshinCardEnv(gym.Env):

    def __init__(
        self,
        env_id: str,
        character_list: Optional[List[Character]],
        card_list: Optional[List[Card]],
        max_card_num: int = 30,     # The deck contains 30 action cards
        character_num: int = 3 * 2,  # our side and other side
        embedding_num: int = 32,
        max_skill_num: int = 4,
        max_usable_card_num: int = 10,
        max_usable_dice_num: int = 16,
        max_summoner_num: int = 4 * 2,  # our side and other side
        max_supporter_num: int = 4 * 2,  # our side and other side
    ):
        self.env_id = env_id
        if self.env_id is None:
            self.character_list = character_list
            self.card_list = card_list
        else:
            raise NotImplementedError
        self.max_card_num = max_card_num
        self.character_num = character_num
        self.embedding_num = embedding_num
        self.max_skill_num = max_skill_num
        self.max_usable_card_num = max_usable_card_num
        self.max_usable_dice_num = max_usable_dice_num
        self.max_summoner_num = max_summoner_num
        self.max_supporter_num = max_supporter_num

        self.observation_space = get_observation_space(
            character_num, embedding_num, max_skill_num, max_usable_card_num, max_usable_dice_num, max_summoner_num,
            max_supporter_num
        )
        self.action_space = get_action_space(character_list)
        # self.reward_space = get_reward_space()

    def reset(self, seed: int = 0):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def is_done(self):
        raise NotImplementedError

    def _get_obs(self, raw_game_info) -> tnp.ndarray:
        
        raise NotImplementedError


if __name__ == "__main__":
    env = GenshinCardEnv(env_id=None, character_list=None, card_list=None)
    obs = env.observation_space.sample()
    print(obs)
    print([v.shape for v in obs.values()])
    flatten_obs = tnp.concatenate(list(obs.values()))
    print(flatten_obs.shape)
    for _ in range(5):
        action = env.action_space.sample(obs=obs)
        print(action)
    batch_size = 8
    embedding_size = 256
    obs_encoder = to_device(ObservationEncoder(env.observation_space,output_size=embedding_size),'cuda')
    batch_obs = to_device([obs.tensor() for i in range(batch_size)],'cuda')
    batch_last_action = to_device([env.action_space.sample(obs=obs).tensor() for i in range(batch_size)],'cuda')
    encoded_obs = obs_encoder(batch_obs, batch_last_action)
    assert encoded_obs.shape == (batch_size, embedding_size), 'shape of encoded_obs wrong'
    print('end')
