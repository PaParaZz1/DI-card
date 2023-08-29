from typing import Optional, List, Tuple, Dict
import gym
import numpy as np
import torch
import treetensor.numpy as tnp
from ding.torch_utils import to_device

from gisim.game import Game
from gisim.agent import AttackOnlyAgent, NoAttackAgent
from gisim.classes.enums import GameStatus, PlayerID

from obs import get_observation_space, Character, Card
from action import get_action_space
from obs_encoder import ObservationEncoder


class GenshinCardEnv(gym.Env):

    def __init__(
        self,
        env_id: str,
        character_list: Optional[List[Character]],
        card_list: Optional[List[Card]],
        max_card_num: int = 30,  # The deck contains 30 action cards
        character_num: int = 3 * 2,  # our side and other side
        embedding_num: int = 32,
        max_skill_num: int = 4,
        max_usable_card_num: int = 10,
        max_usable_dice_num: int = 16,
        max_summoner_num: int = 4 * 2,  # our side and other side
        max_supporter_num: int = 4 * 2,  # our side and other side
        max_episode_count: Optional[int] = 100,
        debug: bool = False,
    ):
        self.env_id = env_id
        if self.env_id is None:
            if character_list is not None:
                self.player1_deck = {
                    "characters": character_list[0],
                    "cards": card_list[0],
                }
                self.player2_deck = {
                    "characters": character_list[1],
                    "cards": card_list[1],
                }
            else:  # for test and debug
                self.player1_deck = {
                    "characters": ["Kamisato Ayaka", "Kamisato Ayaka", "Kamisato Ayaka"],
                    "cards": ["Kanten Senmyou Blessing", "Traveler's Handy Sword"],
                }
                self.player2_deck = {
                    "characters": ["Kamisato Ayaka", "Kamisato Ayaka", "Kamisato Ayaka"],
                    "cards": [],
                }
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
        self.max_episode_count = max_episode_count

        self.observation_space = get_observation_space(
            character_num, embedding_num, max_skill_num, max_usable_card_num, max_usable_dice_num, max_summoner_num,
            max_supporter_num
        )
        self.action_space = get_action_space(character_list)
        # self.reward_space = get_reward_space()

        self.debug = debug
        self.last_action = None
        self.episode_count = 0
        self.init_flag = False

    def reset(self, seed: int = 0):
        self.random_seed = seed
        if not self.init_flag:
            self.init_flag = True
            self.game = Game()
            self.game.init_deck(self.player1_deck, self.player2_deck, seed=self.random_seed)
            if self.debug:
                self.player1 = AttackOnlyAgent(PlayerID.PLAYER1)
                self.player2 = NoAttackAgent(PlayerID.PLAYER2)
            else:
                # TODO RuleBasedAgent
                # TODO RLAgent
                raise NotImplementedError
        else:
            self.game.init_deck(self.player1_deck, self.player2_deck, seed=self.random_seed)
        self.episode_count = 0
        self.game_info = self.game.encode_game_info(PlayerID.SPECTATOR)

        obs = self._get_obs(self.game_info, self.last_action)

        self.last_action = None
        self.last_obs = obs
        return obs

    def step(self, action):
        # preprocess action
        if action is None:
            active_player = self.game_info.active_player
            if active_player == PlayerID.PLAYER1:
                action = self.player1_agent.take_action(self.game_info)
            elif active_player == PlayerID.PLAYER2:
                action = self.player2_agent.take_action(self.game_info)
        else:
            print('action', action)
            raw_action = self.action_space.transform_raw_action(action, self.last_obs)
            print('raw action', raw_action)

        # execute action
        valid = self.game.judge_action(raw_action)
        if not valid:
            raise NotImplementedError
        self.game.step(raw_action)
        self.game_info = self.game.encode_game_info()
        obs = self._get_obs(self.game_info, self.last_action)
        done, info = self.is_done(self.game_info)
        reward = 0.  # TODO

        self.last_obs = obs
        self.last_action = action
        return obs, reward, done, info

    def is_done(self, raw_game_info) -> Tuple[bool, Dict]:
        done, info = False, {}
        if raw_game_info.status == GameStatus.ENDED:
            done = True
            info['winner'] = raw_game_info.winner
        if self.max_episode_count and self.episode_count >= self.max_episode_count:
            done = True
            info['truncated'] = True
        return done, info

    def _get_obs(self, raw_game_info, last_action) -> tnp.ndarray:
        obs = self.observation_space.get_zero_data()
        return obs


if __name__ == "__main__":
    np.random.seed(3)
    env = GenshinCardEnv(env_id=None, character_list=None, card_list=None, debug=True)
    obs = env.observation_space.sample()
    print([v.shape for v in obs.values()])
    obs = env.reset(seed=314)
    flatten_obs = tnp.concatenate(list(obs.values()))
    print(flatten_obs.shape)
    for _ in range(5):
        action = env.action_space.sample(obs=obs)
        print(action)
        obs, rew, done, info = env.step(action)
    batch_size = 8
    embedding_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    obs_encoder = ObservationEncoder(env.observation_space, output_size=embedding_size).to(device)
    batch_obs = to_device([env.observation_space.sample().tensor() for i in range(batch_size)], device)
    batch_last_action = to_device([env.action_space.sample(obs=obs).tensor() for i in range(batch_size)], device)
    encoded_obs = obs_encoder(batch_obs, batch_last_action)
    assert encoded_obs.shape == (batch_size, embedding_size), 'shape of encoded_obs should be {}'.format(
        (batch_size, embedding_size)
    )
    print('end')
