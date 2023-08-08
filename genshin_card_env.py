from typing import Optional, List, Tuple, Dict

import gym
import numpy as np
import torch
import treetensor.numpy as tnp
from ding.torch_utils import to_device
from ditk import logging
from gisim.agent import AttackOnlyAgent, NoAttackAgent
from gisim.classes.enums import GameStatus, PlayerID
from gisim.game import Game, GameInfo

from action import get_action_space
from obs import get_observation_space, Character, Card, ElementalDiceType
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
        self.last_action = None
        self.episode_count = 0
        self.game_info = self.game.encode_game_info(PlayerID.SPECTATOR)
        return self._get_obs(self.game_info, self.last_action)

    def step(self, action):
        # preprocess action
        if action is None:
            active_player = self.game_info.active_player
            if active_player == PlayerID.PLAYER1:
                action = self.player1_agent.take_action(self.game_info)
            elif active_player == PlayerID.PLAYER2:
                action = self.player2_agent.take_action(self.game_info)
        else:
            # TODO action transformation
            raise NotImplementedError

        # execute action
        valid = self.game.judge_action(action)
        if not valid:
            raise NotImplementedError
        self.game.step(action)
        self.game_info = self.game.encode_game_info()
        obs = self._get_obs(self.game_info, self.last_action)
        done, info = self.is_done(self.game_info)
        reward = 0.  # TODO
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

    def _get_obs(self, raw_game_info: GameInfo, last_action) -> tnp.ndarray:
        logging.info(f'raw_game_info: {raw_game_info!r}')
        logging.info(f'last_action: {last_action!r}')

        logging.info(f'active player: {raw_game_info.active_player!r}')
        active_player = raw_game_info.active_player
        if active_player == PlayerID.PLAYER1:
            player = raw_game_info.player1
            op_player = raw_game_info.player2
        elif active_player == PlayerID.PLAYER2:
            player = raw_game_info.player2
            op_player = raw_game_info.player1
        else:
            raise ValueError(f'Unknown active player id - {active_player!r}.')

        # dice_num = len(player.dice_zone_len)
        logging.info(f'Dice zone: {player.dice_zone!r}')
        assert len(player.characters) + len(op_player.characters) == self.character_num
        for i, ch in enumerate(player.characters):
            logging.info(f'character #{i}: {ch!r}, status: {ch.status}, talent: {ch.talent}, '
                         f'weapon: {ch.weapon}, artifact: {ch.artifact}')
        # player.characters[0].character.alive

        # player.support_zone[0].
        all_characters = [*player.characters, *op_player.characters]
        all_ch_enemy = [*(0 for ch in player.characters), *(1 for ch in op_player.characters)]

        all_summoners = [*player.summon_zone, *op_player.summon_zone]
        all_summoners_enemy = [*(0 for s in player.summon_zone), *(1 for s in op_player.summon_zone)]

        all_supporters = [*player.support_zone, *op_player.support_zone]
        all_supporters_enemy = [*(0 for s in player.support_zone), *(1 for s in op_player.support_zone)]

        return tnp.array({
            'last_play': [1 if active_player == PlayerID.PLAYER1 else -1],
            'dice_num': [player.dice_zone_len],
            'opposite_dice_num': [op_player.dice_zone_len],
            'colorless_dice_num': [
                len([_ for dice in (player.dice_zone or []) if dice == ElementalDiceType.colorless])],
            'pyro_dice_num': [len([_ for dice in (player.dice_zone or []) if dice == ElementalDiceType.pyro])],
            'hydro_dice_num': [len([_ for dice in (player.dice_zone or []) if dice == ElementalDiceType.hydro])],
            'electro_dice_num': [len([_ for dice in (player.dice_zone or []) if dice == ElementalDiceType.electro])],
            'cryo_dice_num': [len([_ for dice in (player.dice_zone or []) if dice == ElementalDiceType.cryo])],
            'geo_dice_num': [len([_ for dice in (player.dice_zone or []) if dice == ElementalDiceType.geo])],
            'anemo_dice_num': [len([_ for dice in (player.dice_zone or []) if dice == ElementalDiceType.anemo])],
            'dendro_dice_num': [len([_ for dice in (player.dice_zone or []) if dice == ElementalDiceType.dendro])],

            'character_is_alive': [1 if ch.character.alive else 0 for ch in all_characters],
            'character_is_battle': [1 if ch.character.active else 0 for ch in all_characters],
            'character_hp': [ch.character.health_point for ch in all_characters],
            'character_charge_point': [ch.character.power for ch in all_characters],
            'character_charge_point_max': [ch.character.max_power for ch in all_characters],
            'character_weapon_type': [ch.weapon or 0 for ch in all_characters],
            'character_element_type': [0 for ch in all_characters],  # TODO
            'character_is_enemy': all_ch_enemy,
            'character_element_attachment': [0 for ch in all_characters],  # TODO
            'character_is_full': [1 if (ch.character.max_power and ch.character.power == ch.character.max_power) else 0
                                  for ch in all_characters],
            'character_other_info': np.clip(np.random.random((self.character_num * self.embedding_num,)),
                                            a_min=-1.0, a_max=1.0),  # TODO

            # TODO: implement of skill info not found in game simulator,
            #     maybe it need to be completed, the code now is just placeholders
            # skill_is_available=((max_skill_num,), FeatureType.CATEGORICAL, (0, 2)),
            # skill_is_charge = ((max_skill_num,), FeatureType.CATEGORICAL, (0, 2)),
            # skill_direct_damage = ((max_skill_num,), FeatureType.CATEGORICAL, (0, 8)),
            #     # TODO dynamic skill cost
            # skill_other_info = ((max_skill_num * embedding_num,), FeatureType.SCALAR, (-1, 1)),
            #     # TODO enemy skill information
            'skill_is_available': [1 for _ in range(self.max_skill_num)],  # TODO
            'skill_is_charge': [0 for _ in range(self.max_skill_num)],  # TODO
            'skill_direct_damage': [0 for _ in range(self.max_skill_num)],  # TODO
            'skill_other_info': np.clip(np.random.random((self.max_skill_num * self.embedding_num,)),
                                        a_min=-1.0, a_max=1.0),  # TODO

            # card_is_available = ((max_usable_card_num,), FeatureType.CATEGORICAL, (0, 2)),
            # card_is_same_dice = ((max_usable_card_num,), FeatureType.CATEGORICAL, (0, 2)),
            # card_dice_cost = ((max_usable_card_num,), FeatureType.CATEGORICAL, (0, 4)),
            # card_type = ((max_usable_card_num,), FeatureType.CATEGORICAL, (0, 3)),  # event, support, equipment
            # card_other_info = ((max_usable_card_num * embedding_num,), FeatureType.SCALAR, (-1, 1)),
            # card_num = ((1,), FeatureType.CATEGORICAL, (0, max_usable_card_num + 1)),
            # enemy_card_num = ((1,), FeatureType.CATEGORICAL, (0, max_usable_card_num + 1)),
            'card_is_available': [1 if i in range(len(player.hand_cards)) else 0
                                  for i in range(self.max_usable_card_num)],
            'card_is_same_dice': [1 if i in range(len(player.hand_cards)) else 0
                                  for i in range(self.max_usable_card_num)],  # TODO
            'card_dice_cost': [1 if i in range(len(player.hand_cards)) else 0
                               for i in range(self.max_usable_card_num)],  # TODO
            'card_type': [0 for _ in range(self.max_usable_card_num)],  # TODO
            'card_other_info': np.clip(np.random.random((self.max_usable_card_num * self.embedding_num,)),
                                       a_min=-1.0, a_max=1.0),  # TODO
            'card_num': [len(player.hand_cards)],
            'enemy_card_num': [len(op_player.hand_cards)],

            # summoner_is_available = ((max_summoner_num,), FeatureType.CATEGORICAL, (0, 2)),
            # summoner_is_enemy = ((max_summoner_num,), FeatureType.CATEGORICAL, (0, 2)),
            # summoner_remain_turn = ((max_summoner_num,), FeatureType.CATEGORICAL, (0, 5)),  # 3+1+1
            # summoner_other_info = ((max_summoner_num * embedding_num,), FeatureType.SCALAR, (-1, 1)),
            'summoner_is_available': [1 if i in range(len(all_summoners)) else 0
                                      for i in range(self.max_summoner_num)],
            'summoner_is_enemy': [all_summoners_enemy[i] if i in range(len(all_summoners_enemy)) else 0
                                  for i in range(self.max_summoner_num)],
            'summoner_remain_turn': [1 if i in range(len(all_summoners)) else 0
                                     for i in range(self.max_summoner_num)],  # TODO
            'summoner_other_info': np.clip(np.random.random((self.max_summoner_num * self.embedding_num,)),
                                           a_min=-1.0, a_max=1.0),  # TODO

            # supporter_is_available = ((max_supporter_num,), FeatureType.CATEGORICAL, (0, 2)),
            # supporter_is_enemy = ((max_supporter_num,), FeatureType.CATEGORICAL, (0, 2)),
            # supporter_count = ((max_supporter_num,), FeatureType.CATEGORICAL, (0, 4)),
            # supporter_other_info = ((max_supporter_num * embedding_num,), FeatureType.SCALAR, (-1, 1)),
            'supporter_is_available': [1 if i in range(len(all_supporters)) else 0
                                       for i in range(self.max_supporter_num)],
            'supporter_is_enemy': [all_supporters_enemy[i] if i in range(len(all_supporters_enemy)) else 0
                                   for i in range(self.max_supporter_num)],
            'supporter_count': [1 if i in range(len(all_supporters)) else 0
                                for i in range(self.max_supporter_num)],  # TODO
            'supporter_other_info': np.clip(np.random.random((self.max_supporter_num * self.embedding_num,)),
                                            a_min=-1.0, a_max=1.0),  # TODO
        })


if __name__ == "__main__":
    logging.try_init_root(logging.INFO)

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
    assert encoded_obs.shape == (batch_size, embedding_size), \
        'shape of encoded_obs should be {}'.format((batch_size, embedding_size))
    print('end')
