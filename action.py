from typing import List
from dataclasses import dataclass
import numpy as np
import gym
import treetensor.numpy as tnp
from obs import Character, Card, ObservationSpace, ElementalDiceType


class BasicRuleUtilities:

    @staticmethod
    def replace_card_at_beginning(init_obs:tnp.ndarray):
        # At the beginning, five action cards will be drawn
        # V0: Directly discard the cards that need to consume more than 3 dice
        card_dice_cost = init_obs.card_dice_cost[:5]
        card_selected_change = np.zeros_like(card_dice_cost)
        card_selected_change[card_dice_cost >= 3] = 1
        return card_selected_change

    @staticmethod
    def reroll_dice(obs:tnp.ndarray):
        # Non-currently playing character elements and universal elements, the rest are rerolled
        our_current_character_index = np.where(obs.character_is_battle & (~obs.character_is_enemy))[0]
        our_current_character_element = obs.character_element_type[our_current_character_index[0]]
        dice_reroll = [
            # 'colorless': always keep the colorless dice
            0,
            # 'pyro': 
            obs.pyro_dice_num[0],
            # 'hydro': 
            obs.hydro_dice_num[0],
            # 'electro': 
            obs.electro_dice_num[0],
            # 'cryo': 
            obs.cryo_dice_num[0],
            # 'geo': 
            obs.geo_dice_num[0],
            # 'anemo': 
            obs.anemo_dice_num[0],
            # 'dendro': 
            obs.dendro_dice_num[0],
        ]
        dice_reroll[our_current_character_element] = 0
        return np.array(dice_reroll)

    @staticmethod
    def select_skill_target_character(skill_type, skill_arg, obs:tnp.ndarray):
        """skill contains cards and character skills"""
        # according to skill_type, card/skill to choose
        if skill_type == ActionType.play_card:
            # TODO: require information about usable objects of cards
            # play card: By default, select our current front-end character
            target_character = obs.character_is_battle & (~obs.character_is_enemy)
        elif skill_type == ActionType.use_skill:
            # use skill: By default, select opponent's current front-end character
            target_character = obs.character_is_battle & obs.character_is_enemy
        return  target_character

    @staticmethod
    def select_skill_dice(skill_type, skill_arg, obs:tnp.ndarray):
        dice_list = [
            # 'colorless':
            obs.colorless_dice_num[0],
            # 'pyro': 
            obs.pyro_dice_num[0],
            # 'hydro': 
            obs.hydro_dice_num[0],
            # 'electro': 
            obs.electro_dice_num[0],
            # 'cryo': 
            obs.cryo_dice_num[0],
            # 'geo': 
            obs.geo_dice_num[0],
            # 'anemo': 
            obs.anemo_dice_num[0],
            # 'dendro': 
            obs.dendro_dice_num[0],
        ]
        if skill_type == ActionType.elemental_harmony:
            # By default this case has at least one non-current element dice
            # we directly select the least no. of non-current element dice
            our_current_character_index = np.where(obs.character_is_battle & (~obs.character_is_enemy))[0]
            our_current_character_element = obs.character_element_type[our_current_character_index[0]]
            dice_list[our_current_character_element] = np.maximum
            dice_list[ElementalDiceType.colorless] = np.maximum
            dice_select_type = np.argmin(dice_list)
            dice_select_list = np.zeros_like(dice_list)
            dice_select_list[dice_select_type] = 1
        elif skill_type == ActionType.play_card:
            # Wait for the deck to complete
            pass
        elif skill_type == ActionType.use_skill:
            # Wait for the deck to complete
            pass

        return dice_select_list

    @staticmethod
    def select_elemental_harmony_target_card(obs:tnp.ndarray):
        card_dice_cost = obs.card_dice_cost
        # Temporarily choose the card that consumes the most dice
        max_dice_cost = max(card_dice_cost)
        max_dice_cost_indices = [i for i, cost in enumerate(card_dice_cost) if cost == max_dice_cost]
        max_dice_cost_card_index = np.random.choice(max_dice_cost_indices)
        raise max_dice_cost_card_index

    @staticmethod
    def get_raw_action():
        raise NotImplementedError


@dataclass
class ActionType:
    play_card: int = 0
    elemental_harmony: int = 1
    use_skill: int = 2
    change_character: int = 3
    terminate_turn: int = 4

@dataclass
class ActionArgs:
    card: int = 0
    skill: int = 1
    character: int = 2
    none: int = 3


class ActionSpace(gym.spaces.Dict):

    def __init__(self):

        action_type_space = gym.spaces.Discrete(5)  # play card/elemental harmony/use skill/switch role/end round
        card_action_args_space = gym.spaces.Discrete(10)  # 10 options for the argseter space of play card/elemental harmony
        skill_action_args_space = gym.spaces.Discrete(4)  # 3/4 options for arg space for use skills
        change_character_action_args_space = gym.spaces.Discrete(3)  # 3 options for arg space for switch role
        action_arg_spaces = {
            'play_card': card_action_args_space,
            'elemental_harmony': card_action_args_space,
            'use_skill': skill_action_args_space,
            'change_character': change_character_action_args_space,
            'terminate_turn': gym.spaces.Discrete(1)
        }
        action_space_dict = {
            'action_type_space':action_type_space,
            'action_arg_space':gym.spaces.Dict(action_arg_spaces)
        }
        super().__init__(action_space_dict)

    def sample(self, obs=None):
        action_type = self['action_type_space'].sample()  # Randomly choose the type of action
        if obs is not None:
            card_mask = obs['card_is_available'].astype(np.int8)
            skill_mask = obs['skill_is_available'].astype(np.int8)
            character_mask = obs['character_is_alive'][:3].astype(np.int8)
        else:
            card_mask = np.ones(self['action_arg_space']['play_card'].n,dtype=np.int8)
            skill_mask = np.ones(self['action_arg_space']['use_skill'].n,dtype=np.int8)
            character_mask = np.ones(self['action_arg_space']['change_character'].n,dtype=np.int8)

        if action_type == 0:    # play card
            action_args = self['action_arg_space']['play_card'].sample(mask=card_mask)
        elif action_type == 1:  # element harmony
            action_args = self['action_arg_space']['elemental_harmony'].sample(mask=card_mask)  # Randomly select args for play card/elemental harmony
        elif action_type == 2:  # use skills
            action_args = self['action_arg_space']['use_skill'].sample(mask=skill_mask)
        elif action_type == 3:  # switch role
            action_args = self['action_arg_space']['change_character'].sample(mask=character_mask)  # Randomly select args for switch role
        else:  # end round
            # action_args = None
            action_args = -1
        return tnp.array({'action_type': action_type, 'action_args': action_args})
  
    # def sample(self, obs) -> tnp.ndarray:
    #     # action_type
    #     p = np.array([1. for _ in range(5)])
    #     if obs.card_num == 0:
    #         p[0] = 0.
    #         p[1] = 0.
    #     if obs.skill_is_available.sum() == 0:
    #         p[2] = 0.
    #     if obs.character_is_alive[:3].sum() <= 1:
    #         p[3] = 0.
    #     p /= p.sum()
    #     action_type = np.random.choice(5, p=p)
    #     # action_args
    #     if action_type == 0:
    #         # choice = obs.card_is_available
    #         # p = choice / choice.sum()
    #         # action_args = np.random.choice(choice, p=p)
    #         # maybe like that:
    #         choice = np.where(obs.card_is_available)
    #         action_args = np.random.choice(choice)
    #     elif action_type == 1:
    #         action_args = np.random.choice(obs.card_num)
    #     elif action_type == 2:
    #         # choice = obs.skill_is_available
    #         # p = choice / choice.sum()
    #         # action_args = np.random.choice(choice, p=p)
    #         choice = np.where(obs.skill_is_available)
    #         action_args = np.random.choice(choice)
    #     elif action_type == 3:
    #         # choice = obs.character_is_alive[:3] * (1 - obs.character_is_battle[:3])
    #         # p = choice / choice.sum()
    #         # action_args = np.random.choice(choice, p=p)
    #         choice = np.where(obs.character_is_alive[:3] * (1 - obs.character_is_battle[:3]))
    #         action_args = np.random.choice(choice)
    #     else:
    #         action_args = 0
    #     # np.int64
    #     return tnp.array({'action_type': action_type, 'action_args': action_args})



def get_action_space(character_list: List[Character]):
    return ActionSpace()

