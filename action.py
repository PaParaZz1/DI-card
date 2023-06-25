import gym
from typing import List
from dataclasses import dataclass
import numpy as np
import treetensor.numpy as tnp
from obs import Character, Card, ObservationSpace, ElementalDiceType


class BasicRuleUtilities:

    @staticmethod
    def replace_card_at_beginning(init_obs:tnp):
        # At the beginning, five action cards will be drawn
        card_dice_cost = init_obs.card_dice_cost[:5]
        card_selected_change = np.zeros_like(card_dice_cost)
        card_selected_change[card_dice_cost >= 3] = 1
        return card_selected_change

    @staticmethod
    def reroll_dice(obs:tnp):
        # Non-currently playing character elements and universal elements, the rest are rerolled
        our_current_character_index = np.where((obs.character_is_battle == 1) & (obs.character_is_enemy == 0))[0]
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
    def select_skill_target_character(skill_type, skill_arg, obs:tnp):
        """skill contains cards and character skills"""
        # according to skill_type, card/skill to choose
        if skill_type == ActionType.play_card:
            # play card: By default, select our current front-end character
            target_character = obs.character_is_battle * (1 - obs.character_is_enemy)
        elif skill_type == ActionType.use_skill:
            # use skill: By default, select opponent's current front-end character
            target_character = obs.character_is_battle * obs.character_is_enemy
        return  target_character

    @staticmethod
    def select_skill_dice(skill_type, skill_arg, obs):
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
            our_current_character_index = np.where(obs.character_is_battle * (1 - obs.character_is_enemy))[0]
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
    def select_elemental_harmony_target_card(obs):
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


class ActionSpace(gym.Space):

    def __init__(self):
        self.action_type_cpace = gym.spaces.Discrete(5)     # throw card/elemental harmony/use skill/switch role/end round
        self.card_action_args_space = gym.spaces.Discrete(10) # 10 options for the argseter space of throw card/elemental harmony
        self.skill_action_args_space = gym.spaces.Discrete(4)  # 3/4 options for arg space for use skills
        self.switch_character_action_args_space = gym.spaces.Discrete(3)  # 3 options for arg space for switch role

        super().__init__(self.shape, self.dtype)
    
    def sample(self):
        action_type = self.action_type_space.sample()  # Randomly choose the type of action
        if action_type == 0 or action_type == 1:  # throw card/element harmony
            action_args = self.card_action_args_space.sample()  # Randomly select args for throw card/elemental harmony
        elif action_type == 2:  # use skills
            action_args = self.skill_action_args_space.sample()
        elif action_type == 3:  # switch role
            action_args = self.switch_character_action_args_space.sample()  # Randomly select args for switch role
        else:  # end round
            action_args = None
        return tnp.array({'action_type': action_type, 'action_args': action_args})
    
    # need 'contains'?
    
    @property
    def shape(self):
        return (2,)  # The shape of the action space is a tuple of length 2

    @property
    def dtype(self):
        return (int, int)  # The data type of the action space is a tuple containing two integers

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
