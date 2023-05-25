from typing import List
from dataclasses import dataclass
import numpy as np
import treetensor.numpy as tnp
from obs import Character


class BasicRuleUtilities:

    @staticmethod
    def replace_card_at_beginning():
        raise NotImplementedError

    @staticmethod
    def reroll_dice():
        raise NotImplementedError

    @staticmethod
    def select_skill_target_character():
        """skill contains cards and character skills"""
        raise NotImplementedError

    @staticmethod
    def select_skill_dice():
        raise NotImplementedError

    @staticmethod
    def select_elemental_harmony_target_card():
        raise NotImplementedError

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


class ActionSpace():

    def __init__(self):
        pass

    def sample(self, obs) -> tnp.ndarray:
        # action_type
        p = np.array([1. for _ in range(5)])
        if obs.card_num == 0:
            p[0] = 0.
            p[1] = 0.
        if obs.skill_is_available.sum() == 0:
            p[2] = 0.
        if obs.character_is_alive[:3].sum() <= 1:
            p[3] = 0.
        p /= p.sum()
        action_type = np.random.choice(5, p=p)
        # action_args
        if action_type == 0:
            choice = obs.card_is_available
            p = choice / choice.sum()
            action_args = np.random.choice(choice, p=p)
        elif action_type == 1:
            action_args = np.random.choice(obs.card_num)
        elif action_type == 2:
            choice = obs.skill_is_available
            p = choice / choice.sum()
            action_args = np.random.choice(choice, p=p)
        elif action_type == 3:
            choice = obs.character_is_alive[:3] * (1 - obs.character_is_battle[:3])
            p = choice / choice.sum()
            action_args = np.random.choice(choice, p=p)
        else:
            action_args = 0
        # np.int64
        return tnp.array({'action_type': action_type, 'action_args': action_args})


def get_action_space(character_list: List[Character]):
    return ActionSpace()
