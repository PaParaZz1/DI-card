from collections import namedtuple
from enum import IntEnum, unique
from typing import Tuple

import numpy as np
import six
import treetensor.numpy as tnp


@unique
class Character(IntEnum):
    pass


@unique
class Card(IntEnum):
    pass


@unique
class SkillType(IntEnum):
    pass


@unique
class FeatureType(IntEnum):
    SCALAR: int = 0
    CATEGORICAL: int = 1
    DISCRETE: int = 2


@unique
class ElementalDiceType(IntEnum):
    colorless: int = 0
    pyro: int = 1
    hydro: int = 2
    electro: int = 3
    cryo: int = 4
    geo: int = 5
    anemo: int = 6
    dendro: int = 7


@unique
class WeaponType(IntEnum):
    sword: int = 0
    claymore: int = 1
    polearm: int = 2
    bow: int = 3
    catalyst: int = 4


class Space:

    def __init__(self, shape: Tuple[int, ...], type_: FeatureType, range_: Tuple[int, ...]) -> None:
        self.shape = shape
        self.type_ = type_
        self.range_ = range_

    def sample(self):
        # scalar value, format: (low, high), create float number in [low, high]
        if self.type_ == FeatureType.SCALAR:
            data = np.random.random(size=self.shape)
            low, high = self.range_
            return data * (high - low) + low

        # categorical value, format: (x, upper), create integer number with in [0, upper)
        # (so x is just bullshit?)
        elif self.type_ == FeatureType.CATEGORICAL:
            return np.random.randint(0, self.range_[1], size=self.shape)

        # discrete values, format [v1, v2, ..., vn], create random objects with in [v1, v2, ..., vn]
        elif self.type_ == FeatureType.DISCRETE:
            return np.random.choice(self.range_, size=self.shape, replace=False)

        else:
            assert False, f'Unknown space type {self.type_!r}, should not reach this line!'  # pragma: no cover

    def get_zero_data(self):
        if self.type_ == FeatureType.SCALAR:
            dtype = np.float32
        else:
            dtype = np.int64  # uint8
        return np.zeros(shape=self.shape, dtype=dtype)


class ObservationSpace(namedtuple('ObservationSpace', (
        'last_play',
        'dice_num',
        'opposite_dice_num',
        'colorless_dice_num',
        'pyro_dice_num',
        'hydro_dice_num',
        'electro_dice_num',
        'cryo_dice_num',
        'geo_dice_num',
        'anemo_dice_num',
        'dendro_dice_num',
        'character_is_alive',
        'character_is_battle',
        'character_hp',
        'character_charge_point',
        'character_charge_point_max',
        'character_weapon_type',
        'character_element_type',
        'character_is_enemy',
        'character_element_attachment',
        'character_is_full',
        'character_other_info',
        'skill_is_available',
        'skill_is_charge',
        'skill_direct_damage',
        'skill_other_info',
        'card_is_available',
        'card_is_same_dice',
        'card_dice_cost',
        'card_type',
        'card_other_info',
        'card_num',
        'enemy_card_num',
        'summoner_is_available',
        'summoner_is_enemy',
        'summoner_remain_turn',
        'summoner_other_info',
        'supporter_is_available',
        'supporter_is_enemy',
        'supporter_count',
        'supporter_other_info',
))):
    __slots__ = ()

    def __new__(cls, **kwargs):
        obs = {}
        for name, (shape, type_, range_) in six.iteritems(kwargs):
            obs[name] = Space(shape, type_, range_)
        return super(ObservationSpace, cls).__new__(cls, **obs)

    def sample(self):
        data = {}
        for f in self._fields:
            s = getattr(self, f)
            data[f] = s.sample()
        return tnp.array(data)

    def get_zero_data(self):
        data = {}
        for f in self._fields:
            s = getattr(self, f)
            data[f] = s.get_zero_data()
        return tnp.array(data)


def get_observation_space(
        character_num: int, embedding_num: int, max_skill_num: int, max_usable_card_num: int, max_usable_dice_num: int,
        max_summoner_num: int, max_supporter_num: int
) -> ObservationSpace:
    return ObservationSpace(
        last_play=((1,), FeatureType.DISCRETE, (-1, 1)),
        dice_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_dice_num + 1)),
        opposite_dice_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_dice_num + 1)),
        colorless_dice_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_dice_num + 1)),
        pyro_dice_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_dice_num + 1)),
        hydro_dice_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_dice_num + 1)),
        electro_dice_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_dice_num + 1)),
        cryo_dice_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_dice_num + 1)),
        geo_dice_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_dice_num + 1)),
        anemo_dice_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_dice_num + 1)),
        dendro_dice_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_dice_num + 1)),
        character_is_alive=((character_num,), FeatureType.CATEGORICAL, (0, 2)),
        character_is_battle=((character_num,), FeatureType.CATEGORICAL, (0, 2)),
        character_hp=((character_num,), FeatureType.CATEGORICAL, (0, 11)),
        character_charge_point=((character_num,), FeatureType.CATEGORICAL, (0, 4)),
        character_charge_point_max=((character_num,), FeatureType.CATEGORICAL, (0, 4)),
        character_weapon_type=((character_num,), FeatureType.CATEGORICAL, (0, 5)),
        character_element_type=((character_num,), FeatureType.CATEGORICAL, (0, 7)),
        character_is_enemy=((character_num,), FeatureType.CATEGORICAL, (0, 2)),
        character_element_attachment=((character_num,), FeatureType.CATEGORICAL, (0, 8)),  # 7+1
        character_is_full=((character_num,), FeatureType.CATEGORICAL, (0, 2)),
        # weapon, holy relic, food, skill, card
        character_other_info=((character_num * embedding_num,), FeatureType.SCALAR, (-1, 1)),
        skill_is_available=((max_skill_num,), FeatureType.CATEGORICAL, (0, 2)),
        skill_is_charge=((max_skill_num,), FeatureType.CATEGORICAL, (0, 2)),
        skill_direct_damage=((max_skill_num,), FeatureType.CATEGORICAL, (0, 8)),
        # TODO dynamic skill cost
        skill_other_info=((max_skill_num * embedding_num,), FeatureType.SCALAR, (-1, 1)),
        # TODO enemy skill information
        card_is_available=((max_usable_card_num,), FeatureType.CATEGORICAL, (0, 2)),
        card_is_same_dice=((max_usable_card_num,), FeatureType.CATEGORICAL, (0, 2)),
        card_dice_cost=((max_usable_card_num,), FeatureType.CATEGORICAL, (0, 4)),
        card_type=((max_usable_card_num,), FeatureType.CATEGORICAL, (0, 3)),  # event, support, equipment
        card_other_info=((max_usable_card_num * embedding_num,), FeatureType.SCALAR, (-1, 1)),
        card_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_card_num + 1)),
        enemy_card_num=((1,), FeatureType.CATEGORICAL, (0, max_usable_card_num + 1)),
        summoner_is_available=((max_summoner_num,), FeatureType.CATEGORICAL, (0, 2)),
        summoner_is_enemy=((max_summoner_num,), FeatureType.CATEGORICAL, (0, 2)),
        summoner_remain_turn=((max_summoner_num,), FeatureType.CATEGORICAL, (0, 5)),  # 3+1+1
        summoner_other_info=((max_summoner_num * embedding_num,), FeatureType.SCALAR, (-1, 1)),
        supporter_is_available=((max_supporter_num,), FeatureType.CATEGORICAL, (0, 2)),
        supporter_is_enemy=((max_supporter_num,), FeatureType.CATEGORICAL, (0, 2)),
        supporter_count=((max_supporter_num,), FeatureType.CATEGORICAL, (0, 4)),
        supporter_other_info=((max_supporter_num * embedding_num,), FeatureType.SCALAR, (-1, 1)),
    )
