from typing import Union, Optional, Dict, Callable, List
import torch
import torch.nn as nn
import treetensor.torch as ttorch
from ding.torch_utils.network import VectorMerge
from ding.utils import SequenceType
from obs import ObservationSpace


class SubObsEncoder(nn.Module):
    r"""
    Overview:
        Encoder structure shared by other encoders except global_obs.
    """
    def __init__(self, input_size: int, output_size: int =256, hidden_size: Union[int, SequenceType] = 64):
        super(SubObsEncoder, self).__init__()
        self.sub_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
        
    def forward(self, sub_obs):
        return self.sub_encoder(sub_obs)

class DiceObsEncoder(nn.Module):
    r"""
    Overview:
        Encoder that encodes dice_obs.
    """
    def __init__(self, obs_space: ObservationSpace, output_size: int = 256):
        super(DiceObsEncoder, self).__init__()
        self.output_size = output_size
        # (10,)
        self.dice_obs_shape = 10
        self.encoder = SubObsEncoder(input_size=self.dice_obs_shape, output_size=output_size)
        
    def forward(self, dice_obs):
        return self.encoder(dice_obs)


class CharacterObsEncoder(nn.Module):
    r"""
    Overview:
        Encoder that encodes character_obs.
    """
    def __init__(self, obs_space: ObservationSpace, output_size: int = 256):
        super(CharacterObsEncoder, self).__init__()
        self.output_size = output_size
        self.characters_obs_shape = obs_space.character_is_alive.shape[0]*10
        self.encoder = SubObsEncoder(input_size=self.characters_obs_shape, output_size=output_size)
        
    def forward(self, character_obs):
        return self.encoder(character_obs)


class SkillObsEncoder(nn.Module):
    r"""
    Overview:
        Encoder that encodes skill_obs.
    """
    def __init__(self, obs_space: ObservationSpace, output_size: int = 256):
        super(SkillObsEncoder, self).__init__()
        self.output_size = output_size
        self.skills_obs_shape = obs_space.skill_is_available.shape[0]*3
        self.encoder = SubObsEncoder(input_size=self.skills_obs_shape, output_size=output_size)
        
    def forward(self, skill_obs):
        return self.encoder(skill_obs)


class CardObsEncoder(nn.Module):
    r"""
    Overview:
        Encoder that encodes card_obs.
    """
    def __init__(self, obs_space: ObservationSpace, output_size: int = 256):
        super(CardObsEncoder, self).__init__()
        self.output_size = output_size
        self.cards_obs_shape = obs_space.card_is_available.shape[0]*4
        self.encoder = SubObsEncoder(input_size=self.cards_obs_shape, output_size=output_size)
        
    def forward(self, card_obs):
        return self.encoder(card_obs)


class SummonerObsEncoder(nn.Module):
    r"""
    Overview:
        Encoder that encodes summoner_obs.
    """
    def __init__(self, obs_space: ObservationSpace, output_size: int = 256):
        super(SummonerObsEncoder, self).__init__()
        self.output_size = output_size
        self.summoners_obs_shape = obs_space.summoner_is_available.shape[0]*3
        self.encoder = SubObsEncoder(input_size=self.summoners_obs_shape, output_size=output_size)
        
    def forward(self, summoner_obs):
        return self.encoder(summoner_obs)


class SupporterObsEncoder(nn.Module):
    r"""
    Overview:
        Encoder that encodes supporter_obs.
    """
    def __init__(self, obs_space: ObservationSpace, output_size: int = 256):
        super(SupporterObsEncoder, self).__init__()
        self.output_size = output_size
        self.supporters_obs_shape = obs_space.supporter_is_available.shape[0]*3
        self.encoder = SubObsEncoder(input_size=self.supporters_obs_shape, output_size=output_size)
        
    def forward(self, supporter_obs):
        return self.encoder(supporter_obs)

class GlobalObsEncoder(nn.Module):
    r"""
    Overview:
        Encoder that encodes global_obs.
    """
    def __init__(self, input_size: ObservationSpace, output_size: int = 256, hidden_size: Union[int, SequenceType] = 256):
        super(GlobalObsEncoder, self).__init__()
        self.output_size = output_size
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
        
    def forward(self, gloabl_obs):
        return self.encoder(gloabl_obs)

class ObservationEncoder(nn.Module):
    r"""
    Overview:
        Encoder that processes and encodes the overall obs.
    """
    def __init__(self, obs_space:ObservationSpace, output_size=256, hidden_size=256):
        super(ObservationEncoder, self).__init__()

        # global obs (56,): one-hot last_play(2,), dice_num, (17, ), card_num, (11,), enemy_card_num (11,)
        self.global_encoder = GlobalObsEncoder(input_size=56)
        # sub-obs
        self.dice_encoder = DiceObsEncoder(obs_space)
        self.character_encoder = CharacterObsEncoder(obs_space)
        self.skill_encoder = SkillObsEncoder(obs_space)
        self.card_encoder = CardObsEncoder(obs_space)
        self.summoner_encoder = SummonerObsEncoder(obs_space)
        self.supporter_encoder = SupporterObsEncoder(obs_space)
        self.encoders = {
            'global': self.global_encoder,
            'dice': self.dice_encoder,
            'character': self.character_encoder,
            'skill': self.skill_encoder,
            'card': self.card_encoder,
            'summoner': self.summoner_encoder,
            'supporter': self.supporter_encoder
        }
        # merge
        obs_input_sizes = {
            'global_obs': self.global_encoder.output_size,
            'dice_obs': self.dice_encoder.output_size,
            'character_obs': self.character_encoder.output_size+obs_space.character_other_info.shape[0],
            'skill_obs': self.skill_encoder.output_size+obs_space.skill_other_info.shape[0],
            'card_obs': self.card_encoder.output_size+obs_space.card_other_info.shape[0],
            'summoner_obs': self.summoner_encoder.output_size+obs_space.summoner_other_info.shape[0],
            'supporter_obs': self.supporter_encoder.output_size+obs_space.supporter_other_info.shape[0],
        }
        self.obs_merge = VectorMerge(input_sizes=obs_input_sizes, output_size=output_size)
        
    def forward(self, observation:list, last_action:list):
        encoded_obs = {}
        new_obs = self.process_obs(observation,last_action)
        # Handle global and dice separately as they don't need concatenation
        encoded_obs['global_obs'] = self.encoders['global'](new_obs['global_obs'])
        encoded_obs['dice_obs'] = self.encoders['dice'](new_obs['dice_obs'])
        # Loop over the rest of the keys
        for key in ['character', 'skill', 'card', 'summoner', 'supporter']:
            encoded_obs[f'{key}_obs'] = torch.cat([
                self.encoders[key](new_obs[f'{key}_obs']),
                new_obs[f'{key}_other_info_obs']
            ], dim=1)
        # merge all obs
        merged_obs = self.obs_merge(encoded_obs)

        return merged_obs

    def process_obs(self, obs_list:list, last_action_list):
        processed_obs_lists = {
            "global_obs": [],
            "dice_obs": [],
            "character_obs": [],
            "skill_obs": [],
            "card_obs": [],
            "summoner_obs": [],
            "supporter_obs": [],
            "character_other_info_obs": [],
            "skill_other_info_obs": [],
            "card_other_info_obs": [],
            "summoner_other_info_obs": [],
            "supporter_other_info_obs": [],
        }

        for obs, last_action in zip(obs_list, last_action_list):
            obs = obs.to(torch.float32)
            # One-hot encoding for global obs
            last_action_type_one_hot = nn.functional.one_hot(last_action.action_type.to(torch.int64), num_classes=5).to(torch.float32)
            if last_action.action_args.item() == -1:
                last_action_args_one_hot = torch.zeros((10,),device=last_action.action_args.device)
            else:
                last_action_args_one_hot = nn.functional.one_hot(last_action.action_args.to(torch.int64), num_classes=10).to(torch.float32)
            last_play_one_hot = nn.functional.one_hot(((obs.last_play+2)%2).to(torch.int64), num_classes=2).to(torch.float32)   # obs.last_play will be -1/1
            dice_num_one_hot = nn.functional.one_hot(obs.dice_num.to(torch.int64), num_classes=17).to(torch.float32)
            card_num_one_hot = nn.functional.one_hot(obs.card_num.to(torch.int64), num_classes=11).to(torch.float32)
            enemy_card_num_one_hot = nn.functional.one_hot(obs.enemy_card_num.to(torch.int64), num_classes=11).to(torch.float32)
            global_obs = torch.cat([
                last_play_one_hot.squeeze(0),
                dice_num_one_hot.squeeze(0),
                card_num_one_hot.squeeze(0),
                enemy_card_num_one_hot.squeeze(0),
                last_action_type_one_hot,
                last_action_args_one_hot
            ])
            processed_obs_lists['global_obs'].append(global_obs)

            dice_obs = torch.cat([
                obs.dice_num,
                obs.opposite_dice_num,
                obs.colorless_dice_num,
                obs.pyro_dice_num,
                obs.hydro_dice_num,
                obs.electro_dice_num,
                obs.cryo_dice_num,
                obs.geo_dice_num,
                obs.anemo_dice_num,
                obs.dendro_dice_num
            ], dim=0)   # (10)
            processed_obs_lists['dice_obs'].append(dice_obs)

            character_obs = torch.cat([
                obs.character_is_alive,
                obs.character_is_battle,
                obs.character_hp,
                obs.character_charge_point,
                obs.character_charge_point_max,
                obs.character_weapon_type,
                obs.character_element_type,
                obs.character_is_enemy,
                obs.character_element_attachment,
                obs.character_is_full
            ], dim=0)  # (10*character_num)
            processed_obs_lists['character_obs'].append(character_obs)
            processed_obs_lists['character_other_info_obs'].append(obs.character_other_info)

            skill_obs = torch.cat([
                obs.skill_is_available,
                obs.skill_is_charge,
                obs.skill_direct_damage,
            ], dim=0)  # (3*max_skill_num )
            processed_obs_lists['skill_obs'].append(skill_obs)
            processed_obs_lists['skill_other_info_obs'].append(obs.skill_other_info)

            card_obs = torch.cat([
                obs.card_is_available,
                obs.card_is_same_dice,
                obs.card_dice_cost,
                obs.card_type,
            ], dim=0)  # (4*max_usable_card_num)
            processed_obs_lists['card_obs'].append(card_obs)
            processed_obs_lists['card_other_info_obs'].append(obs.card_other_info)

            summoner_obs = torch.cat([
                obs.summoner_is_available,
                obs.summoner_is_enemy,
                obs.summoner_remain_turn,
            ], dim=0)   # (3*max_summoner_num)
            processed_obs_lists['summoner_obs'].append(summoner_obs)
            processed_obs_lists['summoner_other_info_obs'].append(obs.summoner_other_info)
            
            supporter_obs = torch.cat([
                obs.supporter_is_available,
                obs.supporter_is_enemy,
                obs.supporter_count,
            ], dim=0)  # (3*max_supporter_num)
            processed_obs_lists['supporter_obs'].append(supporter_obs)
            processed_obs_lists['supporter_other_info_obs'].append(obs.supporter_other_info)

        processed_obs_stacks = {}
        for key in processed_obs_lists.keys():
            processed_obs_stacks[key] = torch.stack(processed_obs_lists[key], dim=0)

        return processed_obs_stacks
