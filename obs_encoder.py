import treetensor.torch as ttorch
import torch
import torch.nn as nn
from ding.torch_utils import to_device
from obs import ObservationSpace
from ding.torch_utils.network import VectorMerge

class SubObsEncoder(nn.Module):
    def __init__(self, input_size, output_size=256, hidden_size=64):
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
    def __init__(self, obs_space, output_size=256):
        super(DiceObsEncoder, self).__init__()
        self.output_size = output_size
        # (10,)
        self.dice_obs_shape = 10
        self.encoder = SubObsEncoder(input_size=self.dice_obs_shape, output_size=output_size)
        
    def forward(self, dice_obs):
        return self.encoder(dice_obs)


class CharacterObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(CharacterObsEncoder, self).__init__()
        self.output_size = output_size
        self.characters_obs_shape = obs_space.character_is_alive.shape[0]*10
        self.encoder = SubObsEncoder(input_size=self.characters_obs_shape, output_size=output_size)
        
    def forward(self, character_obs):
        return self.encoder(character_obs)


class SkillObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(SkillObsEncoder, self).__init__()
        self.output_size = output_size
        self.skills_obs_shape = obs_space.skill_is_available.shape[0]*3
        self.encoder = SubObsEncoder(input_size=self.skills_obs_shape, output_size=output_size)
        
    def forward(self, skill_obs):
        return self.encoder(skill_obs)


class CardObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(CardObsEncoder, self).__init__()
        self.output_size = output_size
        self.cards_obs_shape = obs_space.card_is_available.shape[0]*4
        self.encoder = SubObsEncoder(input_size=self.cards_obs_shape, output_size=output_size)
        
    def forward(self, card_obs):
        return self.encoder(card_obs)


class SummonerObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(SummonerObsEncoder, self).__init__()
        self.output_size = output_size
        self.summoners_obs_shape = obs_space.summoner_is_available.shape[0]*3
        self.encoder = SubObsEncoder(input_size=self.summoners_obs_shape, output_size=output_size)
        
    def forward(self, summoner_obs):
        return self.encoder(summoner_obs)


class SupporterObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(SupporterObsEncoder, self).__init__()
        self.output_size = output_size
        self.supporters_obs_shape = obs_space.supporter_is_available.shape[0]*3
        self.encoder = SubObsEncoder(input_size=self.supporters_obs_shape, output_size=output_size)
        
    def forward(self, supporter_obs):
        return self.encoder(supporter_obs)

class GlobalObsEncoder(nn.Module):
    def __init__(self, input_size, output_size=256, hidden_size=256):
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
    def __init__(self, obs_space:ObservationSpace, output_size=256, hidden_size=256, device='cpu'):
        super(ObservationEncoder, self).__init__()

        # global obs
        self.global_encoder = GlobalObsEncoder(input_size=19)
        # sub-obs
        self.dice_encoder = DiceObsEncoder(obs_space)
        self.character_encoder = CharacterObsEncoder(obs_space)
        self.skill_encoder = SkillObsEncoder(obs_space)
        self.card_encoder = CardObsEncoder(obs_space)
        self.summoner_encoder = SummonerObsEncoder(obs_space)
        self.supporter_encoder = SupporterObsEncoder(obs_space)
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
        self._device = device
        to_device(self, self._device)
        
    def forward(self, observation:list, last_action:list):
        encoded_obs = {}
        new_obs = self.process_obs(observation,last_action)
        encoded_obs['global_obs'] = self.global_encoder(new_obs['global_obs'])
        encoded_obs['dice_obs'] = self.dice_encoder(new_obs['dice_obs'])
        encoded_obs['character_obs'] = torch.cat([
            self.character_encoder(new_obs['character_obs']),
            new_obs['character_other_info_obs']
        ], dim=1)
        encoded_obs['skill_obs'] = torch.cat([
            self.skill_encoder(new_obs['skill_obs']),
            new_obs['skill_other_info_obs']
        ], dim=1)
        encoded_obs['card_obs'] = torch.cat([
            self.card_encoder(new_obs['card_obs']),
            new_obs['card_other_info_obs']
        ], dim=1)
        encoded_obs['summoner_obs'] = torch.cat([
            self.summoner_encoder(new_obs['summoner_obs']),
            new_obs['summoner_other_info_obs']
        ], dim=1)
        encoded_obs['supporter_obs'] = torch.cat([
            self.supporter_encoder(new_obs['supporter_obs']),
            new_obs['supporter_other_info_obs']
        ], dim=1)

        # merge all obs
        merged_obs = self.obs_merge(encoded_obs)

        return merged_obs

    def process_obs(self, obs_list:list, last_action_list):
        batch_size = len(obs_list)
        global_obs_list = []
        dice_obs_list = []
        character_obs_list = []
        skill_obs_list = []
        card_obs_list = []
        summoner_obs_list = []
        supporter_obs_list = []
        character_other_info_list = []
        skill_other_info_list = []
        card_other_info_list = []
        summoner_other_info_list = []
        supporter_other_info_list = []
        obs_list = to_device(obs_list,self._device)
        last_action_list = to_device(last_action_list,self._device)
        for obs, last_action in zip(obs_list, last_action_list):
            obs = obs.to(torch.float32)
            last_action_type_one_hot = nn.functional.one_hot(last_action.action_type.to(torch.int64), num_classes=5).to(torch.float32)
            if last_action.action_args == -1:
                last_action_args_one_hot = to_device(torch.zeros((10,)),self._device)
            else:
                last_action_args_one_hot = nn.functional.one_hot(last_action.action_args.to(torch.int64), num_classes=10).to(torch.float32)
            # last_action = last_action.to(torch.float32)
            global_obs = torch.cat([
                obs.last_play,
                obs.dice_num,
                obs.card_num,
                obs.enemy_card_num,
                last_action_type_one_hot,
                last_action_args_one_hot
            ])
            global_obs_list.append(global_obs)

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
            dice_obs_list.append(dice_obs)

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
            ], dim=0)   # (10*character_num)
            character_obs_list.append(character_obs)
            character_other_info_list.append(obs.character_other_info)

            skill_obs = torch.cat([
                obs.skill_is_available,
                obs.skill_is_charge,
                obs.skill_direct_damage,
            ], dim=0)   # (3*max_skill_num )
            skill_obs_list.append(skill_obs)
            skill_other_info_list.append(obs.skill_other_info)

            card_obs = torch.cat([
                obs.card_is_available,
                obs.card_is_same_dice,
                obs.card_dice_cost,
                obs.card_type,
            ], dim=0)   # (4*max_usable_card_num)
            card_obs_list.append(card_obs)
            card_other_info_list.append(obs.card_other_info)

            summoner_obs = torch.cat([
                obs.summoner_is_available,
                obs.summoner_is_enemy,
                obs.summoner_remain_turn,
            ], dim=0)   # (3*max_summoner_num)
            summoner_obs_list.append(summoner_obs)
            summoner_other_info_list.append(obs.summoner_other_info)
            
            supporter_obs = torch.cat([
                obs.supporter_is_available,
                obs.supporter_is_enemy,
                obs.supporter_count,
            ], dim=0)   # (3*max_supporter_num)
            supporter_obs_list.append(supporter_obs)
            supporter_other_info_list.append(obs.supporter_other_info)

        global_obs = torch.stack(global_obs_list, dim=0)
        dice_obs = torch.stack(dice_obs_list, dim=0)
        character_obs = torch.stack(character_obs_list, dim=0)
        skill_obs = torch.stack(skill_obs_list, dim=0)
        card_obs = torch.stack(card_obs_list, dim=0)
        summoner_obs = torch.stack(summoner_obs_list, dim=0)
        supporter_obs = torch.stack(supporter_obs_list, dim=0)

        character_other_info_obs = torch.stack(character_other_info_list, dim=0)
        skill_other_info_obs = torch.stack(skill_other_info_list, dim=0)
        card_other_info_obs = torch.stack(card_other_info_list, dim=0)
        summoner_other_info_obs = torch.stack(summoner_other_info_list, dim=0)
        supporter_other_info_obs = torch.stack(supporter_other_info_list, dim=0)

        return {
            'global_obs': global_obs,
            'dice_obs': dice_obs,
            'character_obs': character_obs,
            'skill_obs': skill_obs,
            'card_obs': card_obs,
            'summoner_obs': summoner_obs,
            'supporter_obs': supporter_obs,
            'character_other_info_obs': character_other_info_obs,
            'skill_other_info_obs': skill_other_info_obs,
            'card_other_info_obs': card_other_info_obs,
            'summoner_other_info_obs': summoner_other_info_obs,
            'supporter_other_info_obs': supporter_other_info_obs,
        }

