import treetensor.torch as ttorch
import torch
import torch.nn as nn
from obs import ObservationSpace

class SubObsEncoder(nn.Module):
    def __init__(self, input_size, output_size=256):
        super(SubObsEncoder, self).__init__()
        self.sub_encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.ReLU()
        )
        
    def forward(self, sub_obs):
        return self.sub_encoder(sub_obs)

class DiceObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(DiceObsEncoder, self).__init__()
        # (10,)
        self.dice_obs_shape = 10
        self.encoder = SubObsEncoder(input_size=self.dice_obs_shape, output_size=output_size)
        
    def forward(self, dice_obs):
        return self.encoder(dice_obs)


class CharacterObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(CharacterObsEncoder, self).__init__()

        self.characters_obs_shape = obs_space.character_is_alive.shape[0]*10
        self.encoder = SubObsEncoder(input_size=self.characters_obs_shape, output_size=output_size)
        
    def forward(self, character_obs):
        return self.encoder(character_obs)


class SkillObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(SkillObsEncoder, self).__init__()

        self.skills_obs_shape = obs_space.skill_is_available.shape[0]*3
        self.encoder = SubObsEncoder(input_size=self.skills_obs_shape, output_size=output_size)
        
    def forward(self, skill_obs):
        return self.encoder(skill_obs)


class CardObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(CardObsEncoder, self).__init__()

        self.cards_obs_shape = obs_space.card_is_available.shape[0]*4
        self.encoder = SubObsEncoder(input_size=self.cards_obs_shape, output_size=output_size)
        
    def forward(self, card_obs):
        return self.encoder(card_obs)


class SummonerObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(SummonerObsEncoder, self).__init__()

        self.summoners_obs_shape = obs_space.summoner_is_available.shape[0]*3
        self.encoder = SubObsEncoder(input_size=self.summoners_obs_shape, output_size=output_size)
        
    def forward(self, summoner_obs):
        return self.encoder(summoner_obs)


class SupporterObsEncoder(nn.Module):
    def __init__(self, obs_space, output_size=256):
        super(SupporterObsEncoder, self).__init__()

        self.supporters_obs_shape = obs_space.supporter_is_available.shape[0]*3
        self.encoder = SubObsEncoder(input_size=self.supporters_obs_shape, output_size=output_size)
        
    def forward(self, supporter_obs):
        return self.encoder(supporter_obs)


class ObservationEncoder(nn.Module):
    def __init__(self, obs_space:ObservationSpace):
        super(ObservationEncoder, self).__init__()

        self.dice_encoder = DiceObsEncoder(obs_space)
        self.character_encoder = CharacterObsEncoder(obs_space)
        self.skill_encoder = SkillObsEncoder(obs_space)
        self.card_encoder = CardObsEncoder(obs_space)
        self.summoner_encoder = SummonerObsEncoder(obs_space)
        self.supporter_encoder = SupporterObsEncoder(obs_space)
        
    def forward(self, observation:list):
        encoded_observation = {}
        new_obs = self.process_obs(observation)
        encoded_observation['dice_obs'] = self.dice_encoder(new_obs['dice_obs'])
        encoded_observation['character_obs'] = torch.cat([
            self.character_encoder(new_obs['character_obs']),
            new_obs['character_other_info_obs']
        ], dim=1)
        encoded_observation['skill_obs'] = torch.cat([
            self.skill_encoder(new_obs['skill_obs']),
            new_obs['skill_other_info_obs']
        ], dim=1)
        encoded_observation['card_obs'] = torch.cat([
            self.card_encoder(new_obs['card_obs']),
            new_obs['card_other_info_obs']
        ], dim=1)
        encoded_observation['summoner_obs'] = torch.cat([
            self.summoner_encoder(new_obs['summoner_obs']),
            new_obs['summoner_other_info_obs']
        ], dim=1)
        encoded_observation['supporter_obs'] = torch.cat([
            self.supporter_encoder(new_obs['supporter_obs']),
            new_obs['summoner_other_info_obs']
        ], dim=1)
        
        return encoded_observation

    def process_obs(self, obs_list:list):
        batch_size = len(obs_list)
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
        for obs in obs_list:
            obs = obs.to(torch.float32)
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

