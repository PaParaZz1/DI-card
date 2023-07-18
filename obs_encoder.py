import treetensor.torch as ttorch
import torch
import torch.nn as nn
from obs import ObservationSpace

class DiceObsEncoder(nn.Module):
    def __init__(self, obs_space):
        super(DiceObsEncoder, self).__init__()
        # (10,)
        self.dice_obs_shape = 10
        self.encoder = nn.Sequential(
            nn.Linear(self.dice_obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
    def forward(self, dice_obs):
        return self.encoder(dice_obs)


class CharacterObsEncoder(nn.Module):
    def __init__(self, obs_space):
        super(CharacterObsEncoder, self).__init__()

        self.num_characters = obs_space.character_is_alive.shape[0]
        self.encoder = nn.Sequential(
            nn.Linear(self.num_characters, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
    def forward(self, character_obs):
        return self.encoder(character_obs)


class SkillObsEncoder(nn.Module):
    def __init__(self, obs_space):
        super(SkillObsEncoder, self).__init__()

        self.num_skills = obs_space.skill_is_available.shape[0]
        self.encoder = nn.Sequential(
            nn.Linear(self.num_skills, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
    def forward(self, skill_obs):
        return self.encoder(skill_obs)


class CardObsEncoder(nn.Module):
    def __init__(self, obs_space):
        super(CardObsEncoder, self).__init__()

        self.num_cards = obs_space.card_is_available.shape[0]
        self.encoder = nn.Sequential(
            nn.Linear(self.num_cards, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
    def forward(self, card_obs):
        return self.encoder(card_obs)


class SummonerObsEncoder(nn.Module):
    def __init__(self, obs_space):
        super(SummonerObsEncoder, self).__init__()

        self.num_summoners = obs_space.summoner_is_available.shape[0]
        self.encoder = nn.Sequential(
            nn.Linear(self.num_summoners, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
    def forward(self, summoner_obs):
        return self.encoder(summoner_obs)


class SupporterObsEncoder(nn.Module):
    def __init__(self, obs_space):
        super(SupporterObsEncoder, self).__init__()

        self.num_supporters = obs_space.supporter_is_available.shape[0]
        self.encoder = nn.Sequential(
            nn.Linear(self.num_supporters, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
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
        
    def forward(self, observation:ttorch.tensor):
        encoded_observation = {}
        new_obs = self.process_obs(observation)
        encoded_observation['dice_obs'] = self.dice_encoder(new_obs['dice_obs'])
        encoded_observation['character_obs'] = self.character_encoder(new_obs['character_obs'])
        encoded_observation['skill_obs'] = self.skill_encoder(new_obs['skill_obs'])
        encoded_observation['card_obs'] = self.card_encoder(new_obs['card_obs'])
        encoded_observation['summoner_obs'] = self.summoner_encoder(new_obs['summoner_obs'])
        encoded_observation['supporter_obs'] = self.supporter_encoder(new_obs['supporter_obs'])
        
        return encoded_observation

    def process_obs(self, obs):
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
        ], dim=0)   # (10,)

        character_obs = torch.stack([
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
        ], dim=0)   # (10,character_num)

        skill_obs = torch.stack([
            obs.skill_is_available,
            obs.skill_is_charge,
            obs.skill_direct_damage,
        ], dim=0)   # (3,max_skill_num )

        card_obs = torch.stack([
            obs.card_is_available,
            obs.card_is_same_dice,
            obs.card_dice_cost,
            obs.card_type,
        ], dim=0)   # (4, max_usable_card_num)

        summoner_obs = torch.stack([
            obs.summoner_is_available,
            obs.summoner_is_enemy,
            obs.summoner_remain_turn,
        ], dim=0)   # (3, max_summoner_num)
        
        supporter_obs = torch.stack([
            obs.supporter_is_available,
            obs.supporter_is_enemy,
            obs.supporter_count,
        ], dim=0)   # (3, max_supporter_num)

        return {
            'dice_obs': dice_obs,
            'character_obs': character_obs,
            'skill_obs': skill_obs,
            'card_obs': card_obs,
            'summoner_obs': summoner_obs,
            'supporter_obs': supporter_obs
        }

