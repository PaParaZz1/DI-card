import torch
import torch.nn as nn
from typing import Union, Dict, Optional

from ding.utils import SequenceType
from obs import ObservationSpace
from action import ActionSpace
from encoder import ObservationEncoder
from head import GenshinVAC

class CardNetwork(nn.Module):
    """
    Overview:
        Network including encoder and head.
    """

    def __init__(
        self,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        encoder_output_size=256,
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
    ):
        super(CardNetwork, self).__init__()
        self.encoder = ObservationEncoder(obs_space, output_size=encoder_output_size)
        self.head = GenshinVAC(
                obs_embedding_shape=encoder_output_size,
                action_space=action_space,
                encoded_obs_shape=self.encoder.obs_merge_input_sizes,
                actor_head_layer_num = actor_head_layer_num,
                critic_head_hidden_size = critic_head_hidden_size,
                critic_head_layer_num = critic_head_layer_num,
                activation = activation,
                norm_type = norm_type,
            )
    
    def forward(
            self,
            observation: list,
            last_action: list,
            head_mode: str,
            sample_action_type: str = 'argmax',
            selected_action_type=None,
            ):
        encoded_obs = self.encoder(observation, last_action)
        obs_embedding = encoded_obs['merged_obs']
        encoded_obs_for_head = encoded_obs['obs_for_head']
        if head_mode=='compute_critic':
            outputs = self.head.forward(
                mode=head_mode,
                obs_embedding=obs_embedding,
            )
        elif head_mode=='compute_actor_critic' or head_mode=='compute_actor':
            outputs = self.head.forward(
                mode=head_mode,
                obs_embedding=obs_embedding,
                encoded_obs=encoded_obs_for_head,
                selected_action_type=selected_action_type,
                sample_action_type=sample_action_type,
            )
        else:
            raise KeyError("Invalid head_mode: {}".format(head_mode))
        return outputs
