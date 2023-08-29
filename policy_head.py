from typing import Union, Dict, Optional
from easydict import EasyDict
from dataclasses import asdict
import torch
import torch.nn as nn
from copy import deepcopy
from ding.utils import SequenceType, squeeze, MODEL_REGISTRY
from ding.model.common import ReparameterizationHead, RegressionHead, DiscreteHead, MultiHead, \
    FCEncoder, ConvEncoder, IMPALAConvEncoder
from action import ActionSpace, ActionType

class ActionArgHead(nn.Module):
    r"""
    Overview:
        
    Interfaces:
        ``__init__``, ``forward``
    """
    def __init__(
        self,
        encoded_part_obs_shape: Union[int, SequenceType],
        action_type_prob_shape,
        hidden_size,    # will be same as obs_embedding_shape
        ):
        super(ActionArgHead, self).__init__()
        self.W_k = nn.Linear(encoded_part_obs_shape, hidden_size)
        self.W_q = nn.Linear(action_type_prob_shape, hidden_size)

    def forward(
        self,
        obs_embedding,  # (B, hidden_size)
        action_type_prob,   # (B, action_type_num)
        encoded_part_obs     # shape (B, args_shape, encoded_part_obs_shape), part of encoded_obs related to the current arg_head
        ):
        # cross attention
        key = self.W_k(encoded_part_obs)   # (B, args_shape, hidden_size)
        key = (key - key.mean()) / (key.std() + 1e-8)

        query = self.W_q(action_type_prob) + obs_embedding     # (B, hidden_size)
        query = query.unsqueeze(1)      # (B, 1, hidden_size)

        logit = torch.matmul(query,key.permute(0, 2, 1))  # (B, 1, args_shape)
        logit = logit.squeeze(1)        # (B, args_shape)
        return logit


class GenshinVAC(nn.Module):
    r"""
    Overview:
        The VAC model for DI-Card.
    Interfaces:
        ``__init__``, ``forward``, ``compute_actor``, ``compute_critic``
    """
    mode = ['compute_actor', 'compute_critic', 'compute_actor_critic']
    # action_type_names is a dict corresponding to the sequence number and action type name
    # e.g {0:'play_card'}
    action_type_names = dict(zip(ActionType().__dict__.values(), ActionType().__dict__.keys()))
    # action_type_names = asdict(ActionType())
    # action_obs_name_map is used to match action names and encoded_obs names for action_args_head
    action_obs_name_map = {'play_card':'card_obs','use_skill':'skill_obs', 'change_character': 'character_obs'}
    def __init__(
        self,
        obs_embedding_shape: Union[int, SequenceType],
        action_space: ActionSpace,
        encoded_obs_shape: Dict,    # Should correspond to obs_merge_input_sizes in ObservationEncoder
        actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
    ) -> None:

        super(GenshinVAC, self).__init__()
        obs_embedding_shape: int = squeeze(obs_embedding_shape)
        # action_shape = squeeze(action_shape)  # will be a dict
        self.obs_embedding_shape = obs_embedding_shape

        self.critic_head = RegressionHead(
            obs_embedding_shape,
            1,
            critic_head_layer_num,
            activation=activation,
            norm_type=norm_type
        )
        
        # actor head
        # action type head
        self.actor_action_type = DiscreteHead(
                obs_embedding_shape,
                action_space['action_type_space'].n,
                actor_head_layer_num,
                activation=activation,
                norm_type=norm_type,
        )
        # three action args heads: 'play_card', 'use_skill', 'change_character'
        self.actor_action_args = nn.ModuleDict({
            action_name: ActionArgHead(
                encoded_part_obs_shape=encoded_obs_shape[self.action_obs_name_map[action_name]],    # e.g. encoded_obs_shape['card_obs']
                action_type_prob_shape=action_space['action_type_space'].n,
                hidden_size=obs_embedding_shape
            )
            for action_name in action_space['action_arg_space'] if action_name not in ['elemental_harmony', 'terminate_turn']
        })
        self.actor_head = nn.ModuleList([self.actor_action_type, self.actor_action_args])

    def forward(self, mode: str, **inputs) -> Dict:
        # TODO: How to deal with the parameters here, there are two different sets of parameters
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(**inputs)

    def compute_actor(
        self,
        obs_embedding,
        encoded_obs,
        sample_action_type: str = 'argmax',
        selected_action_type =None,     # action_type selected outside
        ) -> Dict:
        # sample_action_type could be 'argmax' or 'normal'
        assert sample_action_type in ["argmax", "normal"], "sample_action_type should be 'argmax' or 'normal'"
        action_type_logit = self.actor_action_type(obs_embedding)['logit']
        action_type_prob = torch.softmax(action_type_logit, dim=-1)
        if selected_action_type is not None:
            action_type = selected_action_type
        else:
            action_type = torch.multinomial(action_type_prob, 1).item() if sample_action_type == 'normal'\
                            else torch.argmax(action_type_prob, 1).item()
        action_args_logit = {}
        if not self.training:
            # If it is not training mode, output action_type and the distribution of a single action_arg
            # selected by the corresponding sampling method
            select_action_name = self.action_type_names[action_type]
            if select_action_name in self.actor_action_args.keys():
                # Check if the selected action_type has action_args
                select_encoded_obs_name = self.action_obs_name_map[select_action_name]
                action_args_logit[select_action_name] = self.actor_action_args[select_action_name](
                    obs_embedding=obs_embedding,
                    action_type_prob=action_type_prob,
                    encoded_part_obs=encoded_obs[select_encoded_obs_name]
                )
            else:
                # when there is no or need to evaluate action_args with rules. should return None
                action_args_logit = None
        else:
            for action_type_name in self.action_obs_name_map.keys():
                encoded_obs_name = self.action_obs_name_map[action_type_name]
                action_args_logit[action_type_name] = self.actor_action_args[action_type_name](
                    obs_embedding=obs_embedding,
                    action_type_prob=action_type_prob,
                    encoded_part_obs=encoded_obs[encoded_obs_name]
                )
        # action_type_logit: one distribution
        # action_args(dict): 1 distribution(train mode)/3 distribution(test mode)
        return {'logit': {'action_type': action_type_logit, 'action_args': action_args_logit}}

    def compute_critic(self, obs_embedding) -> Dict:
        x = self.critic_head(obs_embedding)
        return {'value': x['pred']}

    def compute_actor_critic(
        self,
        obs_embedding,
        encoded_obs,
        sample_action_type:str='argmax',
        selected_action_type =None,
        ) -> Dict:
        value = self.critic_head(obs_embedding)['pred']

        assert sample_action_type in ["argmax", "normal"], "sample_action_type should be 'argmax' or 'normal'"
        action_type_logit = self.actor_action_type(obs_embedding)['logit']
        action_type_prob = torch.softmax(action_type_logit, dim=-1)

        if selected_action_type is not None:
            action_type = selected_action_type
        else:
            action_type = torch.multinomial(action_type_prob, 1).item() if sample_action_type == 'normal'\
                            else torch.argmax(action_type_prob, 1).item()
        action_args_logit = {}
        if not self.training:
            # If it is not training mode, output action_type and the distribution of a single action_arg
            # selected by the corresponding sampling method
            select_action_name = self.action_type_names[action_type]
            if select_action_name in self.actor_action_args.keys():
                # Check if the selected action_type has action_args
                select_encoded_obs_name = self.action_obs_name_map[select_action_name]
                action_args_logit[select_action_name] = self.actor_action_args[select_action_name](
                    obs_embedding=obs_embedding,
                    action_type_prob=action_type_prob,
                    encoded_part_obs=encoded_obs[select_encoded_obs_name]
                )
            else:
                # There is no or need to evaluate action_args with rules. should return None
                action_args_logit = None
        else:
            for action_type_name in self.action_obs_name_map.keys():
                encoded_obs_name = self.action_obs_name_map[action_type_name]
                action_args_logit[action_type_name] = self.actor_action_args[action_type_name](
                    obs_embedding=obs_embedding,
                    action_type_prob=action_type_prob,
                    encoded_part_obs=encoded_obs[encoded_obs_name]
                )

        return {'logit': {'action_type': action_type_logit, 'action_args': action_args_logit}, 'value': value}

