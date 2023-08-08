from typing import Union, Dict, Optional
from easydict import EasyDict
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
    def __init__(self, obs_embedding_shape, action_type_logit_shape, args_shape, hidden_size):
        super(ActionArgHead, self).__init__()
        self.args_shape = args_shape
        self.hidden_size = hidden_size
        self.W_k = nn.Linear(obs_embedding_shape, hidden_size)
        self.W_q = nn.Linear(action_type_logit_shape, hidden_size*args_shape)

    def forward(
        self,
        obs_embedding,  # (B, hidden_size)
        action_type_logit   # (B, action_type_num)
        ):
        # cross attention
        key = self.W_k(obs_embedding)   # (B, M*hidden_size)
        key = key.view(-1, self.args_shape, self.hidden_size)   # (B, M, hidden_size)

        query = self.W_q(action_type_logit) + obs_embedding     # (B, hidden_size)
        query = query.unsqueeze(1)      # (B, 1, hidden_size)

        logit = (key * query).sum(2)    # (B, M)
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
    action_type_names = {getattr(ActionType, attr): attr for attr in dir(ActionType) if not callable(getattr(ActionType, attr)) and not attr.startswith("__")}
    def __init__(
        self,
        obs_embedding_shape: Union[int, SequenceType],
        action_shape: Union[int, SequenceType, EasyDict],
        action_space: ActionSpace,
        # actor_head_hidden_size: int = 64,
        # actor_head_layer_num: int = 1,
        critic_head_hidden_size: int = 64,
        critic_head_layer_num: int = 1,
        activation: Optional[nn.Module] = nn.ReLU(),
        norm_type: Optional[str] = None,
    ) -> None:

        super(GenshinVAC, self).__init__()
        obs_embedding_shape: int = squeeze(obs_embedding_shape)
        # action_shape = squeeze(action_shape)  # will be a dict
        self.obs_embedding_shape, self.action_shape = obs_embedding_shape, action_shape

        self.critic_head = RegressionHead(
            critic_head_hidden_size,
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
        # three action args heads
        self.actor_action_args = nn.ModuleDict({
            action_name: ActionArgHead(
                obs_embedding_shape,
                action_space['action_type_space'].n,
                action_space['action_arg_space'][action_name].n,
                hidden_size=obs_embedding_shape
            )
            for action_name in action_space['action_arg_space'] if action_name not in ['elemental_harmony', 'end_round']
        })
        self.actor = nn.ModuleList([self.actor_action_type, self.actor_action_args])

    def forward(self, inputs: Union[torch.Tensor, Dict], mode: str) -> Dict:
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_actor(
        self,
        obs_embedding,
        sample_action_type:str='argmax',
        ) -> Dict:
        # sample_action_type could be 'argmax' or 'normal'
        assert sample_action_typein ["argmax", "normal"], "sample_action_type should be 'argmax' or 'normal'"
        action_type_logit = self.actor_action_type(obs_embedding)['logit']

        if sample_action_type == 'normal':
            action_type = torch.multinomial(action_type_logit, 1).item()
            action_args_logit = self.actor_action_args[action_type_names[action_type]](obs_embedding, action_type_logit)
            action_args = torch.multinomial(action_args_logit, 1).item()
        elif sample_action_type == 'argmax':
            action_type = torch.argmax(action_type_logit).item()
            action_args_logit = self.actor_action_args[action_type_names[action_type]](obs_embedding, action_type_logit)
            action_args = torch.argmax(action_args_logit, 1).item()
        
        return {'logit': {'action_type': action_type, 'action_args': action_args}}

    def compute_critic(self, obs_embedding) -> Dict:
        x = self.critic_head(obs_embedding)
        return {'value': x['pred']}

    def compute_actor_critic(self, obs_embedding, sample_action_type:str='argmax') -> Dict:
        value = self.critic_head(obs_embedding)['pred']

        assert sample_action_typein ["argmax", "normal"], "sample_action_type should be 'argmax' or 'normal'"
        action_type_logit = self.actor_action_type(obs_embedding)['logit']

        if sample_action_type == 'normal':
            action_type = torch.multinomial(action_type_logit, 1).item()
            action_args_logit = self.actor_action_args[action_type_names[action_type]](obs_embedding, action_type_logit)
            action_args = torch.multinomial(action_args_logit, 1).item()
        elif sample_action_type == 'argmax':
            action_type = torch.argmax(action_type_logit).item()
            action_args_logit = self.actor_action_args[action_type_names[action_type]](obs_embedding, action_type_logit)
            action_args = torch.argmax(action_args_logit, 1).item()

        return {'logit': {'action_type': action_type, 'action_args': action_args}, 'value': value}

