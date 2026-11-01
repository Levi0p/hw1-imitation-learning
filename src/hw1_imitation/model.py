"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []

        prev = state_dim

        for val in hidden_dims :
            layers.append(torch.nn.Linear(prev,val))
            layers.append(torch.nn.ReLU())
            prev = val

        layers.append(torch.nn.Linear(prev, chunk_size * action_dim))

        self.mlp = torch.nn.Sequential(*layers)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        batch = action_chunk.shape[0]

        loss = []

        what_we_did = self.sample_actions(state)

        for i in range(batch) :
            temp = (action_chunk[i] - what_we_did[i]) ** 2
            temp = torch.flatten(temp)
            loss.append(torch.sum(temp))

        loss = torch.stack(loss)

        loss = torch.mean(loss)

        return loss


    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        x = self.mlp(state)
        x = x.view(-1,self.chunk_size,self.action_dim)

        return x


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        layers = []

        prev = 1 + chunk_size * action_dim + state_dim

        for val in hidden_dims :
            layers.append(torch.nn.Linear(prev, val))
            layers.append(torch.nn.ReLU())
            prev = val

        layers.append(torch.nn.Linear(prev, chunk_size * action_dim))

        self.mlp = torch.nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        loss = []

        batch, chunk, dim = action_chunk.shape

        noise = torch.randn_like(action_chunk)
        tau = torch.rand(batch,device=action_chunk.device)

        for i in range(batch) :
            temp = tau[i] * action_chunk[i] + (1 - tau[i]) * noise[i]
            pred_velocity = self.mlp(torch.cat([state[i], temp.view(-1), tau[i].view(1)]))
            pred_velocity = pred_velocity.view(chunk,dim)
            # pred_velocity = self.mlp(state,temp,tau)
            temp1 = (pred_velocity - action_chunk[i] + noise[i]) ** 2
            temp1 = torch.flatten(temp1)
            loss.append(torch.sum(temp1))

        loss = torch.stack(loss)

        loss = torch.mean(loss)

        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch = state.shape[0]
        
        prev = torch.randn(batch,self.chunk_size,self.action_dim,device=state.device)

        tau = torch.zeros(batch,1,device=state.device)

        for i in range(num_steps) :
            temp = self.mlp(torch.cat([state, prev.view(batch, -1), tau], dim=1))
            temp = temp.view(batch,self.chunk_size,self.action_dim)
            curr = prev + (1/num_steps) * temp
            prev = curr
            tau += (1/num_steps)

        return prev
        


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
