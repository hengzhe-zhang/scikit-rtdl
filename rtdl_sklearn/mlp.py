import math
import typing as ty
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
            self,
            *,
            d_in: int,
            d_layers: ty.List[int],
            dropout: float,
            d_out: int,
            categories: ty.Optional[ty.List[int]],
            d_embedding: int,
            feature_index: Tuple,
    ) -> None:
        super().__init__()

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            # print(f'{self.category_embeddings.weight.shape=}')

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)
        self.feature_index = feature_index

    def forward(self, x_data):
        x_num, x_cat = x_data[:, self.feature_index[0]], x_data[:, self.feature_index[1]]
        x_cat = x_cat.long()
        if x_cat.size()[1] == 0:
            x_cat = None
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        # x = x.squeeze(-1)
        return x
