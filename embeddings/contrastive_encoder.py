"""An MLP network for encoding context of RL tasks."""
import akro
import copy
import numpy as np
from torch import rand
from torch import nn
from torch import matmul
from torch.nn import functional as F

from garage import InOutSpec
from garage.np.embeddings import Encoder


class ContrastiveEncoder(nn.Module, Encoder):
    """
        This is a contrastive encoder

    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 common_network = True,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False):

        super().__init__()

        self._output_dim = output_dim
        self._input_dim = input_dim
        self._common_network = common_network
        prev_size = self._input_dim

        if self._common_network:
            # Initialize common hidden layer
            self._hidden_layers = nn.ModuleList()
            for size in hidden_sizes:
                hidden_layer = nn.Sequential()
                if layer_normalization:
                    hidden_layer.add_module('layer_normalization',
                                             nn.LayerNorm(prev_size))
                linear_layer = nn.Linear(prev_size, size)
                hidden_w_init(linear_layer.weight)
                hidden_b_init(linear_layer.bias)
                hidden_layer.add_module('linear', linear_layer)

                if hidden_nonlinearity:
                    hidden_layer.add_module('non_linearity',
                                             _NonLinearity(hidden_nonlinearity))

                self._hidden_layers.append(hidden_layer)
                prev_size = size
        else:
            hidden_sizes = hidden_sizes * 2
        # Initialize query and key network layers
        self._query_layers = QueryKeyModule(prev_size, self._output_dim,
                                            hidden_sizes, hidden_nonlinearity,
                                            hidden_w_init, hidden_b_init,
                                            output_nonlinearity, output_w_init,
                                            output_b_init, layer_normalization)
        self._key_layers = copy.deepcopy(self._query_layers)




    def forward(self, input, query=True):
        if self._common_network:
            for layer in self._hidden_layers:
                input = layer(input)

        if query:
            output = self._query_layers(input)
        else:
            output = self._key_layers(input)
        return output

    def get_query_net(self):
        return self._query_layers

    def get_key_net(self):
        return self._key_layers

    @property
    def networks(self):
        if self._common_network:
            return [self._hidden_layers, self._query_layers, self._key_layers]
        else:
            return [self._query_layers, self._key_layers]


    @property
    def spec(self):
        """garage.InOutSpec: Input and output space."""
        input_space = akro.Box(-np.inf, np.inf, self._input_dim)
        output_space = akro.Box(-np.inf, np.inf, self._output_dim)
        return InOutSpec(input_space, output_space)

    @property
    def input_dim(self):
        """int: Dimension of the encoder input."""
        return self._input_dim

    @property
    def output_dim(self):
        """int: Dimension of the encoder output (embedding)."""
        return self._output_dim

    def reset(self, do_resets=None):
        """Reset the encoder.

        This is effective only to recurrent encoder. do_resets is effective
        only to vectoried encoder.

        For a vectorized encoder, do_resets is an array of boolean indicating
        which internal states to be reset. The length of do_resets should be
        equal to the length of inputs.

        Args:
            do_resets (numpy.ndarray): Bool array indicating which states
                to be reset.

        """
        pass


class QueryKeyModule(nn.Module):
    def __init__(self, input_dim, output_dim, query_sizes,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()
        # Initialize query and key network layers
        self._hidden_layers = nn.ModuleList()
        prev_size = input_dim
        for size in query_sizes:
            hidden_layer = nn.Sequential()
            if layer_normalization:
                hidden_layer.add_module('layer_normalization',
                                         nn.LayerNorm(prev_size))
            linear_layer = nn.Linear(prev_size, size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            hidden_layer.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                hidden_layer.add_module(
                    'non_linearity', _NonLinearity(hidden_nonlinearity))
            self._hidden_layers.append(hidden_layer)
            prev_size = size

        self._output_layers = nn.ModuleList()
        output_layer = nn.Sequential()
        linear_layer = nn.Linear(prev_size, output_dim)
        output_w_init(linear_layer.weight)
        output_b_init(linear_layer.bias)
        output_layer.add_module('linear', linear_layer)

        if output_nonlinearity:
            output_layer.add_module(
                'non_linearity', _NonLinearity(output_nonlinearity))
        self._output_layers.append(output_layer)

    # pylint: disable=arguments-differ
    def forward(self, input):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values

        Returns:
            torch.Tensor: Output value

        """
        hidden_output = input
        for layer in self._hidden_layers:
            hidden_output = layer(hidden_output)

        output = hidden_output
        for layer in self._output_layers:
            output = layer(output)
        return output

    # pylint: disable=missing-return-doc, missing-return-type-doc
    def __repr__(self):
        return repr(self.module)


class ContrastiveLoss(nn.Module):
    def __init__(self, query_key_dim):
        self._w = rand(query_key_dim, query_key_dim,requires_grad=True)

    def forward(self, input_query, input_key):
        import torch
        left_product = matmul(input_query, self._w)
        logits = matmul(left_product, input_key)
        logits = logits - torch.max(logits, axis=1)
        labels = torch.arange(logits.shape[0])
        loss = nn.CrossEntropyLoss(logits, labels)
        return loss



class _NonLinearity(nn.Module):
    """Wrapper class for non linear function or module.

    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.

    """

    def __init__(self, non_linear):
        super().__init__()

        if isinstance(non_linear, type):
            self.module = non_linear()
        elif callable(non_linear):
            self.module = copy.deepcopy(non_linear)
        else:
            raise ValueError(
                'Non linear function {} is not supported'.format(non_linear))

    # pylint: disable=arguments-differ
    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values

        Returns:
            torch.Tensor: Output value

        """
        return self.module(input_value)

    # pylint: disable=missing-return-doc, missing-return-type-doc
    def __repr__(self):
        return repr(self.module)
