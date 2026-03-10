"""
Model Registry
==============
Unified interface for all models used in the LNN vs RNN comparison study.

Supported models:
  - LTC  : Liquid Time-Constant Network  (via ncps)
  - CfC  : Closed-form Continuous-time Network (via ncps)
  - LSTM : Long Short-Term Memory        (PyTorch native)
  - GRU  : Gated Recurrent Unit          (PyTorch native)
"""

import torch
import torch.nn as nn
from ncps.torch import LTC, CfC
from ncps.wirings import AutoNCP, FullyConnected


# ---------------------------------------------------------------------------
# Thin wrappers so every model exposes the same forward signature:
#   output, hidden = model(x, hidden)
#   x      : (batch, seq_len, input_size)
#   output : (batch, seq_len, output_size)
# ---------------------------------------------------------------------------

class LTCModel(nn.Module):
    """LTC wrapped with a linear readout layer."""

    def __init__(self, input_size: int, units: int, output_size: int):
        super().__init__()
        wiring = FullyConnected(units, output_size)
        self.rnn = LTC(input_size, wiring, batch_first=True)

    def forward(self, x, hx=None):
        out, hx = self.rnn(x, hx)
        return out, hx

    def init_hidden(self, batch_size: int, device: torch.device):
        return None


class CfCModel(nn.Module):
    """CfC with a linear readout layer.

    CfC with FullyConnected wiring returns the full hidden state; we add an
    explicit linear projection to produce the desired output_size.
    """

    def __init__(self, input_size: int, units: int, output_size: int):
        super().__init__()
        self.rnn = CfC(input_size, units, batch_first=True)
        self.fc  = nn.Linear(units, output_size)

    def forward(self, x, hx=None):
        out, hx = self.rnn(x, hx)   # out: (batch, seq, units)
        out = self.fc(out)            # out: (batch, seq, output_size)
        return out, hx

    def init_hidden(self, batch_size: int, device: torch.device):
        return None


class LSTMModel(nn.Module):
    """Standard LSTM with a linear readout head."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hx=None):
        out, hx = self.lstm(x, hx)
        out = self.fc(out)
        return out, hx

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                        device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                        device=device)
        return (h, c)


class GRUModel(nn.Module):
    """Standard GRU with a linear readout head."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hx=None):
        out, hx = self.gru(x, hx)
        out = self.fc(out)
        return out, hx

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size,
                           device=device)


class RNNModel(nn.Module):
    """Vanilla RNN with a linear readout head."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hx=None):
        out, hx = self.rnn(x, hx)
        out = self.fc(out)
        return out, hx

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size,
                           device=device)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY = {
    "ltc":  LTCModel,
    "cfc":  CfCModel,
    "lstm": LSTMModel,
    "gru":  GRUModel,
    "rnn":  RNNModel,
}


def build_model(name: str, input_size: int, units: int,
                output_size: int) -> nn.Module:
    """
    Instantiate a model by name from the registry.

    Parameters
    ----------
    name        : one of 'ltc', 'cfc', 'lstm', 'gru', 'rnn'
    input_size  : number of input features
    units       : number of hidden units / RNN cells
    output_size : number of output features
    """
    name = name.lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_REGISTRY)}"
        )
    return _REGISTRY[name](input_size, units, output_size)


def list_models():
    """Return the list of registered model names."""
    return list(_REGISTRY.keys())
