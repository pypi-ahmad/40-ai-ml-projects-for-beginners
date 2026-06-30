
from .lstm import LSTM_LM


class StackedLSTM_LM(LSTM_LM):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
