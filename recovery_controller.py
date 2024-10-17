from enum import Enum
from typing import Optional
from network.interface import NetworkInterface
from trajectory import TransformerPredictor, DMPPredictor

class RecoveryState(Enum):
    TELEOPERATION = 1
    SHORT_TERM_RECOVERY = 2
    LONG_TERM_RECOVERY = 3
    E_STOP = 4


class RecoveryController:
    def __init__(self, transformer_predictor: TransformerPredictor, dmp_predictor: DMPPredictor):
        self.transformer_predictor = transformer_predictor
        self.dmp_predictor = dmp_predictor
        self.network_interface = NetworkInterface()
        self.current_state: RecoveryState = RecoveryState.TELEOPERATION

    def set_state(self, new_state: RecoveryState):
        self.current_state = new_state

    def get_state(self) -> RecoveryState:
        return self.current_state

    # Add more methods as needed for your specific use case

