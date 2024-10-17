import time
from enum import Enum
from typing import Optional
from network.interface import NetworkInterface
from trajectory import TransformerPredictor, DMPPredictor
from robot_interface import RobotInterface
from perception import Perception

class RecoveryState(Enum):
    TELEOPERATION = 1
    SHORT_TERM_RECOVERY = 2
    LONG_TERM_RECOVERY = 3
    E_STOP = 4


class RecoveryController:
    def __init__(self, transformer_predictor: TransformerPredictor, dmp_predictor: DMPPredictor):
        self.transformer_predictor = transformer_predictor
        self.dmp_predictor = dmp_predictor
        self.network_interface = NetworkInterface(input_port=5005, output_port=5006)
        self.robot_interface = RobotInterface()
        self.perception = Perception()
        self.current_state: RecoveryState = RecoveryState.TELEOPERATION

    def set_state(self, new_state: RecoveryState):
        self.current_state = new_state

    def get_state(self) -> RecoveryState:
        return self.current_state

    def run(self, frequency: float = 1000):
        period = 1.0 / frequency
        while True:
            start_time = time.time()

            packet = self.network_interface.get_packet()
            robot_command = None

            if packet['type'] == 'NORMAL':
                robot_command = packet
            elif packet['type'] == 'PACKET_LOSS':
                robot_state = self.robot_interface.get_state(w_obs=30)
                robot_command = self.transformer_predictor(robot_state)
            elif packet['type'] == 'COMMUNICATION_LOSS':
                current_mp, mp_goals = self.perception.perception_results()
                robot_command = self.dmp_predictor(current_mp, mp_goals)

            if robot_command:
                self.robot_interface.set_command(robot_command)

            # Sleep for the remaining time to maintain the desired frequency
            elapsed_time = time.time() - start_time
            sleep_time = max(0, period - elapsed_time)
            time.sleep(sleep_time)

if __name__ == "__main__":
    transformer_predictor = TransformerPredictor()
    dmp_predictor = DMPPredictor()
    recovery_controller = RecoveryController(transformer_predictor, dmp_predictor)
    recovery_controller.run(frequency=1000)

