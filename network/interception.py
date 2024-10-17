import random
from enum import Enum
import scipy.stats as stats
import socket
import threading
import time
import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTabWidget, QLineEdit, QCheckBox
from PyQt5.QtCore import Qt

class NetworkState(Enum):
    GOOD = 0
    BAD = 1
    INTERMEDIATE1 = 2
    INTERMEDIATE2 = 3

class StochasticPacketLoss:
    def __init__(self, config):
        self.state = NetworkState.GOOD
        self.transition_matrix = config['transition_matrix']
        self.loss_rates = config['loss_rates']
        self.pareto_alpha = config['pareto_alpha']
        self.pareto_lambda = config['pareto_lambda']
        self.current_burst_length = 0

    def update_state(self):
        self.state = random.choices(list(NetworkState), weights=self.transition_matrix[self.state.value])[0]

    def should_drop_packet(self):
        if self.current_burst_length > 0:
            self.current_burst_length -= 1
            return True

        if random.random() < self.loss_rates[self.state.value]:
            self.current_burst_length = int(stats.pareto.rvs(self.pareto_alpha, scale=self.pareto_lambda))
            return True

        return False

class Jitter:
    def __init__(self, config):
        self.min_jitter = config['min_jitter']
        self.max_jitter = config['max_jitter']

    def get_jitter(self):
        return random.uniform(self.min_jitter, self.max_jitter)

class Delay:
    def __init__(self, config):
        self.min_delay = config['min_delay']
        self.max_delay = config['max_delay']

    def get_delay(self):
        return random.uniform(self.min_delay, self.max_delay)

class CommunicationLoss:
    def __init__(self, config):
        self.loss_probability = config['communication_loss_probability']
        self.min_duration = config['min_loss_duration']
        self.max_duration = config['max_loss_duration']
        self.is_loss_active = False
        self.loss_end_time = 0

    def update(self):
        current_time = time.time()
        if self.is_loss_active and current_time >= self.loss_end_time:
            self.is_loss_active = False

        if not self.is_loss_active and random.random() < self.loss_probability:
            self.is_loss_active = True
            duration = random.uniform(self.min_duration, self.max_duration)
            self.loss_end_time = current_time + duration

    def is_communication_lost(self):
        return self.is_loss_active

class Interception:
    def __init__(self, config):
        self.packet_loss_model = StochasticPacketLoss(config)
        self.jitter_model = Jitter(config)
        self.delay_model = Delay(config)
        self.communication_loss_model = CommunicationLoss(config)
        self.input_port = config['input_port']
        self.output_port = config['output_port']
        self.input_socket = None
        self.output_socket = None
        self.running = False
        self.packet_loss_enabled = False
        self.delay_enabled = False
        self.jitter_enabled = False
        self.communication_loss_enabled = False
        self.simulation_enabled = False

    def start(self):
        self.input_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.input_socket.bind(('0.0.0.0', self.input_port))
        
        self.output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.running = True
        
        # Start listening for packets in a separate thread
        threading.Thread(target=self._listen_for_packets, daemon=True).start()
        
        print(f"Interception started. Listening on port {self.input_port}")

    def _listen_for_packets(self):
        while self.running:
            try:
                data, addr = self.input_socket.recvfrom(4096)  # Buffer size of 4096 bytes
                self.process_packet(data)
            except Exception as e:
                print(f"Error receiving packet: {e}")

    def process_packet(self, packet):
        if not self.simulation_enabled:
            self.relay_packet(packet)
            return

        self.packet_loss_model.update_state()
        self.communication_loss_model.update()

        if self.communication_loss_enabled and self.communication_loss_model.is_communication_lost():
            return  # Drop the packet due to communication loss

        if self.packet_loss_enabled and self.packet_loss_model.should_drop_packet():
            return  # Drop the packet due to packet loss

        jitter = self.jitter_model.get_jitter() if self.jitter_enabled else 0
        delay = self.delay_model.get_delay() if self.delay_enabled else 0
        total_delay = jitter + delay
        
        if total_delay > 0:
            threading.Timer(total_delay, self.relay_packet, args=[packet]).start()
        else:
            self.relay_packet(packet)

    def relay_packet(self, packet):
        try:
            self.output_socket.sendto(packet, ('127.0.0.1', self.output_port))
        except Exception as e:
            print(f"Error relaying packet: {e}")

    def stop(self):
        self.running = False
        if self.input_socket:
            self.input_socket.close()
        if self.output_socket:
            self.output_socket.close()
        print("Interception stopped")

class InterceptionGUI(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.interception = Interception(config)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Interception Control')
        self.setGeometry(100, 100, 400, 300)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # Control tab
        control_tab = QWidget()
        control_layout = QVBoxLayout()
        control_tab.setLayout(control_layout)

        # Main simulation switch
        self.simulation_switch = QCheckBox('Enable Network Error Simulation')
        self.simulation_switch.stateChanged.connect(self.toggle_simulation)
        control_layout.addWidget(self.simulation_switch)

        # Error type switches
        self.packet_loss_switch = QCheckBox('Enable Packet Loss')
        self.delay_switch = QCheckBox('Enable Delay')
        self.jitter_switch = QCheckBox('Enable Jitter')
        self.comm_loss_switch = QCheckBox('Enable Communication Loss')

        for switch in [self.packet_loss_switch, self.delay_switch, self.jitter_switch, self.comm_loss_switch]:
            switch.stateChanged.connect(self.update_error_types)
            control_layout.addWidget(switch)

        # Start/Stop button
        self.start_stop_button = QPushButton('Start Interception')
        self.start_stop_button.clicked.connect(self.toggle_interception)
        control_layout.addWidget(self.start_stop_button)

        tabs.addTab(control_tab, 'Control')

        # Config tab
        config_tab = QWidget()
        config_layout = QVBoxLayout()
        config_tab.setLayout(config_layout)

        self.config_inputs = {}
        for key, value in self.config.items():
            if isinstance(value, (int, float, str)):
                layout = QHBoxLayout()
                layout.addWidget(QLabel(key))
                input_field = QLineEdit(str(value))
                layout.addWidget(input_field)
                self.config_inputs[key] = input_field
                config_layout.addLayout(layout)

        save_button = QPushButton('Save Configuration')
        save_button.clicked.connect(self.save_config)
        config_layout.addWidget(save_button)

        tabs.addTab(config_tab, 'Configuration')

    def toggle_simulation(self, state):
        self.interception.simulation_enabled = state == Qt.Checked

    def update_error_types(self):
        self.interception.packet_loss_enabled = self.packet_loss_switch.isChecked()
        self.interception.delay_enabled = self.delay_switch.isChecked()
        self.interception.jitter_enabled = self.jitter_switch.isChecked()
        self.interception.communication_loss_enabled = self.comm_loss_switch.isChecked()

    def toggle_interception(self):
        if self.interception.running:
            self.interception.stop()
            self.start_stop_button.setText('Start Interception')
        else:
            self.interception.start()
            self.start_stop_button.setText('Stop Interception')

    def save_config(self):
        for key, input_field in self.config_inputs.items():
            try:
                value = type(self.config[key])(input_field.text())
                self.config[key] = value
            except ValueError:
                print(f"Invalid value for {key}")

        with open('config.json', 'w') as f:
            json.dump(self.config, f, indent=2)

        self.interception = Interception(self.config)

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    app = QApplication(sys.argv)
    random.seed(config['simulation_seed'])
    gui = InterceptionGUI(config)
    gui.show()
    sys.exit(app.exec_())
