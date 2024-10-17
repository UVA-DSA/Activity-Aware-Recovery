import json
import time
from enum import Enum
import socket
import statistics

class NetworkState(Enum):
    NORMAL = 1
    PACKET_LOSS = 2
    COMMUNICATION_LOSS = 3

class NetworkInterface:
    def __init__(self, udp_port, output_port):
        self.state = NetworkState.NORMAL
        self.last_sequence_id = 0
        self.last_packet_time = time.time()
        self.arrival_times = []
        self.median_arrival_time = float('inf')
        
        # Set up UDP socket for listening
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(('0.0.0.0', udp_port))
        
        # Set up UDP socket for output
        self.output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.output_port = output_port

    def listen_and_process(self):
        while True:
            data, addr = self.udp_socket.recvfrom(1024)
            self.process_packet(data)

    def process_packet(self, data):
        try:
            packet = json.loads(data)
            sequence_id = packet['sequence_id']
            
            if sequence_id <= self.last_sequence_id:
                return  # Discard old packets
            
            current_time = time.time()
            time_since_last_packet = current_time - self.last_packet_time
            
            self.update_arrival_stats(time_since_last_packet)
            
            if self.state == NetworkState.NORMAL:
                if time_since_last_packet <= 1.5 * self.median_arrival_time:
                    self.relay_packet(packet)
                else:
                    self.set_state(NetworkState.PACKET_LOSS)
                    self.send_special_packet("PACKET_LOSS")
            elif self.state == NetworkState.PACKET_LOSS:
                if time_since_last_packet < 2:
                    self.set_state(NetworkState.NORMAL)
                    self.relay_packet(packet)

                else:
                    self.set_state(NetworkState.COMMUNICATION_LOSS)
                    self.send_special_packet("COMMUNICATION_LOSS")
            elif self.state == NetworkState.COMMUNICATION_LOSS:
                self.set_state(NetworkState.NORMAL)
                self.relay_packet(packet)
            
            self.last_sequence_id = sequence_id
            self.last_packet_time = current_time
        
        except json.JSONDecodeError:
            print("Invalid JSON packet received")

    def update_arrival_stats(self, time_since_last_packet):
        self.arrival_times.append(time_since_last_packet)
        if len(self.arrival_times) > 100:  # Keep only the last 100 arrival times
            self.arrival_times.pop(0)
        self.median_arrival_time = statistics.median(self.arrival_times)

    def set_state(self, new_state):
        self.state = new_state
        print(f"State changed to: {new_state}")

    def relay_packet(self, packet):
        self.output_socket.sendto(json.dumps(packet).encode(), ('0.0.0.0', self.output_port))

    def send_special_packet(self, packet_type):
        special_packet = {"type": packet_type, "timestamp": time.time()}
        self.output_socket.sendto(json.dumps(special_packet).encode(), ('0.0.0.0', self.output_port))

