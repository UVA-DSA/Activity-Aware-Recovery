import json
import time
from enum import Enum
import socket
import statistics
import threading

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
        
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind(('localhost', udp_port))
        self.udp_socket.setblocking(False)
        
        self.output_port = output_port
        self.packet_queue = []
        
        self.lock = threading.Lock()
        
        # Start background threads
        self.listen_thread = threading.Thread(target=self.listen_for_packets, daemon=True)
        self.listen_thread.start()
        self.state_update_thread = threading.Thread(target=self.update_state, daemon=True)
        self.state_update_thread.start()

    def listen_for_packets(self):
        while True:
            try:
                data, _ = self.udp_socket.recvfrom(1024)
                with self.lock:
                    self.process_packet(data)
            except BlockingIOError:
                time.sleep(0.01)

    def process_packet(self, data):
        try:
            packet = json.loads(data)
            sequence_id = packet['sequence_id']
            
            if sequence_id <= self.last_sequence_id:
                return  # Discard old packets
            
            self.update_arrival_stats()
            self.packet_queue.append(packet)
            self.last_sequence_id = sequence_id
            self.last_packet_time = time.time()
        
        except json.JSONDecodeError:
            print("Invalid JSON packet received")

    def update_arrival_stats(self):
        current_time = time.time()
        time_since_last_packet = current_time - self.last_packet_time
        self.arrival_times.append(time_since_last_packet)
        if len(self.arrival_times) > 100:
            self.arrival_times.pop(0)
        self.median_arrival_time = statistics.median(self.arrival_times)

    def update_state(self):
        while True:
            with self.lock:
                current_time = time.time()
                time_since_last_packet = current_time - self.last_packet_time

                if self.state == NetworkState.NORMAL:
                    if time_since_last_packet > 1.5 * self.median_arrival_time:
                        self.state = NetworkState.PACKET_LOSS
                elif self.state == NetworkState.PACKET_LOSS:
                    if time_since_last_packet >= 2:
                        self.state = NetworkState.COMMUNICATION_LOSS
                elif self.state == NetworkState.COMMUNICATION_LOSS:
                    if self.packet_queue:
                        self.state = NetworkState.NORMAL
            
            time.sleep(0.1)  # Check state every 100ms

    def get_packet(self):
        with self.lock:
            if self.packet_queue:
                return self.packet_queue.pop(0)
            else:
                return self.create_special_packet(self.state.name)

    def create_special_packet(self, packet_type):
        return {"type": packet_type, "timestamp": time.time()}

    def relay_packet(self, packet):
        self.output_socket.sendto(json.dumps(packet).encode(), ('0.0.0.0', self.output_port))

    def send_special_packet(self, packet_type):
        special_packet = {"type": packet_type, "timestamp": time.time()}
        self.output_socket.sendto(json.dumps(special_packet).encode(), ('0.0.0.0', self.output_port))
