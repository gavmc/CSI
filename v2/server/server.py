import socket
import struct

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 5566))
while True:
    data, addr = s.recvfrom(1024)

    hdr = struct.unpack('<HBBIqbBH', data[:20])
    magic, sensor_id, channel, seq, esp_time_us, rssi, flags, length = hdr
    payload = data[20:20 + length]
    print(len(payload), payload[:10])
