from re import S
import struct
import socket
from matplotlib.animation import FuncAnimation

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 1234))
s.listen(5)

data_protocol = '6f'
data_size = struct.calcsize(data_protocol)
print("waiting for connection...")
client, address = s.accept()
print("new client")

while True:
    data = client.recv(data_size)
    p1x, p1y, p1z, p2x, p2y, p2z = struct.unpack(data_protocol, data)
    print(p1x, ' .. ', p1y, ' .. ', p1z)
    print(p2x, ' .. ', p2y, ' .. ', p2z)
