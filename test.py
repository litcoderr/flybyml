from connect import XP

xp = XP(verbose=True)

num_engines = xp.get_num_engines()
print(int(num_engines))

while True:
    # TODO test getting dref
    pass

"""
beacon = find_xp()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# 1) Subscribe to receive once per second
cmd = b'RREF'  # "Request DataRef(s)"
freq = 1       # number of times per second (integer)
index = 0      # "my" number, so I can match responsed with my request
msg = struct.pack("<4sxii400s", cmd, freq, index, b'sim/aircraft/engine/acf_num_engines')
sock.sendto(msg, (beacon['ip'], beacon['port']))

# 2) Block, waiting to receive a packet
data, addr = sock.recvfrom(2048)
header = data[0:4]
if header[0:4] != b'RREF':
    raise ValueError("Unknown packet")

# 3) Unpack the data:
idx, value = struct.unpack("<if", data[5:13])
assert idx == index
print("Number of engines is {}".format(int(value)))

# 4) Unsubscribe -- as otherwise we'll continue to get this data, once every second!
freq = 0
msg = struct.pack("<4sxii400s", cmd, freq, index, b'sim/aircraft/engine/acf_num_engines')
sock.sendto(msg, (beacon['ip'], beacon['port']))
sock.close()
"""