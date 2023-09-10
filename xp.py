from typing import Optional
import binascii
import platform
import socket
import struct
import time
import threading
import xpc

from status.pos import Position, AirportPosition
from status.att import Attitude
from aircraft import Aircraft


class XPlaneIpNotFound(Exception):
    args = "Could not find any running xplane instance in network."


def find_xp(wait=3.0):
    """
    Waits for X-Plane to startup, and returns IP (and other) information
    about the first running X-Plane found.

    wait: floating point, maximum seconds to wait for beacon.
    """

    MCAST_GRP = '239.255.1.1'  # Standard multicast group
    MCAST_PORT = 49707  # (MCAST_PORT was 49000 for XPlane10)

    # Set up to listen for a multicast beacon
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if platform.system() == 'Windows':
        sock.bind(('', MCAST_PORT))
    else:
        sock.bind((MCAST_GRP, MCAST_PORT))
    mreq = struct.pack("=4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    if wait > 0:
        sock.settimeout(wait)

    beacon_data = {}
    while not beacon_data:
        try:
            packet, sender = sock.recvfrom(15000)
            header = packet[0:5]
            if header != b"BECN\x00":
                # We assume X-Plane is the only multicaster on this port
                print("Unknown packet from " + sender[0])
                print(str(len(packet)) + " bytes")
                print(packet)
                print(binascii.hexlify(packet))

            else:
                # header matches, so looks like the X-Plane beacon
                # * Data
                data = packet[5:21]

                # X-Plane documentation says:
                # struct becn_struct
                # {
                #    uchar beacon_major_version;    // 1 at the time of X-Plane 10.40, 11.55
                #    uchar beacon_minor_version;    // 1 at the time of X-Plane 10.40, 2 for 11.55
                #    xint application_host_id;      // 1 for X-Plane, 2 for PlaneMaker
                #    xint version_number;           // 104014 is X-Plane 10.40b14, 115501 is 11.55r2
                #    uint role;                     // 1 for master, 2 for extern visual, 3 for IOS
                #    ushort port;                   // port number X-Plane is listening on
                #    xchr    computer_name[500];    // the hostname of the computer
                #    ushort  raknet_port;           // port number the X-Plane Raknet clinet is listening on
                # };

                (beacon_major_version, beacon_minor_version, application_host_id,
                xplane_version_number, role, port) = struct.unpack("<BBiiIH", data)

                computer_name = packet[21:]  # Python3, these are bytes, not a string
                computer_name = computer_name.split(b'\x00')[0]  # get name upto, but excluding first null byte
                (raknet_port, ) = struct.unpack('<H', packet[-2:])

                if all([beacon_major_version == 1,
                        beacon_minor_version == 2,
                        application_host_id == 1]):
                    beacon_data = {
                        'ip': sender[0],
                        'port': port,
                        'hostname': computer_name.decode('utf-8'),
                        'xplane_version': xplane_version_number,
                        'role': role,
                        'raknet_port': raknet_port
                    }

        except socket.timeout:
            raise XPlaneIpNotFound()

    sock.close()
    return beacon_data

class Data:
    def __init__(self, idx, sub_msg, unsub_msg, data=None, last_called=0, last_recieved=None):
        self.idx = idx
        self.data = data
        self.sub_msg = sub_msg
        self.unsub_msg = unsub_msg
        self.last_called = last_called
        self.last_recieved = last_recieved

class DataMap:
    def __init__(self):
        self.max_idx = 1
        self.map = {}
    
    def assign_idx(self) -> int:
        self.max_idx += 1
        return self.max_idx-1
    
    def get_iterator(self):
        return self.map.items()
    
    def allocate(self, idx, sub_msg, unsub_msg) -> int:
        self.map[idx] = Data(idx, sub_msg, unsub_msg)
        return idx
    
    def get(self, idx):
        if idx in self.map.keys():
            return self.map[idx]
        else:
            return None
    
    def set(self, idx, data, last_recieved):
        self.map[idx].data = data
        self.map[idx].last_recieved = last_recieved
    
    def pop(self, idx):
        del self.map[idx]


class XP(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.beacon = find_xp()
        print("Beacon initialized...")

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print("Socket initialized...")
    
    def __del__(self):
        self.socket.close()
    
    def get_posi(self):
        # Implemented using xplane connect for faster response
        with xpc.XPlaneConnect() as client:
            lat, lon, alt, pitch, roll, yaw, gear = client.getPOSI(0)

            pos = Position(lat, lon, alt)
            att = Attitude(pitch, roll, yaw)
        
        return pos, att, gear
    
    def set_posi(self, ac: Aircraft, airport: AirportPosition,
                proposed_alt: float,
                proposed_heading: float,
                proposed_speed: float):
        """
        proposed_alt: m
        proposed_heading: true heading
        proposed_speed: m/s
        """
        type_start = 6  # used by maps and the like
        msg = struct.pack('<4sxi150s2xiii8siiddddd', b'ACPR',
                   0,                               # 0 -> User aircraft, otherwise 1-19
                   ac.f_path.encode('utf-8'),       # remember to encode string as bytes
                   0,                               # livery index for aircraft
                   type_start,                      # See enumeration with PREL
                   0,                               # 0 -> User aircraft, otherwise 1-19
                   airport.id.encode('utf-8'),      # remember to encode string to bytes
                   0,                               # it's an index, not the runway heading
                   0,                               # again, an index
                   airport.lat, airport.lon,        # Not needed, if you use apt_id
                   airport.alt + proposed_alt,      # elevation meters
                   proposed_heading,                # aircraft heading true
                   proposed_speed)                  # speed meters per second
        self.socket.sendto(msg, (self.beacon['ip'], self.beacon['port']))
    
    def get_indicated_airspeed(self):
        return self.get_dref('sim/flightmodel/position/indicated_airspeed')
    
    def get_dref(self, dref):
        with xpc.XPlaneConnect() as client:
            return client.getDREF(dref)
    