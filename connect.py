from typing import Optional
import binascii
import platform
import socket
import struct
import time
import threading


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
        self.max_idx = 0
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
        self.map.pop(idx, None)


class XP(object):
    def __init__(self):
        self.beacon = find_xp()
        print("Beacon initialized...")

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.beacon['ip'], self.beacon['port']))
        print("Socket initialized...")

        self.map = DataMap()
        self.lock = threading.Lock()

        # start a thread that requests dref
        self.dref_requester = threading.Thread(target=self.request_dref, daemon=True)
        self.dref_requester.start()

        # start a thread that listens to incomming dref updated packets
        self.dref_listener = threading.Thread(target=self.listen_dref, daemon=True)
        self.dref_listener.start()
    
    def __del__(self):
        self.socket.close()

    def get_dref(self, dref: str, isAsync = False) -> int:
        """
        Sends a request to get dref.
        Returns request id.

        If the isAsync argument is True,
        updated dref should be monitored through self.map
        """
        self.lock.acquire()

        idx = self.map.assign_idx()
        sub_msg = struct.pack("<4sxii400s", b'RREF', 1, idx, bytes(dref, 'utf-8'))
        unsub_msg = struct.pack("<4sxii400s", b'RREF', 0, idx, bytes(dref, 'utf-8'))

        self.map.allocate(idx, sub_msg, unsub_msg)  # subscribe to idx with unsub_msg to be called

        self.lock.release()

        if not isAsync:
            # TODO wait for self.map to be populated with retrieved data
            pass
        return idx
    
    def request_dref(self):
        timeout = 2

        self.lock.acquire()
        print("Dataref requester started...")
        print(f"initial timeout: {timeout}]")
        self.lock.release()

        while True:
            self.lock.acquire()
            
            for idx, cont in self.map.get_iterator():
                current_time = time.time()
                if cont.data == None and current_time - cont.last_called > timeout:
                    # call request
                    cont.last_called = current_time
                    self.socket.sendto(cont.sub_msg, (self.beacon['ip'], self.beacon['port']))
            
            self.lock.release()
    
    def listen_dref(self):
        garbage_duration = 10

        self.lock.acquire()
        print("Dataref listener started...")
        print(f"garbage duration: {garbage_duration}]")
        self.lock.release()

        while True:
            packet, _ = self.socket.recvfrom(2048)

            self.lock.acquire()

            values = packet[5:]
            n_values = int(len(values) / 8)

            current_time = time.time()
            for i in range(n_values):
                packed = packet[(5 + 8*i): (5 + 8 * (i+1))]
                (idx, data) = struct.unpack("<if", packed)
                self.map.set(idx, data, current_time)

                self.socket.sendto(self.map.get(idx).unsub_msg, (self.beacon['ip'], self.beacon['port']))
            
            # collect garbage
            garbage = []
            for idx, cont in self.map.get_iterator():
                if cont.last_recieved != None and current_time - cont.last_recieved > garbage_duration:
                    garbage.append(idx)

            # delete garbage
            for garbage_idx in garbage:
                self.map.pop(garbage_idx)
            
            self.lock.release()
    