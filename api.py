import binascii
import platform
import socket
import struct
import xpc

from controls import Controls
from state.att import Attitude
from state.pos import Position
from aircraft import Aircraft
from weather import Weather


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


def get_dref(dref):
    with xpc.XPlaneConnect() as client:
        return client.getDREF(dref)

def set_dref(dref, value):
    with xpc.XPlaneConnect() as client:
        client.sendDREF(dref, value)


class API(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.beacon = find_xp()
        print("Beacon initialized...")

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print("Socket initialized...")
    
    def __del__(self):
        self.socket.close()
    
    def get_posi_att(self):
        # Implemented using xplane connect for faster response
        with xpc.XPlaneConnect() as client:
            lat, lon, alt, pitch, roll, yaw, _ = client.getPOSI(0)

            pos = Position(lat, lon, alt)
            att = Attitude(pitch, roll, yaw)
        
        return pos, att

    def set_init_state(self, ac: Aircraft,
                lat: float,
                lon: float,
                alt: float,
                heading: float,
                speed: float):
        """
        lat: degree
        lon: degree
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
                   ''.encode('utf-8'),      # remember to encode string to bytes
                   0,                               # it's an index, not the runway heading
                   0,                               # again, an index
                   lat, lon,        # Not needed, if you use apt_id
                   alt,      # elevation meters
                   heading,                # aircraft heading true
                   speed)                  # speed meters per second
        self.socket.sendto(msg, (self.beacon['ip'], self.beacon['port']))
    
    def send_ctrl(self, controls: Controls):
        with xpc.XPlaneConnect() as client:
            # set controls except for elevator trim
            client.sendCTRL(controls.to_api_compatible())
            self.set_elev_trim(controls.trim)
    
    def get_ctrl(self) -> Controls:
        with xpc.XPlaneConnect() as client:
            elev, ail, rud, thr, gear, flaps, _ = client.getCTRL(0)
            trim = self.get_elev_trim()
            return Controls(elev, ail, rud, thr, gear, flaps, trim)
    
    def get_elev_trim(self) -> float:
        return get_dref("sim/cockpit2/controls/elevator_trim")[0]

    def set_elev_trim(self, value):
        set_dref("sim/cockpit2/controls/elevator_trim", value)
    
    def get_indicated_airspeed(self):
        """
        returns in m/s
        """
        return get_dref('sim/flightmodel/position/indicated_airspeed')[0] * 1.94384
    
    def get_vertical_speed(self):
        """
        returns in m/s
        """
        fpm = get_dref('sim/cockpit2/tcas/targets/position/vertical_speed')[0]
        mps = fpm * (0.3048 / 60)
        return mps
    
    def set_zulu_time(self, zulu_time: float):
        # zulu_time: GMT time. seconds since midnight
        set_dref('sim/time/zulu_time_sec', zulu_time)
    
    def set_weather(self, weather: Weather):
        # set dref of all Weather properties
        for prop in weather.__dict__.values():
            set_dref(prop.dref, prop.value)
        set_dref('sim/weather/region/update_immediately', True)
    
    def pause(self):
        with xpc.XPlaneConnect() as client:
            client.pauseSim(True)
    
    def resume(self):
        with xpc.XPlaneConnect() as client:
            client.pauseSim(False)
