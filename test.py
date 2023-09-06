import time
import xpc

playtime = 60

with xpc.XPlaneConnect() as client:
    client.sendCTRL([0,0,0,0,0,0.0])
    Lat, Lon, Alt, Pitch, Roll, Yaw, Gear = client.getPOSI(0)
    print(f"""
    Lat: {Lat}
    Lon: {Lon}
    Alt: {Alt}
    Pitch: {Pitch}
    Yaw: {Yaw}
    Gear: {Gear}
    """)
    client.sendPOSI([52.6, -1.1, 2500, 0,    0,   0,  1], 0)
    Lat, Lon, Alt, Pitch, Roll, Yaw, Gear = client.getPOSI(0)
    print(f"""
    Lat: {Lat}
    Lon: {Lon}
    Alt: {Alt}
    Pitch: {Pitch}
    Yaw: {Yaw}
    Gear: {Gear}
    """)
