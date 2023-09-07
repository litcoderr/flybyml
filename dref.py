import xpc

def getDref(dref):
    with xpc.XPlaneConnect() as client:
        return client.getDREF(dref)

def sendDref(dref, data):
    with xpc.XPlaneConnect() as client:
        client.sendDREF(dref, data)

def hasCrashed() -> bool:
    return getDref("sim/flightmodel2/misc/has_crashed")[0] == 1

def setIndicatedAirspeed(speed: float):
    # TODO Does not work. Needs to be fixed
    return sendDref("sim/flightmodel/position/indicated_airspeed", float(speed))

def getIndicatedAirspeed():
    return getDref("sim/flightmodel/position/indicated_airspeed")