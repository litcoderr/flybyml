from xplane_airports.AptDat import AptDat
from pathlib import Path

APT_DAT_PATH = Path("C:\\Program Files (x86)\\Steam\\steamapps\\common\\X-Plane 12\\Global Scenery\\Global Airports\\Earth nav data\\apt.dat")
# read all apt data
print("reading airports meta data ...")
apt = AptDat(APT_DAT_PATH)
print("finished reading airports meta data")

# TODO filter landable runway

# TODO impelement sampling method
