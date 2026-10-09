import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bank_routing.calibrate import calibrate

if __name__ == "__main__":
    calibrate()
