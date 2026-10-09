import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bank_routing.baselines import train_zero_shot

if __name__ == "__main__":
    train_zero_shot()
