import numpy as np
from engine.pilot import EnzymeKineticsPilot

def main():
    rng = np.random.default_rng(42)
    pilot = EnzymeKineticsPilot(complexity_method="BIC")
    logs = pilot.run_pilot(rng)

    print("=" * 80)
    print("           AUTONOMOUS HYPOTHESIS ENGINE - ENZYME KINETICS SIMULATION")
    print("=" * 80)
    for line in logs:
        print(line)
    print("=" * 80)

if __name__ == "__main__":
    main()
