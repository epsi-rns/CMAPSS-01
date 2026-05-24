# calc_operating_conditions.py
#
# Prints the six C-MAPSS operating conditions recovered by KMeans.
# Run from the folder containing train_FD002.txt (or train_FD004.txt).
#
# Usage:
#   python calc_operating_conditions.py
#
# Requires: pandas, scikit-learn, numpy

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

COLUMNS = [
    'unit_number', 'time_cycles',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'T2', 'T24', 'T30', 'T50',
    'P2', 'P15', 'P30',
    'Nf', 'Nc', 'epr', 'Ps30', 'phi',
    'NRf', 'NRc', 'BPR', 'farB', 'htBleed',
    'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32'
]

FLIGHT_PHASES = {
    # keyed by rounded (alt_kft, mach_2dp, tra)
    (0,  0.00, 100): 'sea level takeoff',
    (10, 0.25, 100): 'climb',
    (20, 0.70, 100): 'cruise, full thrust',
    (25, 0.62,  60): 'cruise, reduced thrust',
    (35, 0.84, 100): 'high altitude cruise',
    (42, 0.84, 100): 'top of climb',
}


def label_phase(c):
    key = (round(c[0]), round(c[1], 2), round(c[2]))
    return FLIGHT_PHASES.get(key, 'unknown')


def main():
    for fd in ['FD002', 'FD004']:
        try:
            train = pd.read_csv(
                f'train_{fd}.txt',
                sep=r'\s+', header=None, names=COLUMNS)
        except FileNotFoundError:
            print(f"train_{fd}.txt not found, skipping.")
            continue

        ops = train[['op_setting_1',
                     'op_setting_2',
                     'op_setting_3']].values

        km = KMeans(n_clusters=6, random_state=42, n_init=10)
        km.fit(ops)

        centers = km.cluster_centers_
        order   = np.argsort(centers[:, 0])

        print(f"\n{'='*62}")
        print(f"  {fd} — operating conditions (KMeans k=6)")
        print(f"{'='*62}")
        print(
            f"  {'Cond':>4}  "
            f"{'Alt (ft)':>10}  "
            f"{'Mach':>6}  "
            f"{'TRA (deg)':>9}  "
            f"{'Flight phase':<22}"
        )
        print(f"  {'-'*58}")

        for i in order:
            c = centers[i]
            phase = label_phase(c)
            print(
                f"  C{i+1:>1}    "
                f"{c[0]*1000:>10,.0f}  "
                f"{c[1]:>6.2f}  "
                f"{c[2]:>9.0f}  "
                f"{phase:<22}"
            )

        print(f"\n  Inertia (within-cluster sum of squares): "
              f"{km.inertia_:.2f}")
        print(f"  Near-zero inertia = 6 exact discrete flight points.")

    print(f"\n{'='*62}")
    print("  Notes:")
    print("  - op_setting_1 stored in thousands of feet in the file")
    print("  - op_setting_2 = Mach number (dimensionless)")
    print("  - op_setting_3 = TRA, Throttle Resolver Angle (degrees)")
    print("  - TRA 100° = full thrust, TRA 60° = reduced/cruise")
    print("  - FD001/FD003 have only 1 condition (sea level)")
    print(f"{'='*62}\n")


raise SystemExit(main())
