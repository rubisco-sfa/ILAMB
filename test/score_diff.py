import sys

import numpy as np
import pandas as pd

if len(sys.argv) != 3:
    print("usage: python score_diff.py scores1.csv scores2.csv")
    sys.exit(1)
gold = pd.read_csv(sys.argv[1]).set_index("Variables")
test = pd.read_csv(sys.argv[2]).set_index("Variables")
diff = np.abs(gold - test) / gold
if not (diff < 1e12).all()["LandTest"]:
    print("Test failed")
    print(diff[diff > 1e-12].dropna())
    sys.exit(1)
print("Test passed")
