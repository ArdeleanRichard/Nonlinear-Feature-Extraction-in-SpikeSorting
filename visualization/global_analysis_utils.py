import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
from scipy import stats

from constants import LABEL_COLOR_MAP_SMALLER
import seaborn as sn

def filter_columns_and_save(input_csv, columns):
    df = pd.read_csv(input_csv)

    df_filtered = df[columns]

    base_name, ext = os.path.splitext(input_csv)
    output_csv = f"{base_name}_simple{ext}"

    df_filtered.to_csv(output_csv, index=False, header=False)

    return df_filtered.to_numpy()