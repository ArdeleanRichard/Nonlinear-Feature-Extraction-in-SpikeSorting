import os
import pandas as pd

def filter_columns_and_save(input_csv, columns):
    df = pd.read_csv(input_csv)

    df_filtered = df[columns]

    base_name, ext = os.path.splitext(input_csv)
    output_csv = f"{base_name}_simple{ext}"

    df_filtered.to_csv(output_csv, index=False, header=False)

    return df_filtered.to_numpy()