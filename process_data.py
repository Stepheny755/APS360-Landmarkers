import os
import pandas as pd
import numpy as np


def main(folder_path):
    out_data = []
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if not ("runs" in root):
                out_data.append(os.path.join(root, f))

    out_data.sort()

    for f in out_data:
        df = pd.read_excel(f)
        df_dict = df.to_dict(orient="list")

        print(f"{max(df_dict['val_acc'])} for {f}")
        

if __name__ == "__main__":
    folder_path = "eval"
    main(folder_path)