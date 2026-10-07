import pandas as pd
import argparse

parser = argparse.ArgumentParser(description = "Read results from an Excel file")
parser.add_argument('--file', type=str)

args = parser.parse_args()
df = pd.read_excel(args.file)

summary = df.groupby(
    ["Project", "Encoder", "Adapter", "Aggregator", "Init", "LRate", "K-shots"]
)["BACC_Val"].agg(["mean", "std", "count"])

print(summary)