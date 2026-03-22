import pandas as pd
import glob
import os

# Folder containing CSV files
folder_path = "data/source/AgriculturalMetrics"

maha_df = []
yala_df = []

# Read all csv files
for file in glob.glob(os.path.join(folder_path, "*.csv")):
    # Example filename: Maha_2021.csv
    filename = os.path.basename(file)
    season, year = filename.replace(".csv", "").split("_")

    df = pd.read_csv(file)

    # Add new columns to specify yala or maha
    df["Year"] = int(year)
    df["Season"] = season  # Maha or Yala

    if season == "Maha":
        maha_df.append(df)
    else:
        yala_df.append(df)
        
    

# Combine all files into one DataFrame
final_df_mahaseason = pd.concat(maha_df, ignore_index=True)
final_df_yalaseason = pd.concat(yala_df, ignore_index=True)

final_df = final_df_mahaseason.merge(final_df_yalaseason,on=["District", "Year"],how="left")
final_df = final_df[["District","MRiceArea","MRiceYield","Year","SRiceArea","SRiceYield"]]                         

print(final_df)

# Save
final_df.to_csv("agridata.csv", index=False)


