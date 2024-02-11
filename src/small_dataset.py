import glob
import pandas as pd


def load_as_df() -> pd.DataFrame:
    # Get a list of file paths for all CSV files in the data/ folder
    csv_files = glob.glob("../data/*.csv")

    # Create an empty list to store the DataFrames
    dfs = []

    # Read each CSV file into a DataFrame and append it to the list
    for file in csv_files:
        df = pd.read_csv(file)
        df["user"] = file.split("/")[-1].split(".")[
            0
        ]  # Extract the user name from the file path
        dfs.append(df)

    # Concatenate all the DataFrames in the list
    df = pd.concat(dfs)
    df = df[
        [
            "user",
            "card_id",
            "review_th",
            "rating",
            "delta_t",
        ]
    ]

    return df
