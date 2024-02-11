import glob
import pandas as pd
from math import log
from torch.utils.data import Dataset
import torch
from typing import Tuple


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


class ReviewsDataset(Dataset):
    """
    A PyTorch Dataset that provides access to the reviews data.

    Features:
    - delta_t_is_null: whether the delta_t is null
    - delta_t: time since last review
    - past_reviews:
        - past_rating_is_null: whether the past rating is null
        - past_rating: the rating of the past review
        - past_delta_t_is_null: whether the past delta_t is null
        - past_delta_t: the time since the past review
    """

    def __init__(self, num_past_reviews: int) -> None:
        self.df = load_as_df()
        self.num_past_reviews = num_past_reviews

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]

        delta_t_is_null = int(row["delta_t"] == -1)
        delta_t = log(1 + row["delta_t"]) if not delta_t_is_null else 0
        rating = 0 if row["rating"] == 1 else 1

        past_reviews = []
        for i in range(1, self.num_past_reviews + 1):
            past_row = self.df.iloc[index - i]

            if past_row["user"] != row["user"] or past_row["card_id"] != row["card_id"]:
                past_reviews.append(
                    [
                        1,
                        0,
                        1,
                        0,
                    ]
                )
                continue

            past_rating_is_null = 0
            past_rating = 0 if past_row["rating"] == 1 else 1
            past_delta_t_is_null = int(past_row["delta_t"] == -1)
            past_delta_t = (
                log(1 + past_row["delta_t"]) if not past_delta_t_is_null else 0
            )

            past_reviews.append(
                [
                    past_rating_is_null,
                    past_rating,
                    past_delta_t_is_null,
                    past_delta_t,
                ]
            )

        return (
            torch.tensor(
                [delta_t_is_null, delta_t] + sum(past_reviews, []), dtype=torch.float32
            ),
            torch.tensor([rating], dtype=torch.float32),
        )
