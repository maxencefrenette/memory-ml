from src.small_dataset import ReviewsDataset
from inline_snapshot import snapshot
import pandas as pd


def test_data_load_no_past_reviews():
    df = pd.DataFrame(
        {
            "user": [1],
            "card_id": [1],
            "rating": [3],
            "delta_t": [None],
        }
    )

    dataset = ReviewsDataset(reviews_history_size=4, df=df)

    assert len(dataset) == 1
    assert str(dataset[0]) == snapshot(
        "(tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), tensor([1.]))"
    )


def test_data_load_one_past_review():
    df = pd.DataFrame(
        {
            "user": [1, 1],
            "card_id": [1, 1],
            "rating": [3, 3],
            "delta_t": [None, 1],
        }
    )

    dataset = ReviewsDataset(reviews_history_size=4, df=df)

    assert len(dataset) == 2
    assert str(dataset[1]) == snapshot(
        """\
(tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.6931]), tensor([1.]))\
"""
    )


def test_data_load_full():
    df = pd.DataFrame(
        {
            "user": [1, 1, 1],
            "card_id": [1, 1, 1],
            "rating": [3, 3, 1],
            "delta_t": [None, 1, 2],
        }
    )

    dataset = ReviewsDataset(reviews_history_size=4, df=df)

    assert len(dataset) == 3
    assert str(dataset[2]) == snapshot(
        """\
(tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.0000, 1.0000, 1.0000, 0.6931, 1.0000, 1.0000, 1.0000, 1.0986]), tensor([0.]))\
"""
    )
