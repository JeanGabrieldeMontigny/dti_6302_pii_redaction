import pandas as pd

from src.data.preprocess import preprocess
from src.data.split import (
    in_validation,
    leakage_report,
    remove_cross_split_leakage,
    split_train_val,
)


def test_validation_assignment_is_deterministic_and_seeded():
    uids = [f"u{i}" for i in range(2000)]
    a = [in_validation(u, 0.1, seed=42) for u in uids]
    assert a == [in_validation(u, 0.1, seed=42) for u in uids]
    assert a != [in_validation(u, 0.1, seed=7) for u in uids]
    assert 0.07 < sum(a) / len(a) < 0.13


def test_split_keeps_uid_groups_together(raw_train):
    df, _ = preprocess(raw_train)
    train, val = split_train_val(df, val_fraction=0.25, seed=42)
    assert len(train) + len(val) == len(df)
    assert not set(train["uid"]) & set(val["uid"])
    # both locale variants of a persona land in the same split
    assert (val.groupby("uid").size() == 2).all()


def test_cross_split_leakage_removed(raw_train, raw_test):
    train, _ = preprocess(raw_train)
    test, _ = preprocess(raw_test)
    leaked = test.iloc[[0]].copy()
    train = pd.concat([train, leaked], ignore_index=True)
    splits, removed = remove_cross_split_leakage({"train": train, "test": test})
    assert removed["train"] >= 1
    assert all(v == {"uid_overlap": 0, "text_overlap": 0} for v in leakage_report(splits).values())
