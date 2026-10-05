from dataset_tools.balloon import select_subset


def test_select_subset_is_seeded_sized_and_disjoint() -> None:
    names = [f"img{i}.jpg" for i in range(61)]
    train, val = select_subset(names, 40, 10, seed=0)
    assert (len(train), len(val)) == (40, 10)
    assert not set(train) & set(val)
    assert (train, val) == select_subset(list(reversed(names)), 40, 10, seed=0)
    assert (train, val) != select_subset(names, 40, 10, seed=1)
