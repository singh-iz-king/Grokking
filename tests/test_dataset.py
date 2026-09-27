import random

from src.dataset import make_modular_addition_data


def test_dataset_is_complete_and_split_is_disjoint() -> None:
    data = make_modular_addition_data(11, 0.3, 7)
    assert data.all_inputs.shape == (121, 3)
    assert data.all_labels.shape == (121,)
    assert data.train_inputs.shape[0] == int(0.3 * 121)
    assert data.train_inputs.shape[0] + data.test_inputs.shape[0] == 121
    train_pairs = {tuple(pair[:2].tolist()) for pair in data.train_inputs}
    test_pairs = {tuple(pair[:2].tolist()) for pair in data.test_inputs}
    assert train_pairs.isdisjoint(test_pairs)
    assert train_pairs | test_pairs == {
        (a, b) for a in range(11) for b in range(11)
    }
    assert all(
        int(label) == (int(pair[0]) + int(pair[1])) % 11
        for pair, label in zip(data.all_inputs, data.all_labels)
    )


def test_dataset_split_is_seed_deterministic() -> None:
    first = make_modular_addition_data(11, 0.3, 42)
    second = make_modular_addition_data(11, 0.3, 42)
    other = make_modular_addition_data(11, 0.3, 43)
    assert first.train_indices.equal(second.train_indices)
    assert first.test_indices.equal(second.test_indices)
    assert not first.train_indices.equal(other.train_indices)


def test_p113_split_matches_reference_python_shuffle() -> None:
    modulus = 113
    pairs = [(a, b) for a in range(modulus) for b in range(modulus)]
    random.seed(0)
    random.shuffle(pairs)
    split = int(0.3 * len(pairs))

    data = make_modular_addition_data(modulus, 0.3, 0)
    actual_train = [
        tuple(pair[:2].tolist()) for pair in data.train_inputs
    ]
    assert actual_train == pairs[:split]
    assert len(actual_train) == 3830
    assert data.test_inputs.shape[0] == 8939
