import numpy as np
import unittest

from app.rl.observation import OBSERVATION_VECTOR_SIZES
from app.rl.setup_supervised import VERTICES, assess, collect_examples, make_model, train


class SetupSupervisedTests(unittest.TestCase):
    def test_setup_examples_are_legal_and_board_split_ready(self):
        examples = collect_examples(range(35001, 35003))
        assert examples["observations"].shape == (4, OBSERVATION_VECTOR_SIZES["v2"])
        assert examples["masks"].shape == (4, VERTICES)
        assert examples["candidates"].shape == (4, VERTICES, 12)
        assert np.all(examples["masks"][np.arange(4), examples["labels"]])
        assert examples["placements"].tolist() == [1, 2, 1, 2]
        assert examples["seeds"].tolist() == [35001, 35001, 35002, 35002]


    def test_supervised_model_trains_without_illegal_predictions(self):
        examples = collect_examples(range(35003, 35007))
        for architecture in ("mlp", "shared"):
            model, history = train(examples, examples, seed=7, epochs=2,
                                   architecture=architecture)
            result = assess(model, examples)
            assert len(history) == 2
            assert result["examples"] == 8
            assert 0 <= result["teacher_agreement"] <= 1
            assert result["mean_pips_regret"] >= 0
