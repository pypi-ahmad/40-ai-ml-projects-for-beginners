import pytest
from pydantic import ValidationError

from ml_package.validation import IrisBatchFeatures, IrisFeatures, IrisValidator


class TestIrisFeatures:
    def test_valid_features(self):
        f = IrisFeatures(
            sepal_length=5.1, sepal_width=3.5,
            petal_length=1.4, petal_width=0.2
        )
        assert f.sepal_length == 5.1

    def test_invalid_sepal_length_negative(self):
        with pytest.raises(ValidationError):
            IrisFeatures(
                sepal_length=-1, sepal_width=3.5,
                petal_length=1.4, petal_width=0.2
            )

    def test_invalid_sepal_length_zero(self):
        with pytest.raises(ValidationError, match="sepal_length"):
            IrisFeatures(
                sepal_length=0, sepal_width=3.5,
                petal_length=1.4, petal_width=0.2
            )

    def test_invalid_sepal_width_excessive(self):
        with pytest.raises(ValidationError):
            IrisFeatures(
                sepal_length=5.1, sepal_width=20,
                petal_length=1.4, petal_width=0.2
            )

    def test_invalid_petal_length_zero(self):
        with pytest.raises(ValidationError):
            IrisFeatures(
                sepal_length=5.1, sepal_width=3.5,
                petal_length=0, petal_width=0.2
            )


class TestIrisBatchFeatures:
    def test_valid_batch(self):
        batch = IrisBatchFeatures(samples=[
            {"sepal_length": 5.1, "sepal_width": 3.5,
             "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.2, "sepal_width": 2.9,
             "petal_length": 4.3, "petal_width": 1.3},
        ])
        assert len(batch.samples) == 2

    def test_empty_batch(self):
        with pytest.raises(ValidationError):
            IrisBatchFeatures(samples=[])


class TestIrisValidator:
    def test_validate_request_valid(self):
        errors = IrisValidator.validate_request(
            sepal_length=5.1, sepal_width=3.5,
            petal_length=1.4, petal_width=0.2
        )
        assert errors == []

    def test_validate_request_negative(self):
        errors = IrisValidator.validate_request(
            sepal_length=-5, sepal_width=3.5,
            petal_length=1.4, petal_width=0.2
        )
        assert len(errors) > 0

    def test_validate_request_large_value(self):
        errors = IrisValidator.validate_request(
            sepal_length=5.1, sepal_width=3.5,
            petal_length=1.4, petal_width=100
        )
        assert len(errors) > 0

    def test_validate_request_multiple_errors(self):
        errors = IrisValidator.validate_request(
            sepal_length=0, sepal_width=0,
            petal_length=0, petal_width=0
        )
        assert len(errors) >= 4

    def test_validate_request_non_finite(self):
        errors = IrisValidator.validate_request(
            sepal_length=float("nan"),
            sepal_width=3.5,
            petal_length=1.4,
            petal_width=0.2,
        )
        assert len(errors) > 0
