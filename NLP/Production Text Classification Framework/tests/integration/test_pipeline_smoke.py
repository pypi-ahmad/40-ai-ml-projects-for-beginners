from datasets import Dataset, DatasetDict, Features, Value, ClassLabel

from textclf_framework.data.profiling import build_dataset_profile
from textclf_framework.data.versioning import build_manifest


def test_profile_and_manifest_smoke() -> None:
    features = Features({"text": Value("string"), "label": ClassLabel(names=["neg", "pos"])})
    train = Dataset.from_dict({"text": ["hello", "world"], "label": [0, 1]}, features=features)
    valid = Dataset.from_dict({"text": ["foo"], "label": [0]}, features=features)
    test = Dataset.from_dict({"text": ["bar"], "label": [1]}, features=features)
    ds = DatasetDict({"train": train, "validation": valid, "test": test})

    profile = build_dataset_profile("toy", ds)
    manifest = build_manifest(
        dataset_name="toy",
        source="memory",
        dataset_dict=ds,
        split_seed=42,
        preprocessing_config={"lowercase": True},
        label_names=["neg", "pos"],
    )

    assert profile.num_classes == 2
    assert manifest.dataset_name == "toy"
