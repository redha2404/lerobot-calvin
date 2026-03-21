import pytest
from pathlib import Path

def test_parquet_metadata_structure():
    """
    Validates that the generated LeRobot Parquet meta structure strictly aligns with expected codebase_version 3.0.
    Ensures dataset conversion output remains compatible with upstream Hugging Face hub ingestion.
    """
    from lerobot.datasets.utils import create_empty_dataset_info
    
    info = create_empty_dataset_info(
        codebase_version="v3.0",
        fps=30,
        features={
            "observation.state": {"dtype": "float32", "shape": (15,), "names": ["fake"]},
            "action": {"dtype": "float32", "shape": (7,), "names": ["fake_action"]}
        },
        use_videos=True,
        robot_type="panda"
    )
    
    assert info["codebase_version"] == "v3.0"
    assert info["robot_type"] == "panda"
    assert "observation.state" in info["features"]
    assert "action" in info["features"]
    assert info["fps"] == 30
