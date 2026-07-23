from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import infer_city_and_recording
from src.quality_checks import QualityConfig
from src.validation import ValidationConfig


def test_recording_name_detection():
    assert infer_city_and_recording("8_2_1.zip") == ("Tianjin", "8_2_1")
    assert infer_city_and_recording("xian_412_m1.zip")[0] == "Xi'an"
    assert infer_city_and_recording("unrelated_report.zip") is None


def test_default_configs_are_conservative():
    assert QualityConfig().minimum_duration_s == 5.0
    assert ValidationConfig().n_clusters == 4
