import pytest
import yaml
from pathlib import Path

from mode_eci.data import Corpus
from mode_eci.verification import fixture_config


@pytest.fixture(scope="session")
def assets(tmp_path_factory):
    base = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "bert_mlm.yaml").read_text(encoding="utf-8"))
    config = fixture_config(base, tmp_path_factory.mktemp("local_models"))
    return config, Corpus(config["data_dir"])
