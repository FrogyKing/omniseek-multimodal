import pytest
from fastapi.testclient import TestClient

from omniseek_multimodal.api import create_app
from omniseek_multimodal.config import ROOT, Settings
from omniseek_multimodal.demo import install_demo


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, allow_cloud=False)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def demo_client(settings):
    assert (ROOT / "experiments/results/demo-embeddings.npz").exists(), "Real recorded GPU fixture required"
    install_demo(settings)
    with TestClient(create_app(settings)) as client:
        yield client
