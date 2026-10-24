"""Never invoked by the default suite or CI; explicitly billable."""

import numpy as np
import pytest

from omniseek_multimodal.cloud import ModalEmbedder
from omniseek_multimodal.config import Settings


@pytest.mark.gpu
def test_live_text_embedding():
    settings = Settings(allow_cloud=True)
    vectors, metadata = ModalEmbedder(settings).embed(
        [{"kind": "text", "text": "Find the signal", "prompt": "SearchQuery"}]
    )
    assert vectors.shape == (1, 768)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)
    assert metadata["gpu"]
