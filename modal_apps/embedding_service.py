"""Repository entrypoint; packaged implementation is shared with the local API."""

from omniseek_multimodal.modal_apps.embedding_service import app, embed_l4, embed_t4, preflight

__all__ = ["app", "embed_l4", "embed_t4", "preflight"]
