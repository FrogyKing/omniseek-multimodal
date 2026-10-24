from typing import Literal

from pydantic import BaseModel, Field, field_validator

Modality = Literal["text", "image", "audio", "video", "code"]


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)

    @field_validator("name")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Name cannot be blank")
        return value.strip()


class SearchRequest(BaseModel):
    query: str = Field(default="", max_length=2000)
    image_file_id: str | None = None
    collection_id: str | None = None
    modality: Modality | None = None
    dimension: Literal[768, 512, 256, 128] = 768
    k: int = Field(default=12, ge=1, le=100)


class IndexRequest(BaseModel):
    collection_id: str
    rebuild: bool = False
