from dataclasses import dataclass
from temibox.domain import Label


@dataclass
class Publication:
    pub_id:       int | None
    title:        str
    abstract:     str
    topics:       list[Label]
    keywords:     list[Label]
    language:     str
