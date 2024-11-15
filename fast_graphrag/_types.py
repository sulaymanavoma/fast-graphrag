from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, TypeAlias, TypeVar, Union

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, field_validator
from pydantic._internal import _model_construction

####################################################################################################
# GENERICS
####################################################################################################

# Blob
GTBlob = TypeVar("GTBlob")

# KeyValue
GTKey = TypeVar("GTKey")
GTValue = TypeVar("GTValue")

# Vectordb
GTEmbedding = TypeVar("GTEmbedding")
GTHash = TypeVar("GTHash")

# Graph
GTGraph = TypeVar("GTGraph")
GTId = TypeVar("GTId")


@dataclass
class BTNode:
    name: Any


GTNode = TypeVar("GTNode", bound=BTNode)


@dataclass
class BTEdge:
    source: Any
    target: Any

    @staticmethod
    def to_attrs(
        edge: Optional[Any] = None, edges: Optional[Iterable[Any]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        raise NotImplementedError


GTEdge = TypeVar("GTEdge", bound=BTEdge)
GTChunk = TypeVar("GTChunk")


# LLM
def _schema_no_title(schema: dict[str, Any]) -> None:
    schema.pop("required")
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)


class MetaModel(_model_construction.ModelMetaclass):
    def __new__(
        cls, name: str, bases: tuple[type[Any], ...], dct: Dict[str, Any], alias: Optional[str] = None, **kwargs: Any
    ) -> type:
        if alias:
            dct["__qualname__"] = alias
        if "BaseModel" not in [base.__name__ for base in bases]:
            bases = bases + (BaseModel,)
        return super().__new__(cls, name, bases, dct, json_schema_extra=_schema_no_title, **kwargs)


class BTResponseModel:
    class Model(BaseModel):
        @staticmethod
        def to_dataclass(pydantic: Any) -> Any:
            raise NotImplementedError

    def to_str(self) -> str:
        raise NotImplementedError


GTResponseModel = TypeVar("GTResponseModel", bound=Union[str, BaseModel, BTResponseModel])

####################################################################################################
# TYPES
####################################################################################################


def dump_to_csv(
    data: Iterable[object],
    fields: List[str],
    separator: str = ";;",
    with_header: bool = False,
    **values: Dict[str, List[Any]],
) -> List[str]:
    rows = list(
        chain(
            (separator.join(chain(fields, values.keys())),) if with_header else (),
            chain(
                separator.join(
                    chain(
                        (str(getattr(d, field)) for field in fields),
                        (str(v) for v in vs),
                    )
                )
                for d, *vs in zip(data, *values.values())
            ),
        )
    )
    return rows


# Embedding types
TEmbeddingType: TypeAlias = np.float32
TEmbedding: TypeAlias = npt.NDArray[TEmbeddingType]

THash: TypeAlias = np.uint64
TScore: TypeAlias = np.float32
TIndex: TypeAlias = int
TId: TypeAlias = str


@dataclass
class TDocument:
    """A class for representing a piece of data."""

    data: str = field()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TChunk:
    """A class for representing a chunk."""

    id: THash = field()
    content: str = field()
    metadata: Dict[str, Any] = field(default_factory=dict)


# Graph types
@dataclass
class TEntity(BTResponseModel, BTNode):
    name: str = field()
    type: str = Field()
    description: str = Field()

    def to_str(self) -> str:
        return f"{self.name}: {self.description}"

    class Model(BTResponseModel.Model, metaclass=MetaModel, alias="Entity"):
        name: str = Field(..., description="The name of the entity.")
        type: str = Field(..., description="The type of the entity.")
        description: str = Field(..., description="The description of the entity, in details and comprehensive.")

        @staticmethod
        def to_dataclass(pydantic: "TEntity.Model") -> "TEntity":
            return TEntity(name=pydantic.name, type=pydantic.type, description=pydantic.description)

        @field_validator("name", mode="before")
        @classmethod
        def uppercase_name(cls, value: str):
            return value.upper() if value else value

        @field_validator("type", mode="before")
        @classmethod
        def uppercase_type(cls, value: str):
            return value.upper() if value else value


class TQueryEntities(BaseModel):
    entities: List[str] = Field(
        ...,
        description=(
            "The named entities in the query. This list can be empty if there are no meaningful entities in the query."
        ),
    )
    n: int = Field(..., description="The number of named entities found.")  # So that the LLM can answer 0.

    @field_validator("entities", mode="before")
    @classmethod
    def uppercase_source(cls, value: List[str]):
        return [e.upper() for e in value] if value else value


@dataclass
class TRelation(BTResponseModel, BTEdge):
    source: str = field()
    target: str = field()
    description: str = field()
    chunks: List[THash] | None = field(default=None)

    @staticmethod
    def to_attrs(
        edge: Optional["TRelation"] = None,
        edges: Optional[Iterable["TRelation"]] = None,
        include_source_target: bool = False,
        **_,
    ) -> Dict[str, Any]:
        if edge is not None:
            assert edges is None, "Either edge or edges should be provided, not both."
            return {
                "description": edge.description,
                "chunks": edge.chunks,
                **({
                    "source": edge.source,
                    "target": edge.target,
                } if include_source_target else {}),
            }
        elif edges is not None:
            return {
                "description": [e.description for e in edges],
                "chunks": [e.chunks for e in edges],
                **({
                    "source": [e.source for e in edges],
                    "target": [e.target for e in edges],
                } if include_source_target else {}),
            }
        else:
            return {}

    class Model(BTResponseModel.Model, metaclass=MetaModel, alias="Relation"):
        source: str = Field(..., description="The name of the source entity.", alias="source_entity")
        target: str = Field(..., description="The name of the target entity.", alias="target_entity")
        description: str = Field(
            ..., description="The description of the relationship between the source and target entity"
        )

        @staticmethod
        def to_dataclass(pydantic: "TRelation.Model") -> "TRelation":
            return TRelation(source=pydantic.source, target=pydantic.target, description=pydantic.description)

        @field_validator("source", mode="before")
        @classmethod
        def uppercase_source(cls, value: str):
            return value.upper() if value else value

        @field_validator("target", mode="before")
        @classmethod
        def uppercase_target(cls, value: str):
            return value.upper() if value else value


@dataclass
class TGraph(BTResponseModel):
    entities: List[TEntity] = field()
    relationships: List[TRelation] = field()

    class Model(BTResponseModel.Model, metaclass=MetaModel, alias="Graph"):
        entities: List[TEntity.Model] = Field(description="The entities in the graph.")
        relationships: List[TRelation.Model] = Field(description="The relationships between the entities.")

        @staticmethod
        def to_dataclass(pydantic: "TGraph.Model") -> "TGraph":
            return TGraph(
                entities=[p.to_dataclass(p) for p in pydantic.entities],
                relationships=[p.to_dataclass(p) for p in pydantic.relationships],
            )


class TEditRelation(BaseModel):
    ids: List[int] = Field(..., description="The ids of the facts that you are combining into one.")
    description: str = Field(
        ..., description="The summarized description of the combined facts, in detail and comprehensive."
    )


class TEditRelationList(BaseModel):
    groups: List[TEditRelation] = Field(
        ...,
        description="The list of new fact groups. Include only groups of more than one fact.",
        alias="grouped_facts",
    )


@dataclass
class TContext(Generic[GTNode, GTEdge, GTHash, GTChunk]):
    """A class for representing the context used to generate a query response."""

    entities: List[Tuple[GTNode, TScore]]
    relationships: List[Tuple[GTEdge, TScore]]
    chunks: List[Tuple[GTChunk, TScore]]

    def to_str(self, max_chars: Dict[str, int]) -> str:
        """Convert the context to a string representation."""
        csv_tables: Dict[str, List[str]] = {
            "entities": dump_to_csv([e for e, _ in self.entities], ["name", "description"], with_header=True),
            "relationships": dump_to_csv(
                [r for r, _ in self.relationships], ["source", "target", "description"], with_header=True
            ),
            "chunks": dump_to_csv([c for c, _ in self.chunks], ["content"], with_header=True),
        }
        csv_tables_row_length = {k: [len(row) for row in table] for k, table in csv_tables.items()}

        include_up_to = {
            "entities": 0,
            "relationships": 0,
            "chunks": 0,
        }

        # Truncate each csv to the maximum number of assigned tokens
        chars_remainder = 0
        while True:
            last_char_remainder = chars_remainder
            # Keep augmenting the context until feasible
            for table in csv_tables:
                for i in range(include_up_to[table], len(csv_tables_row_length[table])):
                    length = csv_tables_row_length[table][i] + 1  # +1 for the newline character
                    if length <= chars_remainder:  # use up the remainder
                        include_up_to[table] += 1
                        chars_remainder -= length
                    elif length <= max_chars[table]:  # use up the assigned tokens
                        include_up_to[table] += 1
                        max_chars[table] -= length
                    else:
                        break

                if max_chars[table] >= 0:  # if the assigned tokens are not used up store in the remainder
                    chars_remainder += max_chars[table]
                    max_chars[table] = 0

            # Truncate the csv
            if chars_remainder == last_char_remainder:
                break

        data: List[str] = []
        if len(self.entities):
            data.extend(
                [
                    "\n#Entities",
                    "```csv",
                    *csv_tables["entities"][: include_up_to["entities"]],
                    "```",
                ]
            )
        else:
            data.append("\n#Entities: None\n")

        if len(self.relationships):
            data.extend(
                [
                    "\n#Relationships",
                    "```csv",
                    *csv_tables["relationships"][: include_up_to["relationships"]],
                    "```",
                ]
            )
        else:
            data.append("\n\n#Relationships: None\n")

        if len(self.chunks):
            data.extend(
                [
                    "\n\n#Sources",
                    "```csv",
                    *csv_tables["chunks"][: include_up_to["chunks"]],
                    "```",
                ]
            )
        else:
            data.append("\n\n#Sources: None\n")
        return "\n".join(data)


@dataclass
class TQueryResponse(Generic[GTNode, GTEdge, GTHash, GTChunk]):
    """A class for representing a query response."""

    response: str
    context: TContext[GTNode, GTEdge, GTHash, GTChunk]
