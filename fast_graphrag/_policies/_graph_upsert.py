import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import Counter, Dict, Iterable, List, Optional, Set, Tuple, Union

from fast_graphrag._llm._base import format_and_send_prompt
from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._prompt import PROMPTS
from fast_graphrag._storage._base import BaseGraphStorage
from fast_graphrag._types import GTEdge, GTId, GTNode, TEditRelationList, TEntity, THash, TId, TIndex, TRelation
from fast_graphrag._utils import logger

from ._base import BaseEdgeUpsertPolicy, BaseGraphUpsertPolicy, BaseNodeUpsertPolicy


async def summarize_entity_description(
    prompt: str, description: str, llm: BaseLLMService, max_tokens: Optional[int] = None
) -> str:
    """Summarize the given entity description."""
    if max_tokens is not None:
        raise NotImplementedError("Summarization with max tokens is not yet supported.")
    # Prompt
    entity_description_summarization_prompt = prompt

    # Extract entities and relationships
    formatted_entity_description_summarization_prompt = entity_description_summarization_prompt.format(
        description=description
    )
    new_description, _ = await llm.send_message(
        prompt=formatted_entity_description_summarization_prompt, response_model=str, max_tokens=max_tokens
    )

    return new_description


####################################################################################################
# DEFAULT GRAPH UPSERT POLICIES
####################################################################################################


@dataclass
class DefaultNodeUpsertPolicy(BaseNodeUpsertPolicy[GTNode, GTId]):
    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, GTEdge, GTId], source_nodes: Iterable[GTNode]
    ) -> Tuple[BaseGraphStorage[GTNode, GTEdge, GTId], Iterable[Tuple[TIndex, GTNode]]]:
        upserted: Dict[TIndex, GTNode] = {}
        for node in source_nodes:
            _, index = await target.get_node(node)
            if index is not None:
                await target.upsert_node(node=node, node_index=index)
            else:
                index = await target.upsert_node(node=node, node_index=None)
            upserted[index] = node

        return target, upserted.items()


@dataclass
class DefaultEdgeUpsertPolicy(BaseEdgeUpsertPolicy[GTEdge, GTId]):
    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, GTEdge, GTId], source_edges: Iterable[GTEdge]
    ) -> Tuple[BaseGraphStorage[GTNode, GTEdge, GTId], Iterable[Tuple[TIndex, GTEdge]]]:
        upserted: List[Tuple[TIndex, GTEdge]] = []
        for edge in source_edges:
            index = await target.upsert_edge(edge=edge, edge_index=None)
            upserted.append((index, edge))
        return target, upserted


@dataclass
class DefaultGraphUpsertPolicy(BaseGraphUpsertPolicy[GTNode, GTEdge, GTId]):  # noqa: N801
    async def __call__(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, GTEdge, GTId],
        source_nodes: Iterable[GTNode],
        source_edges: Iterable[GTEdge],
    ) -> Tuple[
        BaseGraphStorage[GTNode, GTEdge, GTId],
        Iterable[Tuple[TIndex, GTNode]],
        Iterable[Tuple[TIndex, GTEdge]],
    ]:
        target, upserted_nodes = await self._nodes_upsert(llm, target, source_nodes)
        target, upserted_edges = await self._edges_upsert(llm, target, source_edges)

        return target, upserted_nodes, upserted_edges


####################################################################################################
# NODE UPSERT POLICIES
####################################################################################################


@dataclass
class NodeUpsertPolicy_SummarizeDescription(BaseNodeUpsertPolicy[TEntity, TId]):  # noqa: N801
    @dataclass
    class Config:
        max_node_description_size: int = field(default=512)
        node_summarization_ratio: float = field(default=0.5)
        node_summarization_prompt: str = field(default=PROMPTS["summarize_entity_descriptions"])
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[TEntity, GTEdge, TId], source_nodes: Iterable[TEntity]
    ) -> Tuple[BaseGraphStorage[TEntity, GTEdge, TId], Iterable[Tuple[TIndex, TEntity]]]:
        upserted: List[Tuple[TIndex, TEntity]] = []

        async def _upsert_node(node_id: TId, nodes: List[TEntity]) -> Optional[Tuple[TIndex, TEntity]]:
            existing_node, index = await target.get_node(node_id)
            if existing_node:
                nodes.append(existing_node)

            # Resolve descriptions
            node_description = "\n".join((node.description for node in nodes))

            if len(node_description) > self.config.max_node_description_size:
                node_description = await summarize_entity_description(
                    self.config.node_summarization_prompt,
                    node_description,
                    llm,
                    # int(
                    #     self.config.max_node_description_size
                    #     * self.config.node_summarization_ratio
                    #     / TOKEN_TO_CHAR_RATIO
                    # ),
                )

            # Resolve types (pick most frequent)
            node_type = Counter((node.type for node in nodes)).most_common(1)[0][0]

            node = TEntity(name=node_id, description=node_description, type=node_type)
            index = await target.upsert_node(node=node, node_index=index)

            upserted.append((index, node))

        # Group nodes by name
        grouped_nodes: Dict[TId, List[TEntity]] = defaultdict(lambda: [])
        for node in source_nodes:
            grouped_nodes[node.name].append(node)

        if self.config.is_async:
            node_upsert_tasks = (_upsert_node(node_id, nodes) for node_id, nodes in grouped_nodes.items())
            await asyncio.gather(*node_upsert_tasks)
        else:
            for node_id, nodes in grouped_nodes.items():
                await _upsert_node(node_id, nodes)

        return target, upserted


####################################################################################################
# EDGE UPSERT POLICIES
####################################################################################################


@dataclass
class EdgeUpsertPolicy_UpsertIfValidNodes(BaseEdgeUpsertPolicy[TRelation, TId]):  # noqa: N801
    @dataclass
    class Config:
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, TRelation, TId], source_edges: Iterable[TRelation]
    ) -> Tuple[BaseGraphStorage[GTNode, TRelation, TId], Iterable[Tuple[TIndex, TRelation]]]:
        upserted_edges: Dict[TIndex, TRelation] = {}

        async def _upsert_edge(edge: TRelation) -> Optional[Tuple[TIndex, TRelation]]:
            source_node, _ = await target.get_node(edge.source)
            target_node, _ = await target.get_node(edge.target)

            if source_node and target_node:
                index = await target.upsert_edge(edge=edge, edge_index=None)
                upserted_edges[index] = edge

        if self.config.is_async:
            edge_upsert_tasks = (_upsert_edge(edge) for edge in source_edges)
            await asyncio.gather(*edge_upsert_tasks)
        else:
            for edge in source_edges:
                await _upsert_edge(edge)

        return target, upserted_edges.items()


@dataclass
class EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM(BaseEdgeUpsertPolicy[TRelation, TId]):  # noqa: N801
    @dataclass
    class Config:
        edge_merge_threshold: int = field(default=5)
        is_async: bool = field(default=True)

    config: Config = field(default_factory=Config)

    async def _upsert_edge(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, TRelation, TId],
        edges: List[TRelation],
        source_entity: TId,
        target_entity: TId,
    ) -> List[Tuple[TIndex, TRelation]]:
        existing_edges = list(await target.get_edges(source_entity, target_entity))

        # Check if we need to run edges maintenance
        if (len(existing_edges) + len(edges)) >= self.config.edge_merge_threshold:
            upserted_eges = await self._merge_similar_edges(llm, target, existing_edges, edges)
        else:
            upserted_eges: List[Tuple[TIndex, TRelation]] = []
            for edge in edges:
                upserted_eges.append((await target.upsert_edge(edge, None), edge))

        return upserted_eges

    async def _merge_similar_edges(
        self,
        llm: BaseLLMService,
        target: BaseGraphStorage[GTNode, TRelation, TId],
        existing_edges: List[Tuple[TRelation, TIndex]],
        edges: List[TRelation],
    ) -> List[Tuple[TIndex, TRelation]]:
        upserted_eges: List[Tuple[TIndex, TRelation]] = []
        map_incremental_to_edge: Dict[int, Tuple[TRelation, Union[TIndex, None]]] = {
            **dict(enumerate(existing_edges)),
            **{idx + len(existing_edges): (edge, None) for idx, edge in enumerate(edges)},
        }

        # Extract entities and relationships
        edge_grouping, _ = await format_and_send_prompt(
            prompt_key="edges_group_similar",
            llm=llm,
            format_kwargs={
                "edge_list": "\n".join(
                    (f"{idx}, {edge.description}" for idx, (edge, _) in map_incremental_to_edge.items())
                )
            },
            response_model=TEditRelationList,
        )

        def is_existing(index: int) -> bool:
            return index < len(existing_edges)

        indices_to_delete: Dict[TIndex, bool] = {}
        for edges_group in edge_grouping.groups:
            relation_indices = [
                index
                for index in edges_group.ids
                if index < len(existing_edges) + len(edges)  # Only consider valid indices
            ]
            if len(relation_indices) < 2:
                logger.info("LLM returned invalid index for edge maintenance, ignoring.")
                continue

            chunks: Set[THash] = set()

            for second in relation_indices[1:]:
                if is_existing(second):
                    relationship, index = existing_edges[second]

                    if relationship.chunks:
                        chunks.update(relationship.chunks)

                    if index not in indices_to_delete:
                        indices_to_delete[index] = True

            first_index = relation_indices[0]
            if is_existing(first_index):
                edge, index = existing_edges[first_index]
                indices_to_delete[index] = False
                edge.description = edges_group.description
            else:
                edge, index = map_incremental_to_edge[first_index]

            if edge.chunks:
                chunks.update(edge.chunks)

            edge.chunks = chunks
            upserted_eges.append((await target.upsert_edge(edge, index), edge))

        for idx, (edge, _) in map_incremental_to_edge.items():
            if len(existing_edges) <= idx < len(existing_edges) + len(edges):  # New edge
                upserted_eges.append((await target.upsert_edge(edge, None), edge))

        await target.delete_edges_by_index([i for i, v in indices_to_delete.items() if v])

        return upserted_eges

    async def __call__(
        self, llm: BaseLLMService, target: BaseGraphStorage[GTNode, TRelation, TId], source_edges: Iterable[TRelation]
    ) -> Tuple[BaseGraphStorage[GTNode, TRelation, TId], Iterable[Tuple[TIndex, TRelation]]]:
        grouped_edges: Dict[Tuple[TId, TId], List[TRelation]] = defaultdict(lambda: [])

        for edge in source_edges:
            grouped_edges[(edge.source, edge.target)].append(edge)

        if self.config.is_async:
            edge_upsert_tasks = (
                self._upsert_edge(llm, target, edges, source_entity, target_entity)
                for (source_entity, target_entity), edges in grouped_edges.items()
            )
            return target, chain(*await asyncio.gather(*edge_upsert_tasks))
        else:
            return target, chain(
                *[
                    await self._upsert_edge(llm, target, edges, source_entity, target_entity)
                    for (source_entity, target_entity), edges in grouped_edges.items()
                ]
            )
