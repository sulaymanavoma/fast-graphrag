"""Top-level package for GraphRAG."""

from dataclasses import dataclass, field
from typing import Type

from fast_graphrag._llm import DefaultEmbeddingService, DefaultLLMService
from fast_graphrag._llm._base import BaseEmbeddingService
from fast_graphrag._llm._llm_openai import BaseLLMService
from fast_graphrag._policies._base import BaseGraphUpsertPolicy
from fast_graphrag._policies._graph_upsert import (
    DefaultGraphUpsertPolicy,
    EdgeUpsertPolicy_UpsertIfValidNodes,
    EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM,
    NodeUpsertPolicy_SummarizeDescription,
)
from fast_graphrag._policies._ranking import RankingPolicy_TopK, RankingPolicy_WithThreshold
from fast_graphrag._services import (
    BaseChunkingService,
    BaseInformationExtractionService,
    BaseStateManagerService,
    DefaultChunkingService,
    DefaultInformationExtractionService,
    DefaultStateManagerService,
)
from fast_graphrag._storage import (
    DefaultGraphStorage,
    DefaultGraphStorageConfig,
    DefaultIndexedKeyValueStorage,
    DefaultVectorStorage,
    DefaultVectorStorageConfig,
)
from fast_graphrag._types import TChunk, TEmbedding, TEntity, THash, TId, TIndex, TRelation

from ._graphrag import BaseGraphRAG


@dataclass
class GraphRAG(BaseGraphRAG[TEmbedding, THash, TChunk, TEntity, TRelation, TId]):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""

    class Config:
        """Configuration for the GraphRAG class."""
        llm_service_cls: Type[BaseLLMService] = DefaultLLMService
        embedding_service_cls: Type[BaseEmbeddingService] = DefaultEmbeddingService
        chunking_service_cls: Type[BaseChunkingService[TChunk]] = DefaultChunkingService
        information_extraction_service_cls: Type[BaseInformationExtractionService[TChunk, TEntity, TRelation, TId]] = (
            DefaultInformationExtractionService
        )
        information_extraction_upsert_policy: BaseGraphUpsertPolicy[TEntity, TRelation, TId] = DefaultGraphUpsertPolicy(
            config=NodeUpsertPolicy_SummarizeDescription.Config(),
            nodes_upsert_cls=NodeUpsertPolicy_SummarizeDescription,
            edges_upsert_cls=EdgeUpsertPolicy_UpsertIfValidNodes,
        )
        state_manager_cls: Type[BaseStateManagerService[TEntity, TRelation, THash, TChunk, TId, TEmbedding]] = (
            DefaultStateManagerService
        )

        graph_storage: DefaultGraphStorage[TEntity, TRelation, TId] = DefaultGraphStorage(
            DefaultGraphStorageConfig(node_cls=TEntity, edge_cls=TRelation)
        )
        entity_storage: DefaultVectorStorage[TIndex, TEmbedding] = DefaultVectorStorage(
            DefaultVectorStorageConfig(embedding_dim=DefaultEmbeddingService.embedding_dim)
        )
        chunk_storage: DefaultIndexedKeyValueStorage[THash, TChunk] = DefaultIndexedKeyValueStorage(None)

        entity_ranking_policy: RankingPolicy_WithThreshold = RankingPolicy_WithThreshold(
            RankingPolicy_WithThreshold.Config(threshold=0.05)
        )
        chunk_ranking_policy: RankingPolicy_TopK = RankingPolicy_TopK(RankingPolicy_TopK.Config(top_k=10))
        node_upsert_policy: NodeUpsertPolicy_SummarizeDescription = NodeUpsertPolicy_SummarizeDescription()
        edge_upsert_policy: EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM = (
            EdgeUpsertPolicy_UpsertValidAndMergeSimilarByLLM()
        )

    config: Config = field(default_factory=Config)

    def __post_init__(self):
        """Initialize the GraphRAG class."""
        self.llm_service = self.config.llm_service_cls()
        self.embedding_service = self.config.embedding_service_cls()
        self.chunking_service = self.config.chunking_service_cls()
        self.information_extraction_service = self.config.information_extraction_service_cls(
            graph_upsert=self.config.information_extraction_upsert_policy
        )
        self.state_manager = self.config.state_manager_cls(
            working_dir=self.working_dir,
            embedding_service=self.embedding_service,
            graph_storage=self.config.graph_storage,
            entity_storage=self.config.entity_storage,
            chunk_storage=self.config.chunk_storage,
            entity_ranking_policy=self.config.entity_ranking_policy,
            chunk_ranking_policy=self.config.chunk_ranking_policy,
            node_upsert_policy=self.config.node_upsert_policy,
            edge_upsert_policy=self.config.edge_upsert_policy,
        )

        super().__post_init__()
