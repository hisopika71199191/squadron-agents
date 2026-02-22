"""
Graphiti Memory Integration

Temporal Knowledge Graph implementation using Graphiti.
Handles entity extraction, edge management, and temporal invalidation.
"""

import asyncio
import time
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import structlog

from squadron.core.config import MemoryConfig
from squadron.core.state import Message
from squadron.memory.types import (
    Edge,
    EdgeType,
    Entity,
    EntityType,
    Fact,
    MemoryQuery,
    MemoryResult,
)

logger = structlog.get_logger(__name__)


class GraphitiMemory:
    """
    Temporal Knowledge Graph memory system.
    
    Built on Graphiti (the open-source engine behind Zep).
    Provides:
    - Entity and relationship extraction from conversations
    - Temporal tracking with edge invalidation
    - Semantic search over the knowledge graph
    
    Example:
        ```python
        memory = GraphitiMemory(config=MemoryConfig())
        await memory.initialize()
        
        # Store conversation
        await memory.store(messages, session_id="session-1")
        
        # Retrieve relevant context
        result = await memory.retrieve(
            query="What does the user prefer?",
            session_id="session-1",
        )
        ```
    """

    def __init__(self, config: MemoryConfig | None = None):
        """
        Initialize the memory system.
        
        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        self._initialized = False
        self._graphiti_client: Any = None
        
        # In-memory caches (for development/testing)
        self._entities: dict[UUID, Entity] = {}
        self._edges: dict[UUID, Edge] = {}
        self._facts: dict[UUID, Fact] = {}
        
        # Session index
        self._session_entities: dict[str, set[UUID]] = {}
        self._session_edges: dict[str, set[UUID]] = {}

    async def initialize(self) -> None:
        """Initialize connection to the graph database."""
        if self._initialized:
            return
        
        try:
            # Try to import and initialize Graphiti
            # This is optional - we fall back to in-memory storage
            try:
                from graphiti_core import Graphiti
                from graphiti_core.nodes import EpisodeType
                import os
                
                if not self.config.neo4j_password:
                    logger.info("NEO4J_PASSWORD not set, using in-memory storage")
                    self._graphiti_client = None
                    self._initialized = True
                    return
                
                password = self.config.neo4j_password.get_secret_value()
                
                # Resolve API credentials (ALI takes priority over OPENAI env vars)
                api_key = (
                    (self.config.ali_api_key.get_secret_value() if self.config.ali_api_key else None)
                    or os.environ.get("ALI_API_KEY")
                    or (self.config.openai_api_key.get_secret_value() if self.config.openai_api_key else None)
                    or os.environ.get("OPENAI_API_KEY")
                )

                if not api_key:
                    logger.info(
                        "No LLM API key configured (ALI_API_KEY / OPENAI_API_KEY not set), "
                        "using in-memory storage"
                    )
                    self._graphiti_client = None
                    self._initialized = True
                    return

                base_url = (
                    (self.config.ali_base_url if self.config.ali_base_url else None)
                    or os.environ.get("ALI_BASE_URL")
                    or (self.config.openai_base_url if self.config.openai_base_url else None)
                    or os.environ.get("OPENAI_BASE_URL")
                )
                llm_model = (
                    self.config.llm_model
                    or os.environ.get("LLM_MODEL")
                    or "gpt-4o"
                )
                embed_model = (
                    self.config.embed_model
                    or os.environ.get("EMBED_MODEL")
                    or self.config.embedding_model
                )

                # Configure LLM Client using a patched OpenAIGenericClient that:
                # 1. Injects "json" into messages when using json_object response format
                #    (required by some providers like Ali Cloud / Qwen).
                # 2. Falls back from json_schema to json_object when the provider
                #    does not support structured outputs, ensuring ExtractedEntities
                #    and other response models are still validated correctly.
                # 3. Recursively resolves $ref in the JSON schema hint so that nested
                #    fields (e.g. ExtractedEntity.name inside ExtractedEntities) are
                #    shown to the model with their correct field names.
                from graphiti_core.llm_client import LLMConfig
                from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
                from graphiti_core.llm_client.config import ModelSize
                from graphiti_core.prompts.models import Message as GMessage
                import json as _json
                import typing as _typing
                from pydantic import BaseModel as _BaseModel

                class _AliCompatibleClient(OpenAIGenericClient):
                    """
                    Drop-in replacement for OpenAIGenericClient that works with
                    providers (e.g. Ali Cloud / Qwen) that:
                    - Require the word "json" to appear in the conversation when
                      response_format={'type': 'json_object'} is requested.
                    - Do not support response_format={'type': 'json_schema'} and
                      instead return free-form JSON that may not match field names.
                    """

                    async def _generate_response(  # type: ignore[override]
                        self,
                        messages: list[GMessage],
                        response_model: type[_BaseModel] | None = None,
                        max_tokens: int = 4096,
                        model_size: ModelSize = ModelSize.medium,
                    ) -> dict[str, _typing.Any]:
                        import openai as _openai  # local import to mirror parent

                        openai_messages = []
                        for m in messages:
                            m.content = self._clean_input(m.content)
                            if m.role == "user":
                                openai_messages.append({"role": "user", "content": m.content})
                            elif m.role == "system":
                                openai_messages.append({"role": "system", "content": m.content})

                        # Always use json_object format (broadest provider support).
                        # Inject a JSON-schema hint into the last user message so that:
                        #   a) Providers that require "json" in messages are satisfied.
                        #   b) Models without json_schema support still return the
                        #      correct field names by seeing the expected schema.
                        response_format: dict[str, _typing.Any] = {"type": "json_object"}

                        if response_model is not None:
                            schema = response_model.model_json_schema()
                            defs = schema.get("$defs", {})

                            def _resolve_ref(ref_str: str) -> dict:
                                """Resolve a $ref like '#/$defs/SomeModel' to its schema dict."""
                                parts = ref_str.lstrip("#/").split("/")
                                node: dict = schema
                                for part in parts:
                                    node = node.get(part, {})
                                return node

                            def _describe_props(props: dict, required: list, indent: int = 2) -> list[str]:
                                """Recursively build hint lines for a properties dict."""
                                pad = " " * indent
                                lines: list[str] = []
                                for field_name, field_info in props.items():
                                    req_marker = " (required)" if field_name in required else ""
                                    # Resolve $ref if present
                                    if "$ref" in field_info:
                                        ref_schema = _resolve_ref(field_info["$ref"])
                                        sub_props = ref_schema.get("properties", {})
                                        sub_required = ref_schema.get("required", [])
                                        if sub_props:
                                            lines.append(f'{pad}"{field_name}": object{req_marker} with fields:')
                                            lines.extend(_describe_props(sub_props, sub_required, indent + 2))
                                            continue
                                    # Array whose items are a $ref
                                    if field_info.get("type") == "array":
                                        items = field_info.get("items", {})
                                        if "$ref" in items:
                                            ref_schema = _resolve_ref(items["$ref"])
                                            sub_props = ref_schema.get("properties", {})
                                            sub_required = ref_schema.get("required", [])
                                            if sub_props:
                                                lines.append(
                                                    f'{pad}"{field_name}": array of objects{req_marker}, each with fields:'
                                                )
                                                lines.extend(_describe_props(sub_props, sub_required, indent + 2))
                                                continue
                                    description = field_info.get("description", field_info.get("type", "value"))
                                    lines.append(f'{pad}"{field_name}": {description}{req_marker}')
                                return lines

                            # Build a concise field list so the model knows what to return
                            props = schema.get("properties", {})
                            required = schema.get("required", [])
                            hint_lines = [
                                "Respond with a JSON object containing exactly these fields:",
                            ]
                            hint_lines.extend(_describe_props(props, required))
                            json_hint = "\n".join(hint_lines)
                            # Append hint to the last user message
                            if openai_messages:
                                openai_messages[-1]["content"] = (
                                    openai_messages[-1]["content"] + "\n\n" + json_hint
                                )
                        else:
                            # No schema — still ensure "json" appears somewhere so
                            # json_object mode is accepted by strict providers.
                            if openai_messages and "json" not in openai_messages[-1]["content"].lower():
                                openai_messages[-1]["content"] += (
                                    "\n\nRespond with a valid JSON object."
                                )

                        try:
                            response = await self.client.chat.completions.create(
                                model=self.model or "gpt-4o",
                                messages=openai_messages,  # type: ignore[arg-type]
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                                response_format=response_format,  # type: ignore[arg-type]
                            )
                            result = response.choices[0].message.content or "{}"
                            return _json.loads(result)
                        except _openai.RateLimitError as e:
                            from graphiti_core.llm_client.errors import RateLimitError
                            raise RateLimitError from e
                        except Exception as e:
                            logger.error(f"Error in generating LLM response: {e}")
                            raise

                llm_config = LLMConfig(
                    api_key=api_key,
                    base_url=base_url,
                    model=llm_model,
                )
                llm_client = _AliCompatibleClient(config=llm_config)

                # Configure Embedder
                # Use a batching-aware subclass that respects the Ali Cloud
                # embedding API limit of 10 inputs per request.
                from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

                _ALI_EMBED_BATCH_SIZE = 10

                class _BatchedEmbedder(OpenAIEmbedder):
                    """
                    OpenAIEmbedder that chunks create_batch() calls to avoid
                    exceeding the provider's per-request input limit (e.g. Ali Cloud
                    allows at most 10 texts per embedding API call).
                    """

                    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
                        if len(input_data_list) <= _ALI_EMBED_BATCH_SIZE:
                            return await super().create_batch(input_data_list)
                        # Process in chunks and concatenate results
                        all_embeddings: list[list[float]] = []
                        for i in range(0, len(input_data_list), _ALI_EMBED_BATCH_SIZE):
                            chunk = input_data_list[i : i + _ALI_EMBED_BATCH_SIZE]
                            chunk_embeddings = await super().create_batch(chunk)
                            all_embeddings.extend(chunk_embeddings)
                        return all_embeddings

                embedder_config = OpenAIEmbedderConfig(
                    api_key=api_key,
                    base_url=base_url,
                    embedding_model=embed_model,
                )
                embedder = _BatchedEmbedder(config=embedder_config)

                # Graphiti internally constructs an OpenAIRerankerClient() with no
                # parameters, which requires OPENAI_API_KEY in the environment.
                # Satisfy this by injecting the resolved api_key (which may be an
                # ALI_API_KEY or any other OpenAI-compatible key) as a fallback.
                os.environ.setdefault("OPENAI_API_KEY", api_key)

                client = Graphiti(
                    uri=self.config.neo4j_uri,
                    user=self.config.neo4j_user,
                    password=password,
                    llm_client=llm_client,
                    embedder=embedder,
                )
                await client.build_indices_and_constraints()

                # Backfill missing `entity_edges` property on Episodic nodes created
                # before this schema field was introduced, to suppress Neo4j warning
                # gql_status='01N52' ("property key does not exist").
                try:
                    await client.driver.execute_query(
                        "MATCH (e:Episodic) WHERE e.entity_edges IS NULL "
                        "SET e.entity_edges = []",
                    )
                    logger.debug("Backfilled entity_edges on legacy Episodic nodes")
                except Exception as migration_err:
                    logger.debug(
                        "entity_edges backfill skipped",
                        reason=str(migration_err),
                    )

                self._graphiti_client = client
                logger.info("Graphiti client initialized", uri=self.config.neo4j_uri)
            except ImportError:
                logger.warning(
                    "Graphiti not available, using in-memory storage",
                    hint="Install graphiti-core for persistent storage",
                )
            except Exception as e:
                self._graphiti_client = None
                logger.warning(
                    "Failed to connect to Neo4j, using in-memory storage",
                    error=str(e),
                )
            
            self._initialized = True
            logger.info("Memory system initialized")
            
        except Exception as e:
            logger.error("Memory initialization failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close the memory system connections."""
        if self._graphiti_client:
            await self._graphiti_client.close()
        self._initialized = False
        logger.info("Memory system closed")

    async def store(
        self,
        messages: tuple[Message, ...] | list[Message],
        session_id: str,
        extract_facts: bool = True,
    ) -> list[Fact]:
        """
        Store messages in the knowledge graph.
        
        Extracts entities and relationships from the messages
        and stores them with temporal tracking.
        
        Args:
            messages: Messages to store
            session_id: Session identifier
            extract_facts: Whether to extract facts from messages
            
        Returns:
            List of extracted facts
        """
        if not self._initialized:
            await self.initialize()
        
        logger.debug(
            "Storing messages",
            session_id=session_id,
            message_count=len(messages),
        )
        
        extracted_facts: list[Fact] = []
        
        if self._graphiti_client:
            # Use Graphiti for storage
            try:
                from graphiti_core.nodes import EpisodeType
                
                for msg in messages:
                    # Add episode to Graphiti
                    await self._graphiti_client.add_episode(
                        name=f"message_{msg.id}",
                        episode_body=msg.content,
                        source=EpisodeType.message,
                        reference_time=msg.timestamp,
                        source_description=f"Session: {session_id}, Role: {msg.role}",
                    )
                
                logger.debug("Messages stored in Graphiti")
                
            except Exception as e:
                logger.error("Graphiti storage failed", error=str(e))
                # Fall back to in-memory
                extracted_facts = await self._store_in_memory(messages, session_id)
        else:
            # Use in-memory storage
            extracted_facts = await self._store_in_memory(messages, session_id)
        
        return extracted_facts

    async def _store_in_memory(
        self,
        messages: tuple[Message, ...] | list[Message],
        session_id: str,
    ) -> list[Fact]:
        """Store messages in in-memory graph."""
        extracted_facts: list[Fact] = []
        
        # Initialize session indices
        if session_id not in self._session_entities:
            self._session_entities[session_id] = set()
        if session_id not in self._session_edges:
            self._session_edges[session_id] = set()
        
        for msg in messages:
            # Extract facts from message
            facts = await self._extract_facts(msg, session_id)
            
            for fact in facts:
                # Create or update entities
                subject_entity = await self._get_or_create_entity(
                    name=fact.subject,
                    session_id=session_id,
                    source_message_id=msg.id,
                )
                object_entity = await self._get_or_create_entity(
                    name=fact.object,
                    session_id=session_id,
                    source_message_id=msg.id,
                )
                
                # Check for conflicting edges and invalidate
                await self._handle_edge_conflicts(
                    subject_entity.id,
                    fact.predicate,
                    session_id,
                )
                
                # Create edge
                edge = Edge(
                    source_id=subject_entity.id,
                    target_id=object_entity.id,
                    edge_type=self._predicate_to_edge_type(fact.predicate),
                    properties={"predicate": fact.predicate},
                    source_session_id=UUID(session_id) if self._is_valid_uuid(session_id) else None,
                    source_message_id=msg.id,
                    confidence=fact.confidence,
                )
                
                self._edges[edge.id] = edge
                self._session_edges[session_id].add(edge.id)
                
                # Update fact with entity references
                fact = fact.model_copy(
                    update={
                        "subject_entity_id": subject_entity.id,
                        "object_entity_id": object_entity.id,
                        "edge_id": edge.id,
                    }
                )
                self._facts[fact.id] = fact
                extracted_facts.append(fact)
        
        logger.debug(
            "Facts extracted and stored",
            fact_count=len(extracted_facts),
            session_id=session_id,
        )
        
        return extracted_facts

    async def _extract_facts(
        self,
        message: Message,
        session_id: str,
    ) -> list[Fact]:
        """
        Extract facts from a message.
        
        This is a simplified extraction. In production, this would
        use an LLM for more sophisticated extraction.
        """
        facts: list[Fact] = []
        content = message.content.lower()
        
        # Simple pattern-based extraction (placeholder for LLM extraction)
        # Pattern: "I [verb] [object]" or "User [verb] [object]"
        patterns = [
            # Location patterns
            ("live in", "lives_in"),
            ("moved to", "lives_in"),
            ("located in", "located_in"),
            ("based in", "located_in"),
            # Preference patterns
            ("prefer", "prefers"),
            ("like", "prefers"),
            ("love", "prefers"),
            ("hate", "dislikes"),
            ("dislike", "dislikes"),
            # Work patterns
            ("work at", "works_at"),
            ("work for", "works_at"),
            ("employed at", "works_at"),
            # Usage patterns
            ("use", "uses"),
            ("using", "uses"),
        ]
        
        for pattern, predicate in patterns:
            if pattern in content:
                # Extract subject and object (simplified)
                # In production, use NER and dependency parsing
                parts = content.split(pattern)
                if len(parts) >= 2:
                    subject = "User"  # Default subject
                    obj = parts[1].strip().split()[0] if parts[1].strip() else ""
                    
                    if obj:
                        fact = Fact(
                            subject=subject,
                            predicate=predicate,
                            object=obj.capitalize(),
                            raw_text=message.content,
                            confidence=0.7,  # Lower confidence for pattern matching
                            session_id=UUID(session_id) if self._is_valid_uuid(session_id) else None,
                            message_id=message.id,
                        )
                        facts.append(fact)
        
        return facts

    async def _get_or_create_entity(
        self,
        name: str,
        session_id: str,
        source_message_id: UUID | None = None,
    ) -> Entity:
        """Get an existing entity or create a new one."""
        # Check for existing entity with same name
        for entity in self._entities.values():
            if entity.name.lower() == name.lower():
                return entity
        
        # Create new entity
        entity = Entity(
            name=name,
            entity_type=self._infer_entity_type(name),
            source_session_id=UUID(session_id) if self._is_valid_uuid(session_id) else None,
            source_message_id=source_message_id,
        )
        
        self._entities[entity.id] = entity
        self._session_entities[session_id].add(entity.id)
        
        return entity

    async def _handle_edge_conflicts(
        self,
        subject_id: UUID,
        predicate: str,
        session_id: str,
    ) -> None:
        """
        Handle conflicting edges by invalidating old ones.
        
        For example, if user says "I moved to Berlin" and there's
        an existing edge "User lives_in Munich", invalidate the old edge.
        """
        # Predicates that should be unique per subject
        unique_predicates = {"lives_in", "works_at", "located_in"}
        
        if predicate not in unique_predicates:
            return
        
        edge_type = self._predicate_to_edge_type(predicate)
        
        # Find conflicting edges
        for edge_id, edge in list(self._edges.items()):
            if (
                edge.source_id == subject_id
                and edge.edge_type == edge_type
                and edge.is_valid
            ):
                # Invalidate the old edge
                invalidated_edge = edge.invalidate(
                    reason=f"Superseded by new {predicate} relationship",
                )
                self._edges[edge_id] = invalidated_edge
                
                logger.info(
                    "Edge invalidated due to conflict",
                    edge_id=str(edge_id),
                    predicate=predicate,
                )

    def _predicate_to_edge_type(self, predicate: str) -> EdgeType:
        """Convert a predicate string to an EdgeType."""
        mapping = {
            "lives_in": EdgeType.LIVES_IN,
            "located_in": EdgeType.LIVES_IN,
            "works_at": EdgeType.WORKS_AT,
            "prefers": EdgeType.PREFERS,
            "dislikes": EdgeType.DISLIKES,
            "uses": EdgeType.USES,
            "knows": EdgeType.KNOWS,
            "member_of": EdgeType.MEMBER_OF,
        }
        return mapping.get(predicate, EdgeType.RELATED_TO)

    def _infer_entity_type(self, name: str) -> EntityType:
        """Infer entity type from name (simplified)."""
        name_lower = name.lower()
        
        if name_lower in ("user", "i", "me"):
            return EntityType.PERSON
        elif any(loc in name_lower for loc in ("city", "country", "street")):
            return EntityType.LOCATION
        elif any(org in name_lower for org in ("inc", "corp", "company", "ltd")):
            return EntityType.ORGANIZATION
        else:
            return EntityType.CONCEPT

    def _is_valid_uuid(self, value: str) -> bool:
        """Check if a string is a valid UUID."""
        try:
            UUID(value)
            return True
        except (ValueError, TypeError):
            return False

    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """
        Retrieve relevant context from the knowledge graph.
        
        Args:
            query: Search query
            session_id: Optional session filter
            max_results: Maximum results to return
            
        Returns:
            Context dictionary with facts, entities, and relationships
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        memory_query = MemoryQuery(
            query=query,
            session_id=session_id,
            max_facts=max_results,
            max_entities=max_results,
        )
        
        if self._graphiti_client:
            # Use Graphiti for retrieval
            try:
                results = await self._graphiti_client.search(
                    query=query,
                    num_results=max_results,
                )
                
                # Convert Graphiti results to our format
                facts = []
                entities = []
                edges = []
                
                for result in results:
                    # Process Graphiti results
                    # This depends on Graphiti's result format
                    pass
                
                query_time = (time.time() - start_time) * 1000
                
                memory_result = MemoryResult(
                    query=memory_query,
                    facts=facts,
                    entities=entities,
                    edges=edges,
                    query_time_ms=query_time,
                )
                
                return memory_result.to_context_dict()
                
            except Exception as e:
                logger.error("Graphiti retrieval failed", error=str(e))
                # Fall back to in-memory
        
        # In-memory retrieval
        result = await self._retrieve_in_memory(memory_query)
        return result.to_context_dict()

    async def _retrieve_in_memory(self, query: MemoryQuery) -> MemoryResult:
        """Retrieve from in-memory storage."""
        start_time = time.time()
        
        # Simple keyword matching (placeholder for semantic search)
        query_terms = set(query.query.lower().split())
        
        # Find matching facts
        matching_facts: list[Fact] = []
        for fact in self._facts.values():
            fact_terms = set(fact.to_sentence().lower().split())
            if query_terms & fact_terms:  # Intersection
                matching_facts.append(fact)
        
        # Sort by confidence and limit
        matching_facts.sort(key=lambda f: f.confidence, reverse=True)
        matching_facts = matching_facts[:query.max_facts]
        
        # Get related entities
        entity_ids = set()
        for fact in matching_facts:
            if fact.subject_entity_id:
                entity_ids.add(fact.subject_entity_id)
            if fact.object_entity_id:
                entity_ids.add(fact.object_entity_id)
        
        matching_entities = [
            self._entities[eid]
            for eid in entity_ids
            if eid in self._entities
        ][:query.max_entities]
        
        # Get related edges (only valid ones)
        matching_edges = [
            edge for edge in self._edges.values()
            if edge.is_valid and (
                edge.source_id in entity_ids or edge.target_id in entity_ids
            )
        ][:query.max_edges]
        
        query_time = (time.time() - start_time) * 1000
        
        return MemoryResult(
            query=query,
            facts=matching_facts,
            entities=matching_entities,
            edges=matching_edges,
            query_time_ms=query_time,
            total_entities_searched=len(self._entities),
            total_edges_searched=len(self._edges),
        )

    async def invalidate_fact(
        self,
        fact_id: UUID,
        reason: str,
    ) -> bool:
        """
        Invalidate a fact and its associated edge.
        
        Args:
            fact_id: ID of the fact to invalidate
            reason: Reason for invalidation
            
        Returns:
            True if invalidation was successful
        """
        if fact_id not in self._facts:
            return False
        
        fact = self._facts[fact_id]
        
        # Invalidate the associated edge
        if fact.edge_id and fact.edge_id in self._edges:
            edge = self._edges[fact.edge_id]
            invalidated_edge = edge.invalidate(reason)
            self._edges[fact.edge_id] = invalidated_edge
        
        # Mark fact as invalid by setting valid_until
        invalidated_fact = fact.model_copy(
            update={"valid_until": datetime.utcnow()}
        )
        self._facts[fact_id] = invalidated_fact
        
        logger.info(
            "Fact invalidated",
            fact_id=str(fact_id),
            reason=reason,
        )
        
        return True

    async def get_entity_history(
        self,
        entity_id: UUID,
    ) -> list[Edge]:
        """
        Get the history of relationships for an entity.
        
        Includes both valid and invalidated edges to show
        how facts have changed over time.
        """
        history = [
            edge for edge in self._edges.values()
            if edge.source_id == entity_id or edge.target_id == entity_id
        ]
        
        # Sort by creation time
        history.sort(key=lambda e: e.created_at)
        
        return history

    async def clear_session(self, session_id: str) -> None:
        """Clear all data for a session."""
        # Remove entities
        if session_id in self._session_entities:
            for entity_id in self._session_entities[session_id]:
                self._entities.pop(entity_id, None)
            del self._session_entities[session_id]
        
        # Remove edges
        if session_id in self._session_edges:
            for edge_id in self._session_edges[session_id]:
                self._edges.pop(edge_id, None)
            del self._session_edges[session_id]
        
        # Remove facts
        session_uuid = UUID(session_id) if self._is_valid_uuid(session_id) else None
        if session_uuid:
            facts_to_remove = [
                fid for fid, fact in self._facts.items()
                if fact.session_id == session_uuid
            ]
            for fid in facts_to_remove:
                del self._facts[fid]
        
        logger.info("Session cleared", session_id=session_id)

    @property
    def stats(self) -> dict[str, int]:
        """Get memory statistics."""
        valid_edges = sum(1 for e in self._edges.values() if e.is_valid)
        return {
            "total_entities": len(self._entities),
            "total_edges": len(self._edges),
            "valid_edges": valid_edges,
            "invalidated_edges": len(self._edges) - valid_edges,
            "total_facts": len(self._facts),
            "sessions": len(self._session_entities),
        }