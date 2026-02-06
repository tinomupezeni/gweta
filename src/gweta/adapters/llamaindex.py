"""LlamaIndex adapter for Gweta.

This module provides conversion between LlamaIndex
Node objects and Gweta Chunk objects.
"""

from typing import Any

from gweta.core.types import Chunk


class LlamaIndexAdapter:
    """Adapter for LlamaIndex Node objects.

    Converts between LlamaIndex TextNodes and Gweta Chunks,
    preserving metadata and quality information.

    Example:
        >>> adapter = LlamaIndexAdapter()
        >>> chunks = adapter.from_nodes(llamaindex_nodes)
        >>> nodes = adapter.to_nodes(gweta_chunks)
    """

    @staticmethod
    def from_nodes(nodes: list[Any]) -> list[Chunk]:
        """Convert LlamaIndex Nodes to Gweta Chunks.

        Args:
            nodes: List of LlamaIndex Node objects

        Returns:
            List of Gweta Chunk objects
        """
        chunks = []
        for node in nodes:
            metadata = dict(node.metadata) if hasattr(node, "metadata") else {}
            source = metadata.pop("source", metadata.pop("file_path", ""))

            text = ""
            if hasattr(node, "text"):
                text = node.text
            elif hasattr(node, "get_content"):
                text = node.get_content()

            chunks.append(
                Chunk(
                    id=node.node_id if hasattr(node, "node_id") else None,
                    text=text,
                    metadata=metadata,
                    source=source,
                )
            )
        return chunks

    @staticmethod
    def to_nodes(chunks: list[Chunk]) -> list[Any]:
        """Convert Gweta Chunks to LlamaIndex Nodes.

        Args:
            chunks: List of Gweta Chunk objects

        Returns:
            List of LlamaIndex TextNode objects
        """
        try:
            from llama_index.core.schema import TextNode
        except ImportError:
            try:
                from llama_index.schema import TextNode
            except ImportError as e:
                raise ImportError(
                    "LlamaIndex is required for this adapter. "
                    "Install it with: pip install llama-index-core"
                ) from e

        nodes = []
        for chunk in chunks:
            metadata = {
                **chunk.metadata,
                "source": chunk.source,
            }
            if chunk.quality_score is not None:
                metadata["quality_score"] = chunk.quality_score

            nodes.append(
                TextNode(
                    id_=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                )
            )
        return nodes

    @staticmethod
    def wrap_validator(validator: Any) -> Any:
        """Create a LlamaIndex-compatible validator wrapper.

        Args:
            validator: Gweta ChunkValidator

        Returns:
            Callable that validates LlamaIndex nodes
        """
        adapter = LlamaIndexAdapter()

        def validate_nodes(nodes: list[Any]) -> list[Any]:
            chunks = adapter.from_nodes(nodes)
            report = validator.validate_batch(chunks)
            return adapter.to_nodes(report.accepted())

        return validate_nodes
