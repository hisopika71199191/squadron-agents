"""
Agent Skills Data Models

Defines the core data structures for Agent Skills following the
agentskills.io specification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Regex to strip punctuation for word-level matching
_WORD_CLEAN_RE = re.compile(r"[^a-z0-9]+")

# Common English stop-words excluded from semantic matching
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "of", "to", "and", "or",
    "for", "be", "as", "at", "by", "if", "do", "no", "so", "up", "we",
    "he", "she", "my", "any", "all", "not", "but", "are", "was", "has",
    "had", "this", "that", "with", "from", "they", "them", "will", "can",
    "its", "also", "what", "when", "how", "used", "like", "even", "both",
    "use", "need", "way", "may", "you", "your",
})


def _stem(word: str) -> str:
    """
    Very lightweight English stemmer.

    Handles the most common inflections without an NLP library:
    ``creating → creat``, ``presentations → presentation``,
    ``created → creat``, ``slides → slide``, ``create → creat``.
    """
    w = word
    if w.endswith("ations") and len(w) > 7:
        return w[:-1]  # presentations → presentation
    if w.endswith("ating") and len(w) > 6:
        return w[:-3]  # creating → creat  (via -ating)
    if w.endswith("ing") and len(w) > 5:
        return w[:-3]
    if w.endswith("tion") and len(w) > 5:
        return w  # keep as-is (presentation stays)
    if w.endswith("ed") and len(w) > 4:
        return w[:-2]
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    # Strip trailing 'e' for verb stems (create → creat)
    if w.endswith("e") and len(w) > 4:
        return w[:-1]
    return w


def _normalise_words(text: str, remove_stopwords: bool = False) -> set[str]:
    """
    Normalise a text string into a set of cleaned, lower-case tokens.

    Strips punctuation, quotes, and other non-alphanumeric characters so that
    ``"presentations;"`` and ``"presentation"`` share the same stem token.
    Optionally removes common stop-words.  Always adds stemmed variants.
    """
    tokens: set[str] = set()
    for raw in text.lower().split():
        cleaned = _WORD_CLEAN_RE.sub("", raw)
        if not cleaned:
            continue
        if remove_stopwords and cleaned in _STOP_WORDS:
            continue
        tokens.add(cleaned)
        stem = _stem(cleaned)
        if stem != cleaned:
            tokens.add(stem)
    return tokens


@dataclass
class SkillMetadata:
    """
    Level 1: Skill metadata from YAML frontmatter.
    
    Always loaded at startup for skill discovery.
    """
    name: str
    description: str
    short_description: str | None = None
    version: str | None = None
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    
    def matches_query(self, query: str) -> float:
        """
        Score how well this skill matches a query.
        
        Uses normalised word matching with simple stemming so that
        punctuation-laden descriptions (e.g. ``"presentations;"``)
        still match query words like ``"presentation"``.
        
        Scoring strategy:
        - 0.4 for skill name appearing in the query
        - 0.4 for content-word overlap (query-coverage oriented,
          counting at the *concept* level — each original query word
          counts once even if it expands to multiple stem forms)
        - 0.2 for tag match
        
        Returns a relevance score between 0.0 and 1.0.
        """
        query_lower = query.lower()
        score = 0.0
        
        # Check name match (exact substring)
        if self.name.lower() in query_lower:
            score += 0.4
        
        # Build description token set (expanded with stems)
        desc_tokens = _normalise_words(self.description, remove_stopwords=True)
        
        # For query, count *original concepts* — each cleaned word is one
        # concept.  A concept matches if its cleaned form OR its stem appears
        # in the description token set.
        query_concepts: list[str] = []
        matched_concepts = 0
        for raw in query_lower.split():
            cleaned = _WORD_CLEAN_RE.sub("", raw)
            if not cleaned or cleaned in _STOP_WORDS:
                continue
            query_concepts.append(cleaned)
            stem = _stem(cleaned)
            if cleaned in desc_tokens or stem in desc_tokens:
                matched_concepts += 1
        
        if query_concepts:
            query_coverage = matched_concepts / len(query_concepts)
            score += 0.4 * min(query_coverage, 1.0)
        
        # Check tags match
        query_tokens_set = _normalise_words(query, remove_stopwords=True)
        for tag in self.tags:
            tag_tokens = _normalise_words(tag, remove_stopwords=True)
            if tag_tokens & query_tokens_set:
                score += 0.2
                break
        
        return min(1.0, score)


@dataclass
class Skill:
    """
    Full Skill representation with all levels of content.
    
    Levels:
    - Level 1: Metadata (always loaded)
    - Level 2: Instructions (loaded when triggered)
    - Level 3: Resources (loaded as needed)
    """
    metadata: SkillMetadata
    path: Path
    instructions: str | None = None  # Level 2: SKILL.md body
    resources: dict[str, str] = field(default_factory=dict)  # Level 3: additional files
    
    @property
    def name(self) -> str:
        return self.metadata.name
    
    @property
    def description(self) -> str:
        return self.metadata.description
    
    @property
    def is_loaded(self) -> bool:
        """Check if Level 2 instructions have been loaded."""
        return self.instructions is not None
    
    def get_resource_paths(self) -> list[Path]:
        """Get paths to all resources in the skill directory."""
        if not self.path.exists():
            return []
        
        resources = []
        for item in self.path.iterdir():
            if item.name != "SKILL.md" and item.is_file():
                resources.append(item)
            elif item.is_dir():
                # Include files from subdirectories
                for subitem in item.rglob("*"):
                    if subitem.is_file():
                        resources.append(subitem)
        return resources
    
    def to_context_string(self, include_resources: bool = False) -> str:
        """
        Format skill for inclusion in LLM context.
        
        Args:
            include_resources: Whether to include Level 3 resources
        """
        parts = [
            f"## Skill: {self.name}",
            f"**Description**: {self.description}",
        ]
        
        if self.instructions:
            parts.append(f"\n### Instructions\n{self.instructions}")
        
        if include_resources and self.resources:
            parts.append("\n### Resources")
            for name, content in self.resources.items():
                parts.append(f"\n#### {name}\n```\n{content}\n```")
        
        return "\n".join(parts)


@dataclass
class SkillMatch:
    """A skill that matched a query with its relevance score."""
    skill: Skill
    score: float
    
    def __lt__(self, other: SkillMatch) -> bool:
        return self.score < other.score
