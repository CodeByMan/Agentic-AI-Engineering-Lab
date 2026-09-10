import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class FactEntry:
    entity: str
    attribute: str
    value: Any
    source: str = "conversation"
    confidence: float = 1.0
    updated_at: float = field(default_factory=time.time)


class LongTermMemoryStore:
    """Structured store for semantic long-term-memory facts."""

    def __init__(self, storage_path: Optional[str] = None):
        # Structure: {entity: {attribute: FactEntry}}.
        self._store: Dict[str, Dict[str, FactEntry]] = {}
        self._storage_path = Path(storage_path) if storage_path else None
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Invalid memory storage format")
            for entity, attributes in payload.items():
                if not isinstance(entity, str) or not isinstance(attributes, dict):
                    raise ValueError("Invalid memory storage format")
                self._store[entity] = {}
                for attribute, raw_entry in attributes.items():
                    if not isinstance(attribute, str) or not isinstance(raw_entry, dict):
                        raise ValueError("Invalid memory fact format")
                    confidence = float(raw_entry.get("confidence", 1.0))
                    if not 0.0 <= confidence <= 1.0:
                        raise ValueError("Invalid memory confidence")
                    self._store[entity][attribute] = FactEntry(
                        entity=entity,
                        attribute=attribute,
                        value=raw_entry["value"],
                        source=raw_entry.get("source", "conversation"),
                        confidence=confidence,
                        updated_at=float(raw_entry.get("updated_at", time.time())),
                    )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Memory store is unreadable: {self._storage_path}") from exc

    def _persist(self) -> None:
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            entity: {
                attribute: {
                    "value": entry.value,
                    "source": entry.source,
                    "confidence": entry.confidence,
                    "updated_at": entry.updated_at,
                }
                for attribute, entry in attributes.items()
            }
            for entity, attributes in self._store.items()
        }
        temporary_path = self._storage_path.with_name(f"{self._storage_path.name}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary_path, self._storage_path)
        try:
            self._storage_path.chmod(0o600)
        except OSError:
            # Windows ACLs are managed by the host environment.
            pass

    def upsert_fact(
        self,
        entity: str,
        attribute: str,
        value: Any,
        source: str = "conversation",
        confidence: float = 1.0,
    ):
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError("entity must be a non-empty string")
        if not isinstance(attribute, str) or not attribute.strip():
            raise ValueError("attribute must be a non-empty string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("source must be a non-empty string")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        try:
            json.dumps(value)
        except TypeError as exc:
            raise TypeError("memory values must be JSON serializable") from exc

        entity_key = entity.lower().strip()
        attr_key = attribute.lower().strip()

        with self._lock:
            if entity_key not in self._store:
                self._store[entity_key] = {}

            # Insert or update the fact as a simple conflict-resolution policy.
            self._store[entity_key][attr_key] = FactEntry(
                entity=entity_key,
                attribute=attr_key,
                value=value,
                source=source.strip(),
                confidence=confidence,
            )
            self._persist()
        print(
            "[LTM CONSOLIDATED] Fact updated "
            f"(entity_type={type(entity_key).__name__}, "
            f"attribute_type={type(attr_key).__name__}, "
            f"value_type={type(value).__name__})"
        )

    def query_entity(
        self, entity: str, include_metadata: bool = False
    ) -> Optional[Dict[str, Any]]:
        entity_key = entity.lower().strip()
        with self._lock:
            if entity_key in self._store:
                if include_metadata:
                    return {
                        attr: {
                            "value": entry.value,
                            "source": entry.source,
                            "confidence": entry.confidence,
                            "updated_at": entry.updated_at,
                        }
                        for attr, entry in self._store[entity_key].items()
                    }
                return {attr: entry.value for attr, entry in self._store[entity_key].items()}
        return None


class MemoryConsolidationWorker:
    """Background-style processor that promotes facts from STM to LTM."""

    def __init__(self, ltm: LongTermMemoryStore):
        self.ltm = ltm

    def process_short_term_buffer(self, conversation_turn: Dict[str, str]):
        """Simulate structured fact extraction by an LLM."""
        if not isinstance(conversation_turn, dict):
            raise TypeError("conversation_turn must be a dictionary")
        user_input = conversation_turn.get("user", "")
        if not isinstance(user_input, str):
            raise TypeError("conversation_turn['user'] must be a string")

        # Simulate extraction logic; production code would use a validated LLM schema.
        normalized_input = user_input.strip()
        lower_input = normalized_input.lower()
        if "my preferred language is" in lower_input:
            _, _, lang = normalized_input.partition("is")
            if lang.strip():
                self.ltm.upsert_fact(
                    entity="user",
                    attribute="preferred_language",
                    value=lang.strip(),
                    source="conversation",
                    confidence=0.95,
                )

        elif "i moved to" in lower_input:
            _, _, city = normalized_input.partition("to")
            if city.strip():
                self.ltm.upsert_fact(
                    entity="user",
                    attribute="location",
                    value=city.strip(),
                    source="conversation",
                    confidence=0.95,
                )


# --- Performance demo ---
if __name__ == "__main__":
    ltm_store = LongTermMemoryStore()
    worker = MemoryConsolidationWorker(ltm=ltm_store)

    # Interaction 1: the user states a preferred language.
    turn1 = {"user": "Hello, my preferred language is Python"}
    worker.process_short_term_buffer(turn1)

    # Interaction 2: the user states a Pakistan-focused location.
    turn2 = {"user": "I moved to Karachi"}
    worker.process_short_term_buffer(turn2)

    # Interaction 3: the user changes location to test conflict resolution.
    turn3 = {"user": "I moved to Lahore"}
    worker.process_short_term_buffer(turn3)

    # Query the consolidated memory.
    print("\n--- Current Semantic Long-Term Memory for 'user' ---")
    user_facts = ltm_store.query_entity("user")
    print(json.dumps(user_facts, indent=2))
