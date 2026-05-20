from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Default location: ``serving/landmarks/`` next to this file. Override
# via the ``LANDMARKS_DIR`` env var so tests / Docker can point elsewhere.
_DEFAULT_DIR = Path(__file__).resolve().parent / "landmarks"
LANDMARKS_DIR = Path(os.getenv("LANDMARKS_DIR") or _DEFAULT_DIR)


@dataclass(frozen=True)
class MatchRule:
    require_label: str
    min_confidence: float
    min_count: int

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MatchRule":
        return cls(
            require_label=str(raw.get("require_label", "landmark")).lower(),
            min_confidence=float(raw.get("min_confidence", 0.5)),
            min_count=int(raw.get("min_count", 1)),
        )


@dataclass
class LandmarkEntry:
    """One authored landmark inside a facility's JSON file."""
    space_id_or_name: str
    match: MatchRule
    display_name: Optional[str] = None
    building_id: Optional[str] = None
    building_name: Optional[str] = None
    campus_id: Optional[str] = None
    floor_id: Optional[str] = None
    floor_index: Optional[int] = None

    # Resolved at enrichment time. Falls back to ``space_id_or_name`` when
    # the user already authored an explicit Space.id.
    resolved_space_id: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LandmarkEntry":
        return cls(
            space_id_or_name=str(raw["space_id_or_name"]),
            match=MatchRule.from_dict(raw.get("match") or {}),
            display_name=raw.get("display_name"),
            building_id=raw.get("building_id"),
            building_name=raw.get("building_name"),
            campus_id=raw.get("campus_id"),
            floor_id=raw.get("floor_id"),
            floor_index=raw.get("floor_index"),
        )


class LandmarkStore:
    """Thread-safe per-facility landmark lookup with optional Neo4j enrichment."""

    def __init__(
        self,
        *,
        landmarks_dir: Path = LANDMARKS_DIR,
        neo4j_driver: Any = None,
    ) -> None:
        self._dir = Path(landmarks_dir)
        self._driver = neo4j_driver
        self._lock = threading.Lock()
        self._cache: Dict[str, List[LandmarkEntry]] = {}


    def _path_for(self, facility_id: str) -> Path:
        safe = "".join(ch for ch in facility_id if ch.isalnum() or ch in "-_.")
        return self._dir / f"{safe}.json"

    def _load_raw(self, facility_id: str) -> List[LandmarkEntry]:
        path = self._path_for(facility_id)
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[landmarks_store] failed to read {path}: {exc}", flush=True)
            return []
        entries_raw = data.get("landmarks") or []
        entries: List[LandmarkEntry] = []
        for raw in entries_raw:
            try:
                entries.append(LandmarkEntry.from_dict(raw))
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[landmarks_store] skipping malformed entry in {path}: {exc}", flush=True)
        return entries

    def _enrich(self, entry: LandmarkEntry) -> None:
        """Fill in missing fields by walking the graph from the matched Space."""
        if self._driver is None:
            entry.resolved_space_id = entry.space_id_or_name
            return

        cypher = (
            "MATCH (s:Space) "
            "WHERE s.id = $key OR toLower(s.display_name) = toLower($key) "
            "OPTIONAL MATCH (s)<-[:HAS_SPACE]-(f:Floor) "
            "OPTIONAL MATCH (f)<-[:HAS_FLOOR]-(b:Building) "
            "OPTIONAL MATCH (b)<-[:HAS_BUILDING]-(c:Campus) "
            "RETURN s.id           AS space_id, "
            "       s.display_name AS display_name, "
            "       f.id           AS floor_id, "
            "       f.floor_index  AS floor_index, "
            "       b.id           AS building_id, "
            "       b.display_name AS building_name, "
            "       c.id           AS campus_id "
            "LIMIT 1"
        )
        try:
            with self._driver.session() as session:
                row = session.run(cypher, key=entry.space_id_or_name).single()
        except Exception as exc:
            print(f"[landmarks_store] Neo4j enrichment failed for {entry.space_id_or_name!r}: {exc}", flush=True)
            entry.resolved_space_id = entry.space_id_or_name
            return

        if row is None:
            print(f"[landmarks_store] no Space matched {entry.space_id_or_name!r} — using author values only", flush=True)
            entry.resolved_space_id = entry.space_id_or_name
            return

        entry.resolved_space_id = str(row.get("space_id") or entry.space_id_or_name)
        entry.display_name = entry.display_name or row.get("display_name")
        entry.floor_id = entry.floor_id or row.get("floor_id")
        entry.floor_index = entry.floor_index if entry.floor_index is not None else row.get("floor_index")
        entry.building_id = entry.building_id or row.get("building_id")
        entry.building_name = entry.building_name or row.get("building_name")
        entry.campus_id = entry.campus_id or row.get("campus_id")

    def entries_for(self, facility_id: str) -> List[LandmarkEntry]:
        with self._lock:
            cached = self._cache.get(facility_id)
            if cached is not None:
                return cached
            entries = self._load_raw(facility_id)
            for entry in entries:
                self._enrich(entry)
            self._cache[facility_id] = entries
            if entries:
                print(
                    f"[landmarks_store] loaded {len(entries)} landmark(s) for "
                    f"facility={facility_id!r}: "
                    + ", ".join(
                        f"{e.display_name or e.space_id_or_name}@{e.resolved_space_id}"
                        for e in entries
                    ),
                    flush=True,
                )
            else:
                print(f"[landmarks_store] no landmarks file for facility={facility_id!r}", flush=True)
            return entries


    def find_match(
        self,
        *,
        facility_id: str,
        detections: Sequence[Any],
    ) -> Optional["LandmarkMatch"]:
        entries = self.entries_for(facility_id)
        if not entries or not detections:
            return None

        best: Optional[LandmarkMatch] = None
        for entry in entries:
            rule = entry.match
            hit_indices: List[int] = []
            best_conf = 0.0
            for i, d in enumerate(detections):
                if (
                    str(getattr(d, "label", "")).lower() == rule.require_label
                    and float(getattr(d, "confidence", 0.0)) >= rule.min_confidence
                ):
                    hit_indices.append(i)
                    conf = float(getattr(d, "confidence", 0.0))
                    if conf > best_conf:
                        best_conf = conf
            if len(hit_indices) < rule.min_count:
                continue
            if best is None or best_conf > best.confidence:
                best = LandmarkMatch(
                    entry=entry,
                    confidence=best_conf,
                    supporting_count=len(hit_indices),
                    supporting_indices=hit_indices,
                )
        return best


@dataclass
class LandmarkMatch:
    entry: LandmarkEntry
    confidence: float
    supporting_count: int
    supporting_indices: List[int]
