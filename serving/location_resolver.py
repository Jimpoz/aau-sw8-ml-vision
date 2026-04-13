from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


class LocationEntity(BaseModel):
    kind: str = Field(description="e.g. 'room', 'landmark', 'corridor'")
    id: str
    name: str
    confidence: float = 0.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class RouteStep(BaseModel):
    from_location_id: str
    to_location_id: str
    instruction: str = Field(description="Human-readable instruction for navigation")


class ResolvedLocation(BaseModel):
    facility_id: str
    current_location: LocationEntity
    route: Optional[List[RouteStep]] = None
    debug: Dict[str, Any] = Field(default_factory=dict)


class SpatialBackend:
    """
    Spatial backend façade.

    In the full system this is where you would:
    - resolve detection labels against Neo4j
    - enrich with spatial middleware (graph, geometry, routes)
    - return a single contextualized response
    """

    def __init__(self, *, neo4j_uri: str = "", neo4j_user: str = "", neo4j_password: str = "") -> None:
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        self._neo4j_available = bool(neo4j_uri and neo4j_user and neo4j_password)
        self._driver = None

        if self._neo4j_available:
            try:
                from neo4j import GraphDatabase  # local import to keep optional

                self._driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            except Exception:
                self._neo4j_available = False
                self._driver = None

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def resolve_current_location(
        self,
        *,
        facility_id: str,
        detections: Sequence[Any],
        label_hint: Optional[str] = None,
    ) -> Tuple[LocationEntity, Dict[str, Any]]:
        """
        Best-effort resolution:
        - If Neo4j is configured, attempt a lookup by label_hint / detection label.
        - Otherwise use deterministic heuristics.
        """
        debug: Dict[str, Any] = {"neo4j_enabled": self._neo4j_available}

        if self._neo4j_available and self._driver is not None and len(detections) > 0:
            # NOTE: This query is intentionally a template because the Neo4j schema
            # for this project may differ. It provides a safe integration seam.
            #
            # Expected model:
            # - (l:Landmark { facility_id, name })
            # - (r:Room { facility_id, room_type })
            try:
                # Choose top confidence detection as label hint.
                best = max(detections, key=lambda d: getattr(d, "confidence", 0.0))
                detection_label = str(getattr(best, "label", ""))

                cypher = """
                MATCH (n)
                WHERE n.facility_id = $facility_id AND (
                  toLower(n.name) = toLower($label) OR
                  toLower(n.room_type) = toLower($label)
                )
                RETURN labels(n)[0] AS kind, n.id AS id, coalesce(n.name, n.room_type) AS name, 1.0 AS confidence, properties(n) AS properties
                LIMIT 1
                """
                with self._driver.session() as session:
                    res = session.run(
                        cypher,
                        facility_id=facility_id,
                        label=label_hint or detection_label,
                    )
                    r0 = res.single()
                if r0 is not None:
                    entity = LocationEntity(
                        kind=str(r0.get("kind") or "unknown"),
                        id=str(r0.get("id") or f"{facility_id}:{label_hint or detection_label}"),
                        name=str(r0.get("name") or label_hint or detection_label),
                        confidence=float(r0.get("confidence") or 1.0),
                        properties=dict(r0.get("properties") or {}),
                    )
                    return entity, {**debug, "neo4j_resolution": "matched"}
            except Exception as e:
                debug["neo4j_error"] = str(e)

        # Heuristic fallback: pick top detection and tag kind based on label.
        if len(detections) == 0:
            entity = LocationEntity(
                kind="unknown",
                id=f"{facility_id}:unknown",
                name="unknown",
                confidence=0.0,
            )
            return entity, {**debug, "fallback": "no_detections"}

        best = max(detections, key=lambda d: getattr(d, "confidence", 0.0))
        label = str(getattr(best, "label", "unknown"))
        conf = float(getattr(best, "confidence", 0.0))
        kind = "room" if "room" in label.lower() else ("landmark" if "landmark" in label.lower() else "unknown")
        entity = LocationEntity(
            kind=kind,
            id=f"{facility_id}:{label}",
            name=label,
            confidence=conf,
        )
        return entity, {**debug, "fallback": "heuristic"}

    def plan_route(
        self,
        *,
        facility_id: str,
        from_location: LocationEntity,
        navigation_to: Optional[str],
    ) -> Tuple[Optional[List[RouteStep]], Dict[str, Any]]:
        debug: Dict[str, Any] = {"route_planning": navigation_to is not None}
        if not navigation_to:
            return None, debug

        # Heuristic route: if the system knows no graph, return a simple step.
        to_id = f"{facility_id}:{navigation_to}"
        route = [
            RouteStep(
                from_location_id=from_location.id,
                to_location_id=to_id,
                instruction=f"Navigate to {navigation_to}",
            )
        ]
        return route, {**debug, "fallback": True}


class LocationResolver:
    def __init__(self) -> None:
        self._backend = SpatialBackend(
            neo4j_uri=os.getenv("NEO4J_URI", ""),
            neo4j_user=os.getenv("NEO4J_USER", ""),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        )

    def resolve(
        self,
        *,
        facility_id: str,
        detections: Sequence[Any],
        navigation_to: Optional[str] = None,
    ) -> ResolvedLocation:
        current, dbg0 = self._backend.resolve_current_location(
            facility_id=facility_id,
            detections=detections,
            label_hint=None,
        )
        route, dbg1 = self._backend.plan_route(
            facility_id=facility_id,
            from_location=current,
            navigation_to=navigation_to,
        )
        return ResolvedLocation(
            facility_id=facility_id,
            current_location=current,
            route=route,
            debug={**dbg0, **dbg1},
        )
