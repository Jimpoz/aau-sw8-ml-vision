from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


_RATIO_TEST = _env_float("ORB_RATIO_TEST", 0.75)

_MIN_GOOD_MATCHES = _env_int("ORB_MIN_GOOD_MATCHES", 8)

_ORB_FEATURES = _env_int("ORB_FEATURES", 1000)

_MIN_INLIERS = _env_int("ORB_MIN_INLIERS", 12)

_RANSAC_REPROJ_PX = _env_float("ORB_RANSAC_REPROJ_PX", 5.0)

_GLOBAL_FALLBACK = (os.getenv("ORB_GLOBAL_FALLBACK", "true").strip().lower()
                    in ("1", "true", "yes", "on"))


def _try_import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except ImportError:
        return None


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except ImportError:
        return None


_cv2 = _try_import_cv2()
_np = _try_import_numpy()


class Neo4jLandmarkSource:
    """Pulls Landmark records straight from Neo4j by ``campus_id``."""

    def __init__(self, driver: Any) -> None:
        self._driver = driver

    _RETURN = (
        "RETURN l.id           AS id, "
        "       l.name         AS name, "
        "       l.space_id     AS space_id, "
        "       l.building_id  AS building_id, "
        "       l.campus_id    AS campus_id, "
        "       l.image_b64    AS image_b64, "
        "       l.image_width  AS image_width, "
        "       l.image_height AS image_height"
    )

    def fetch(self, facility_id: str) -> List[Dict[str, Any]]:
        if self._driver is None:
            return []
        cypher = "MATCH (l:Landmark {campus_id: $campus_id}) " + self._RETURN
        with self._driver.session() as session:
            rows = session.run(cypher, campus_id=facility_id)
            return [dict(r) for r in rows]

    def fetch_all(self) -> List[Dict[str, Any]]:
        if self._driver is None:
            return []
        cypher = "MATCH (l:Landmark) " + self._RETURN
        with self._driver.session() as session:
            rows = session.run(cypher)
            return [dict(r) for r in rows]


class PostgresLandmarkSource:
    """Pulls Landmark records from the PostGIS ``landmarks`` mirror by
    ``campus_id``. Used as a fallback when Neo4j is unreachable."""

    def __init__(self, dsn: Optional[str]) -> None:
        # Normalise a SQLAlchemy-style URL (postgresql+psycopg2://) down
        # to what psycopg2.connect accepts (postgresql://).
        if dsn:
            dsn = dsn.replace("postgresql+psycopg2://", "postgresql://")
        self._dsn = dsn or None

    def fetch(self, facility_id: str) -> List[Dict[str, Any]]:
        if not self._dsn:
            return []
        try:
            import psycopg2  # type: ignore
            import psycopg2.extras  # type: ignore
        except ImportError:
            print(
                "[orb_matcher] psycopg2 not importable — PostGIS landmark "
                "fallback disabled. Add psycopg2-binary to requirements.",
                flush=True,
            )
            return []

        conn = None
        try:
            conn = psycopg2.connect(self._dsn, connect_timeout=5)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Satisfy the org-isolation RLS policy as a service read.
                cur.execute("SET app.is_service = 'true'")
                cur.execute(
                    "SELECT id, name, space_id, building_id, campus_id, "
                    "       image_b64, image_width, image_height "
                    "FROM landmarks WHERE campus_id = %s",
                    (facility_id,),
                )
                return [dict(r) for r in cur.fetchall()]
        except Exception as exc:
            print(f"[orb_matcher] PostGIS fallback fetch({facility_id!r}) failed: {exc}", flush=True)
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def fetch_all(self) -> List[Dict[str, Any]]:
        if not self._dsn:
            return []
        try:
            import psycopg2  # type: ignore
            import psycopg2.extras  # type: ignore
        except ImportError:
            return []
        conn = None
        try:
            conn = psycopg2.connect(self._dsn, connect_timeout=5)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SET app.is_service = 'true'")
                cur.execute(
                    "SELECT id, name, space_id, building_id, campus_id, "
                    "       image_b64, image_width, image_height "
                    "FROM landmarks"
                )
                return [dict(r) for r in cur.fetchall()]
        except Exception as exc:
            print(f"[orb_matcher] PostGIS fallback fetch_all failed: {exc}", flush=True)
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


class FallbackLandmarkSource:
    """Tries each wrapped source in order and returns the first
    non-empty result."""

    def __init__(self, sources: Sequence[Any]) -> None:
        self._sources = list(sources)

    def fetch(self, facility_id: str) -> List[Dict[str, Any]]:
        for source in self._sources:
            try:
                records = source.fetch(facility_id)
            except Exception as exc:
                print(
                    f"[orb_matcher] source {type(source).__name__} "
                    f"raised for {facility_id!r}: {exc}",
                    flush=True,
                )
                continue
            if records:
                return records
        return []

    def fetch_all(self) -> List[Dict[str, Any]]:
        for source in self._sources:
            if not hasattr(source, "fetch_all"):
                continue
            try:
                records = source.fetch_all()
            except Exception as exc:
                print(
                    f"[orb_matcher] source {type(source).__name__} "
                    f"fetch_all raised: {exc}",
                    flush=True,
                )
                continue
            if records:
                return records
        return []


@dataclass
class CachedLandmark:
    """In-memory representation of a registered landmark after the
    reference image has been decoded and ORB-extracted."""
    id: str
    name: str
    space_id: str
    building_id: Optional[str]
    campus_id: Optional[str]
    descriptors: Any  # numpy.ndarray
    keypoint_count: int
    keypoints: Any = None  # numpy.ndarray (N, 2)


@dataclass
class OrbMatch:
    """Result of matching the live frame against the per-facility
    landmark cache."""
    landmark: CachedLandmark
    good_matches: int
    elapsed_ms: float
    bbox_norm: Optional[Tuple[float, float, float, float]] = None
    debug: Dict[str, Any] = field(default_factory=dict)


class OrbLandmarkMatcher:
    """Thread-safe per-facility landmark recogniser."""

    def __init__(self, *, source: Any) -> None:
        """``source`` is anything with a ``fetch(facility_id) -> list[dict]``
        method that returns the per-facility landmark records (id, name,
        space_id, building_id, campus_id, image_b64, image_width,
        image_height). See OrbLandmarkSource for the production impl."""
        self._source = source
        self._lock = threading.Lock()
        self._cache: Dict[str, List[CachedLandmark]] = {}
        self._available = _cv2 is not None and _np is not None
        if not self._available:
            print(
                "[orb_matcher] cv2 or numpy not importable — ORB matching disabled. "
                "Install opencv-python-headless to enable.",
                flush=True,
            )

    # Cache loading

    def _load_facility(self, facility_id: str) -> List[CachedLandmark]:
        if not self._available:
            return []
        try:
            records = self._source.fetch(facility_id)
        except Exception as exc:
            print(f"[orb_matcher] source.fetch({facility_id!r}) failed: {exc}", flush=True)
            return []

        if not records and _GLOBAL_FALLBACK and hasattr(self._source, "fetch_all"):
            try:
                records = self._source.fetch_all()
            except Exception as exc:
                print(f"[orb_matcher] source.fetch_all() failed: {exc}", flush=True)
                records = []
            if records:
                print(
                    f"[orb_matcher] facility={facility_id!r}: no facility-scoped "
                    f"landmarks; falling back to ALL {len(records)} registered "
                    f"landmark(s)",
                    flush=True,
                )

        orb = _cv2.ORB_create(nfeatures=_ORB_FEATURES)
        out: List[CachedLandmark] = []
        for record in records:
            image_b64 = record.get("image_b64")
            if not image_b64:
                continue
            try:
                image_bytes = base64.b64decode(image_b64)
                buf = _np.frombuffer(image_bytes, dtype=_np.uint8)
                img = _cv2.imdecode(buf, _cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"[orb_matcher] could not decode landmark {record.get('id')!r}", flush=True)
                    continue
                keypoints, descriptors = orb.detectAndCompute(img, None)
            except Exception as exc:
                print(f"[orb_matcher] ORB extraction failed for landmark {record.get('id')!r}: {exc}", flush=True)
                continue

            if descriptors is None or len(descriptors) == 0:
                print(f"[orb_matcher] landmark {record.get('id')!r} produced no descriptors", flush=True)
                continue

            ref_pts = _np.float32([kp.pt for kp in keypoints]) if keypoints else None
            out.append(CachedLandmark(
                id=str(record["id"]),
                name=str(record.get("name") or record["id"]),
                space_id=str(record["space_id"]),
                building_id=record.get("building_id"),
                campus_id=record.get("campus_id"),
                descriptors=descriptors,
                keypoint_count=len(keypoints),
                keypoints=ref_pts,
            ))

        print(
            f"[orb_matcher] facility={facility_id!r}: "
            f"loaded {len(out)} landmark(s) "
            + ", ".join(f"{c.name}@{c.space_id} ({c.keypoint_count} kp)" for c in out),
            flush=True,
        )
        return out

    def entries_for(self, facility_id: str) -> List[CachedLandmark]:
        with self._lock:
            cached = self._cache.get(facility_id)
            if cached is not None:
                return cached
            entries = self._load_facility(facility_id)
            self._cache[facility_id] = entries
            return entries

    def invalidate(self, facility_id: Optional[str] = None) -> None:
        """Drop the cache so the next match rebuilds from the source.
        Call after a user registers / deletes a landmark."""
        with self._lock:
            if facility_id is None:
                self._cache.clear()
            else:
                self._cache.pop(facility_id, None)

    def match(self, *, facility_id: str, image_bytes: bytes) -> Optional[OrbMatch]:
        if not self._available:
            return None
        entries = self.entries_for(facility_id)
        if not entries:
            return None

        t0 = time.perf_counter()
        try:
            buf = _np.frombuffer(image_bytes, dtype=_np.uint8)
            frame = _cv2.imdecode(buf, _cv2.IMREAD_GRAYSCALE)
            if frame is None:
                return None
            orb = _cv2.ORB_create(nfeatures=_ORB_FEATURES)
            frame_keypoints, frame_descriptors = orb.detectAndCompute(frame, None)
        except Exception as exc:
            print(f"[orb_matcher] frame extraction failed: {exc}", flush=True)
            return None

        if frame_descriptors is None or len(frame_descriptors) == 0:
            return None

        frame_h, frame_w = frame.shape[:2]

        bf = _cv2.BFMatcher(_cv2.NORM_HAMMING, crossCheck=False)
        best: Optional[CachedLandmark] = None
        best_score = 0
        best_good: List[Any] = []  # cv2.DMatch list for the winner
        per_landmark_scores: Dict[str, int] = {}

        for entry in entries:
            try:
                pairs = bf.knnMatch(frame_descriptors, entry.descriptors, k=2)
            except Exception:
                continue
            good: List[Any] = []
            for pair in pairs:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < _RATIO_TEST * n.distance:
                    good.append(m)
            per_landmark_scores[entry.id] = len(good)
            if len(good) > best_score:
                best_score = len(good)
                best = entry
                best_good = good

        if best is None or best_score < _MIN_GOOD_MATCHES:
            if best is not None and best_score > 0 and best_score % 3 == 0:
                print(
                    f"[orb_matcher] near-miss facility={facility_id!r} "
                    f"best={best.name!r}@{best.space_id} good={best_score} "
                    f"min_good={_MIN_GOOD_MATCHES} "
                    f"frame_kp={len(frame_descriptors)} "
                    f"all_scores={per_landmark_scores}",
                    flush=True,
                )
            return None

        inliers = 0
        inlier_pts: List[Any] = []
        if best.keypoints is not None and len(best_good) >= 4:
            try:
                src = _np.float32(
                    [best.keypoints[m.trainIdx] for m in best_good
                     if m.trainIdx < len(best.keypoints)]
                ).reshape(-1, 1, 2)
                dst = _np.float32(
                    [frame_keypoints[m.queryIdx].pt for m in best_good
                     if m.queryIdx < len(frame_keypoints)]
                ).reshape(-1, 1, 2)
                if len(src) >= 4 and len(src) == len(dst):
                    _h, mask = _cv2.findHomography(
                        src, dst, _cv2.RANSAC, _RANSAC_REPROJ_PX
                    )
                    if mask is not None:
                        inliers = int(mask.sum())
                        inlier_pts = [dst[i][0] for i in range(len(dst)) if mask[i][0]]
            except Exception as exc:
                print(f"[orb_matcher] homography failed: {exc}", flush=True)
                inliers = 0
        else:
            inliers = best_score

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if inliers < _MIN_INLIERS:
            print(
                f"[orb_matcher] rejected facility={facility_id!r} "
                f"best={best.name!r}@{best.space_id} good={best_score} "
                f"inliers={inliers} min_inliers={_MIN_INLIERS} (failed geometry check)",
                flush=True,
            )
            return None

        bbox_norm: Optional[Tuple[float, float, float, float]] = None
        pts = inlier_pts or [
            frame_keypoints[m.queryIdx].pt for m in best_good
            if m.queryIdx < len(frame_keypoints)
        ]
        if pts and frame_w > 0 and frame_h > 0:
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            pad_x = (max(xs) - min(xs)) * 0.10 + 8.0
            pad_y = (max(ys) - min(ys)) * 0.10 + 8.0
            x1 = max(0.0, (min(xs) - pad_x) / frame_w)
            y1 = max(0.0, (min(ys) - pad_y) / frame_h)
            x2 = min(1.0, (max(xs) + pad_x) / frame_w)
            y2 = min(1.0, (max(ys) + pad_y) / frame_h)
            if x2 > x1 and y2 > y1:
                bbox_norm = (x1, y1, x2, y2)

        print(
            f"[orb_matcher] HIT facility={facility_id!r} "
            f"landmark={best.name!r}@{best.space_id} "
            f"good={best_score} inliers={inliers} "
            f"elapsed_ms={elapsed_ms:.1f} bbox={bbox_norm}",
            flush=True,
        )
        return OrbMatch(
            landmark=best,
            good_matches=inliers,
            elapsed_ms=elapsed_ms,
            bbox_norm=bbox_norm,
            debug={
                "frame_keypoints": int(len(frame_descriptors)),
                "good_matches": best_score,
                "inliers": inliers,
                "per_landmark_scores": per_landmark_scores,
                "ratio_test": _RATIO_TEST,
                "min_good_matches": _MIN_GOOD_MATCHES,
                "min_inliers": _MIN_INLIERS,
            },
        )
