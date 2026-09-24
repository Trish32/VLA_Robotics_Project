"""A shared world model: what is in the scene, where, and how things relate.

Three stages produce facts about the world at wildly different rates, and every
downstream consumer currently has to re-derive the same structure from raw perception:

    SLAM             robot pose               30+ Hz
    FoundationPose   object 6-DoF pose        ~30 Hz for ONE object, ~6 Hz for five
    OpenMask3D       instance + open-vocab label   seconds per scene
    places / rooms   rarely, or never after mapping

This module holds those in one graph so planning, manipulation and navigation query
structured facts ("the cup is on the table", "where is the pear") instead of each
reprocessing point clouds.

WHY THIS IS NOT JUST TF2
------------------------
TF2 is a transform TREE: one parent per frame, time-indexed, interpolated. A scene graph
is not a tree and its edges are not transforms — `on`, `inside` and `near` have no
inverse transform and no interpolation. Conflating them fails in both directions: you
cannot express "the cup is on the table" in TF2, and you cannot ask TF2 for "everything
on the table".

So the split is deliberate and the boundary is one-way:

    TF2            owns live geometry. Frames, interpolation, the kinematic chain.
    scene graph    owns identity, semantics and relations. Every node names its TF
                   frame; consumers needing a live pose look it up there.

A node still caches a pose, because relations have to be computed from something — but
it is cached WITH the stamp and the frame it was expressed in, and it is never the
authority. `tf_frame` is.

ANCHORING, BECAUSE LOOP CLOSURE MOVES THE WORLD
-----------------------------------------------
DROID's backend bundle adjustment revises keyframe poses after the fact. An object pose
baked into world coordinates from an old camera pose is stale the moment the map is
optimised, and the object visibly drifts off the thing it is sitting on.

So a node's pose is stored relative to an `anchor_frame` — normally the keyframe it was
observed from, not `map`. When the optimiser moves that keyframe, TF2 republishes the
correction and every anchored object follows for free. This is what object-SLAM systems
do, and it is the difference between a map that heals and one that quietly decays.

STALENESS IS PER LAYER, NOT GLOBAL
----------------------------------
A single "is this fresh?" threshold cannot serve a 30 Hz robot pose and a semantic label
that is refreshed once a minute. Applied at the fast rate it discards every label;
applied at the slow rate it happily hands a planner a robot pose from two seconds ago.
Each node kind therefore carries its own tolerance, and queries refuse rather than return
a stale fact — the same contract the bridge already uses for `max_obs_age_s`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

NS_PER_S = 1_000_000_000


class NodeKind(str, Enum):
    ROBOT = "robot"        # the base; pose from SLAM
    OBJECT = "object"      # a manipulable instance; pose from FoundationPose
    SURFACE = "surface"    # something things rest ON: table, shelf, floor
    PLACE = "place"        # a region: room, workspace, staging area


# How old a fact of each kind may be before a query refuses it. These are ceilings on
# the producing stage's period, not guesses: a robot pose older than ~100 ms is a
# different place at walking speed, while a semantic label does not spoil.
DEFAULT_MAX_AGE_S: dict[NodeKind, float] = {
    NodeKind.ROBOT: 0.15,
    NodeKind.OBJECT: 0.50,
    NodeKind.SURFACE: 30.0,
    NodeKind.PLACE: float("inf"),
}


class Predicate(str, Enum):
    ON = "on"                 # resting on top of, in contact
    INSIDE = "inside"         # contained by
    NEAR = "near"             # within reach-ish distance, no contact implied
    PART_OF = "part_of"       # belongs to a place


class StaleFact(RuntimeError):
    """Raised instead of returning a fact too old to act on."""


@dataclass
class SceneNode:
    """One entity. Geometry is cached; TF2 is the authority."""

    node_id: str
    kind: NodeKind
    label: str                        # open-vocabulary, from OpenMask3D's CLIP match
    tf_frame: str                     # where a consumer gets the LIVE pose
    anchor_frame: str                 # what `pose` is expressed in; a keyframe, not map
    pose: np.ndarray                  # (4, 4) in anchor_frame
    extent: np.ndarray                # (3,) axis-aligned size in metres
    stamp_ns: int
    confidence: float = 1.0
    metric: bool = True               # False if SLAM was monocular; see scene_export
    clip_similarity: float | None = None
    # PLACE nodes only: the enclosing place. Building -> floor -> room -> workspace.
    # The place hierarchy is a TREE (one parent), unlike the relation graph around it.
    place_parent: str | None = None
    # Convex hull of the instance's points, in two forms, both optional. `inside` is
    # decided from these when present and falls back to the bounding box when not.
    #
    # An AABB says "inside" whenever one coarse proposal's box sits within another's,
    # which on a real scene is constantly and mostly wrongly: measured on
    # freiburg3_walking_xyz it produced `the chair is inside the desk` for a chair whose
    # points are 4.8% inside that desk. The hull is the smallest convex region the
    # evidence supports, and it is CHEAP -- 35-80 vertices and 66-156 planes for
    # instances of 382-6,694 points -- so nodes stay small enough to publish.
    hull_vertices: np.ndarray | None = None   # (V, 3) in anchor_frame
    hull_planes: np.ndarray | None = None     # (F, 4) outward half-spaces, Ax + b <= 0
    # A capped, evenly-spread sample of the instance's points, for `near`.
    #
    # The hull cannot serve here: its vertices are by construction the points reaching
    # FURTHEST toward a neighbour, so hull-to-hull distance is biased short and fired
    # `near` on 11 of 15 pairs where the full point sets support 7. Interior samples do
    # not have that bias. Capped at a couple of hundred points so a node stays publishable.
    sample_points: np.ndarray | None = None   # (S, 3) in anchor_frame
    # Area of the largest horizontal slab in the instance, m^2, when it was measured.
    # None means the kind came from the label fallback rather than from geometry — worth
    # being able to tell apart when a relation looks wrong.
    support_area: float | None = None

    def __post_init__(self) -> None:
        self.pose = np.asarray(self.pose, dtype=np.float64).reshape(4, 4)
        self.extent = np.asarray(self.extent, dtype=np.float64).reshape(3)
        if self.hull_vertices is not None:
            self.hull_vertices = np.asarray(self.hull_vertices, float).reshape(-1, 3)
        if self.hull_planes is not None:
            self.hull_planes = np.asarray(self.hull_planes, float).reshape(-1, 4)
        if self.sample_points is not None:
            self.sample_points = np.asarray(self.sample_points, float).reshape(-1, 3)
        if not self.metric:
            # A non-metric pose is not a slightly worse pose, it is a pose in unknown
            # units. Letting it into the graph means every derived distance is wrong by
            # an unknown factor, and nothing downstream can detect it.
            raise ValueError(
                f"node {self.node_id!r} carries a non-metric pose. Monocular SLAM is "
                "scale-free — re-run with depth so the gauge is fixed."
            )

    @property
    def centre(self) -> np.ndarray:
        return self.pose[:3, 3]

    @property
    def volume(self) -> float:
        return float(np.prod(self.extent))

    def contains(self, other: "SceneNode") -> bool:
        """Does this region enclose the other's centre?

        Centre rather than full containment on purpose: a cup on a table at a room
        boundary is still in the room, and requiring the whole box inside would put it
        in no room at all.
        """
        lo, hi = self.aabb()
        return bool(np.all(other.centre >= lo) and np.all(other.centre <= hi))

    def hull_fraction_inside(self, other: "SceneNode", tol: float = 0.02) -> float:
        """Fraction of THIS node's hull vertices lying inside `other`'s hull.

        Testing vertices rather than every point is exact, not an approximation: a
        convex hull is the convex combination of its vertices, so if every vertex is
        inside a convex region then so is everything between them.

        Returns NaN when either hull is missing, which the caller reads as "no hull
        evidence" and falls back to the bounding box rather than silently answering 0.
        """
        if self.hull_vertices is None or other.hull_planes is None:
            return float("nan")
        d = self.hull_vertices @ other.hull_planes[:, :3].T + other.hull_planes[:, 3]
        return float((d.max(axis=1) <= tol).mean())

    def separation_from(self, other: "SceneNode", pct: float = 5.0) -> float:
        """How far this node's surface actually is from `other`'s, in metres.

        The `pct`-th percentile of per-point nearest-neighbour distance, not the minimum:
        a single stray point from a ragged segmentation would otherwise put two objects
        in contact. 0 means the two point sets interpenetrate.

        Returns NaN when either node lacks samples, so the caller can fall back to
        centre distance rather than treat "unknown" as "touching".
        """
        if self.sample_points is None or other.sample_points is None:
            return float("nan")
        if not len(self.sample_points) or not len(other.sample_points):
            return float("nan")
        # Brute force: both sides are capped at a couple of hundred points, so this is a
        # small dense product and needs no spatial index.
        d = np.linalg.norm(
            self.sample_points[:, None, :] - other.sample_points[None, :, :], axis=-1)
        return float(np.percentile(d.min(axis=1), pct))

    def age_s(self, now_ns: int) -> float:
        return (now_ns - self.stamp_ns) / NS_PER_S

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned box in the anchor frame.

        Axis-aligned rather than oriented on purpose: relations like `on` and `near` are
        coarse predicates, and an OBB would imply a precision the upstream extents do
        not have.
        """
        half = self.extent / 2.0
        return self.centre - half, self.centre + half


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: Predicate
    object: str
    stamp_ns: int
    evidence: str = ""       # why this was inferred, for debugging a wrong plan

    def as_text(self, nodes: dict[str, SceneNode]) -> str:
        """Render for a policy prompt.

        Two instances that share a label must not render identically: "the desk is near
        the desk" is geometrically true of two distinct desks and reads as nonsense to
        whatever consumes the prompt. Naming the second one "another desk" keeps the
        sentence honest without exposing instance ids, which mean nothing to a policy.
        """
        s = nodes[self.subject].label if self.subject in nodes else self.subject
        o = nodes[self.object].label if self.object in nodes else self.object
        if s == o:
            return f"the {s} is {self.predicate.value} another {o}"
        return f"the {s} is {self.predicate.value} the {o}"


@dataclass
class SceneGraph:
    """Nodes plus derived relations, with per-kind staleness."""

    nodes: dict[str, SceneNode] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    max_age_s: dict[NodeKind, float] = field(
        default_factory=lambda: dict(DEFAULT_MAX_AGE_S)
    )

    # ------------------------------------------------------------------ writes

    def upsert(self, node: SceneNode) -> None:
        """Insert or refresh. Identity is the node id, not the label.

        Refreshing by id rather than by label matters as soon as there are two cups:
        keying on the label would collapse them into one flickering object.
        """
        existing = self.nodes.get(node.node_id)
        if existing is not None and node.stamp_ns < existing.stamp_ns:
            # Out-of-order arrival. FoundationPose runs on its own thread and a late
            # reply must not roll the world backwards.
            return
        self.nodes[node.node_id] = node

    def forget_older_than(self, now_ns: int) -> list[str]:
        """Drop nodes past their kind's tolerance. Returns what was removed."""
        removed = []
        for node_id, node in list(self.nodes.items()):
            limit = self.max_age_s[node.kind]
            if limit != float("inf") and node.age_s(now_ns) > limit:
                del self.nodes[node_id]
                removed.append(node_id)
        if removed:
            gone = set(removed)
            self.relations = [
                r for r in self.relations
                if r.subject not in gone and r.object not in gone
            ]
        return removed

    # -------------------------------------------------------------- inference

    def infer_relations(self, now_ns: int, *, contact_tol_m: float = 0.04,
                        near_m: float = 0.45, hull_frac: float = 0.95) -> list[Relation]:
        """Derive spatial predicates from cached geometry. Replaces prior relations.

        Recomputed rather than incrementally patched: objects move, and a stale `on`
        edge pointing at a table the cup has left is worse than no edge, because a
        planner will happily use it.
        """
        self.relations = []
        movable = [n for n in self.nodes.values() if n.kind is NodeKind.OBJECT]
        supports = [n for n in self.nodes.values()
                    if n.kind in (NodeKind.SURFACE, NodeKind.OBJECT)]

        for obj in movable:
            lo, hi = obj.aabb()
            for base in supports:
                if base.node_id == obj.node_id:
                    continue
                blo, bhi = base.aabb()

                # ON: the object's underside sits at the support's top surface, and
                # their footprints overlap. The vertical test is a band, not equality —
                # extents from a fused cloud are good to a few centimetres at best.
                resting = abs(lo[2] - bhi[2]) <= contact_tol_m
                overlap = (lo[0] < bhi[0] and hi[0] > blo[0]
                           and lo[1] < bhi[1] and hi[1] > blo[1])
                if resting and overlap:
                    self.relations.append(Relation(
                        obj.node_id, Predicate.ON, base.node_id, now_ns,
                        evidence=f"base {lo[2]:.3f} m vs top {bhi[2]:.3f} m, footprints overlap",
                    ))
                    continue

                # INSIDE: the box test is only a cheap PREFILTER. Where hulls exist the
                # claim has to survive them too, because an AABB says "inside" for any
                # two coarse proposals whose boxes happen to nest -- which on real
                # segmentations is most overlapping pairs.
                if np.all(lo >= blo - contact_tol_m) and np.all(hi <= bhi + contact_tol_m):
                    frac = obj.hull_fraction_inside(base)
                    if np.isnan(frac):
                        self.relations.append(Relation(
                            obj.node_id, Predicate.INSIDE, base.node_id, now_ns,
                            evidence="bounding box contained (no hull evidence)",
                        ))
                        continue
                    if frac >= hull_frac:
                        self.relations.append(Relation(
                            obj.node_id, Predicate.INSIDE, base.node_id, now_ns,
                            evidence=f"{frac * 100:.0f}% of hull vertices contained",
                        ))
                        continue
                    # Box said inside, hull disagreed. Fall through so the pair can
                    # still earn a NEAR edge -- which is usually what it actually is.

                # NEAR: surface separation where samples exist, centre distance only as
                # a fallback. Centre distance asks the wrong question for extended
                # objects -- a chair 7 cm from a desk has its centre 79 cm away, so the
                # true relation was being missed while nothing useful replaced it.
                sep = obj.separation_from(base)
                if np.isnan(sep):
                    sep = float(np.linalg.norm(obj.centre - base.centre))
                    why = f"centres {sep:.3f} m apart (no sample points)"
                else:
                    why = f"surfaces {sep:.3f} m apart"
                if sep <= near_m:
                    self.relations.append(Relation(
                        obj.node_id, Predicate.NEAR, base.node_id, now_ns, evidence=why,
                    ))

        self.relations.extend(self._infer_containment(now_ns))
        return self.relations

    def _infer_containment(self, now_ns: int) -> list[Relation]:
        """PART_OF edges from things to the INNERMOST place that holds them.

        One edge, not one per ancestor. A cup in a workspace inside a kitchen inside a
        floor gets `part_of workspace`; the rest is recovered by walking `place_parent`.
        Materialising every ancestor edge would mean a room that moves leaves a trail of
        edges that individually still look valid.

        Innermost is resolved by smallest volume, which is also the tiebreak for two
        overlapping places at the same depth — the tighter region is the more specific
        claim.
        """
        places = [n for n in self.nodes.values() if n.kind is NodeKind.PLACE]
        if not places:
            return []

        out: list[Relation] = []
        for node in self.nodes.values():
            if node.kind is NodeKind.PLACE:
                # A place's parent is declared, not inferred: nesting is a statement
                # about the map, and guessing it from overlapping boxes would reparent
                # rooms whenever an extent was re-estimated.
                if node.place_parent and node.place_parent in self.nodes:
                    out.append(Relation(node.node_id, Predicate.PART_OF,
                                        node.place_parent, now_ns,
                                        evidence="declared nesting"))
                continue
            holding = [p for p in places if p.contains(node)]
            if not holding:
                continue
            innermost = min(holding, key=lambda p: p.volume)
            out.append(Relation(
                node.node_id, Predicate.PART_OF, innermost.node_id, now_ns,
                evidence=f"inside {innermost.label} ({innermost.volume:.2f} m^3), "
                         f"innermost of {len(holding)}",
            ))
        return out

    # ------------------------------------------------------------- the hierarchy

    def ancestry(self, node_id: str) -> list[str]:
        """Place ids from innermost outward. Empty if the node is in no place."""
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(node_id)

        if node.kind is NodeKind.PLACE:
            current = node.place_parent
        else:
            holding = [p for p in self.nodes.values()
                       if p.kind is NodeKind.PLACE and p.contains(node)]
            current = min(holding, key=lambda p: p.volume).node_id if holding else None

        chain: list[str] = []
        seen: set[str] = set()
        while current and current in self.nodes:
            if current in seen:
                # A cycle means the map was declared wrong. Returning the prefix is
                # better than looping forever inside a navigation query.
                raise ValueError(
                    f"place hierarchy cycles at {current!r}; chain so far {chain}"
                )
            seen.add(current)
            chain.append(current)
            current = self.nodes[current].place_parent
        return chain

    def nodes_in_place(self, place_id: str, *, recursive: bool = True) -> list[SceneNode]:
        """Everything in a place. Recursive by default.

        "What is in the kitchen" must include the cup on the counter in the workspace
        inside the kitchen — a non-recursive answer is almost never the question asked.
        """
        if place_id not in self.nodes:
            raise KeyError(place_id)
        out = []
        for node in self.nodes.values():
            if node.node_id == place_id:
                continue
            chain = self.ancestry(node.node_id)
            if not chain:
                continue
            if chain[0] == place_id or (recursive and place_id in chain):
                out.append(node)
        return out

    def place_of(self, node_id: str) -> SceneNode | None:
        chain = self.ancestry(node_id)
        return self.nodes[chain[0]] if chain else None

    # ---------------------------------------------------------------- queries

    def _fresh(self, node: SceneNode, now_ns: int) -> SceneNode:
        limit = self.max_age_s[node.kind]
        if limit != float("inf") and node.age_s(now_ns) > limit:
            raise StaleFact(
                f"{node.kind.value} {node.node_id!r} is {node.age_s(now_ns):.3f} s old, "
                f"past the {limit:.2f} s tolerance for its kind. Acting on it would use "
                "a pose the world has already left."
            )
        return node

    def resolve(self, phrase: str, now_ns: int) -> SceneNode:
        """Referring expression -> one node. The grounding step before a grasp.

        Ranked by CLIP similarity where OpenMask3D supplied it, so "the mug" picks the
        best match rather than the first insertion.
        """
        phrase_l = phrase.lower().strip()
        candidates = [
            n for n in self.nodes.values()
            if phrase_l in n.label.lower() or n.label.lower() in phrase_l
        ]
        if not candidates:
            known = sorted({n.label for n in self.nodes.values()})
            raise KeyError(f"nothing matching {phrase!r}; graph holds {known}")
        best = max(candidates, key=lambda n: (n.clip_similarity or 0.0, n.confidence))
        return self._fresh(best, now_ns)

    def objects_on(self, support_id: str, now_ns: int) -> list[SceneNode]:
        return [
            self.nodes[r.subject] for r in self.relations
            if r.predicate is Predicate.ON and r.object == support_id
            and r.subject in self.nodes
            and self.nodes[r.subject].age_s(now_ns) <= self.max_age_s[NodeKind.OBJECT]
        ]

    def describe(self) -> list[str]:
        """Human-readable relations — what a language model would be handed."""
        return [r.as_text(self.nodes) for r in self.relations]

    def snapshot(self, now_ns: int) -> dict:
        """Flat, serialisable view for a 30 Hz consumer.

        Deliberately cheap and local: the VLA tick cannot afford a service round trip
        per frame, so consumers keep a cached graph and read from it. `stale` is
        reported rather than filtered, because whether a 200 ms-old object pose is
        usable depends on what the caller is about to do with it.
        """
        return {
            "stamp_ns": now_ns,
            "nodes": [
                {
                    "id": n.node_id, "kind": n.kind.value, "label": n.label,
                    "tf_frame": n.tf_frame, "anchor_frame": n.anchor_frame,
                    "age_s": round(n.age_s(now_ns), 4),
                    "stale": (self.max_age_s[n.kind] != float("inf")
                              and n.age_s(now_ns) > self.max_age_s[n.kind]),
                    "extent": n.extent.tolist(),
                }
                for n in self.nodes.values()
            ],
            "relations": [
                {"subject": r.subject, "predicate": r.predicate.value,
                 "object": r.object, "evidence": r.evidence}
                for r in self.relations
            ],
        }
