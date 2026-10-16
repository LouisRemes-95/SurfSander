
from collections import deque
from collections.abc import Callable
from heapq import heapify, heappop, heappush
from itertools import product

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from surfsander.common import console


# --------------------------------------------------
# 1. Lookup tables
# --------------------------------------------------

def make_lut(n_bits: int, is_critical: Callable[[int], bool]) -> np.ndarray:
    critical = np.array([is_critical(p) for p in range(1 << n_bits)], dtype=np.int8)
    patterns = np.arange(1 << n_bits)
    return np.stack([critical - critical[patterns ^ (1 << bit)] for bit in range(n_bits)])


def is_critical_c1(pattern: int) -> bool:
    return pattern in (0b0110, 0b1001)


C2_PATTERNS = set()

for bit in range(4):
    diagonal = (1 << bit) | (1 << (7 - bit))
    C2_PATTERNS.update((diagonal, 255 ^ diagonal))


def is_critical_c2(pattern: int) -> bool:
    return pattern in C2_PATTERNS


C1_LUT = make_lut(4, is_critical_c1)
C2_LUT = make_lut(8, is_critical_c2)


# --------------------------------------------------
# 2. Configuration offsets
# --------------------------------------------------

def configurations():
    for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
        offsets = []

        for i, j in product(range(2), repeat=2):
            offset = [0, 0, 0]
            offset[axis_a] = i
            offset[axis_b] = j
            offsets.append(tuple(offset))

        yield offsets, C1_LUT

    yield list(product(range(2), repeat=3)), C2_LUT


CONFIGS = tuple(configurations())


# --------------------------------------------------
# 3. Configuration counting and initial scores
# --------------------------------------------------

def compute_patterns(volume: np.ndarray, offsets: list[tuple[int, int, int]]) -> np.ndarray:
    extent = np.max(offsets, axis=0)
    base_shape = tuple(volume.shape[d] - extent[d] for d in range(3))
    patterns = np.zeros(base_shape, dtype=np.uint8)

    for bit, offset in enumerate(offsets):
        slices = tuple(slice(offset[d], offset[d] + base_shape[d]) for d in range(3))
        patterns |= volume[slices].astype(np.uint8) << bit

    return patterns


def count_critical_configurations(volume: np.ndarray) -> int:
    count = 0

    for offsets, _ in CONFIGS:
        extent = np.max(offsets, axis=0)

        if np.any(np.array(volume.shape) <= extent):
            continue

        patterns = compute_patterns(volume, offsets)
        critical = (0b0110, 0b1001) if len(offsets) == 4 else C2_PATTERNS
        count += int(np.count_nonzero(np.isin(patterns, list(critical))))

    return count


def compute_flip_scores(volume: np.ndarray) -> np.ndarray:
    scores = np.zeros(volume.shape, dtype=np.int16)

    for offsets, lut in CONFIGS:
        extent = np.max(offsets, axis=0)

        if np.any(np.array(volume.shape) <= extent):
            continue

        patterns = compute_patterns(volume, offsets)
        base_shape = patterns.shape

        for bit, offset in enumerate(offsets):
            slices = tuple(slice(offset[d], offset[d] + base_shape[d]) for d in range(3))
            scores[slices] += lut[bit, patterns]

    return scores


# --------------------------------------------------
# 4. Local configuration and flip score
# --------------------------------------------------

def configuration_pattern(
    volume: np.ndarray,
    origin: tuple[int, int, int],
    offsets: list[tuple[int, int, int]],
) -> int:
    pattern = 0

    for bit, offset in enumerate(offsets):
        p = tuple(origin[d] + offset[d] for d in range(3))
        pattern |= int(volume[p]) << bit

    return pattern


def compute_flip_score(volume: np.ndarray, p: tuple[int, int, int]) -> int:
    score = 0

    for offsets, lut in CONFIGS:
        extent = np.max(offsets, axis=0)

        for bit, offset in enumerate(offsets):
            origin = tuple(p[d] - offset[d] for d in range(3))

            if any(origin[d] < 0 or origin[d] + extent[d] >= volume.shape[d] for d in range(3)):
                continue

            pattern = configuration_pattern(volume, origin, offsets)
            score += int(lut[bit, pattern])

    return score


# --------------------------------------------------
# 5. Guaranteed addition-only repair
# --------------------------------------------------

def finish_well_composedness(volume: np.ndarray) -> int:
    queue = deque()
    pending = set()

    # Find all critical configurations remaining after greedy repair.
    for config_id, (offsets, _) in enumerate(CONFIGS):
        extent = np.max(offsets, axis=0)

        if np.any(np.array(volume.shape) <= extent):
            continue

        patterns = compute_patterns(volume, offsets)
        critical = (0b0110, 0b1001) if len(offsets) == 4 else C2_PATTERNS

        for location in np.argwhere(np.isin(patterns, list(critical))):
            key = (config_id, tuple(int(x) for x in location))
            queue.append(key)
            pending.add(key)

    n_flips = 0

    def enqueue_affected(p: tuple[int, int, int]) -> None:
        # Only configurations containing the modified voxel can change.
        for config_id, (offsets, _) in enumerate(CONFIGS):
            extent = np.max(offsets, axis=0)

            for offset in offsets:
                origin = tuple(p[d] - offset[d] for d in range(3))

                if any(origin[d] < 0 or origin[d] + extent[d] >= volume.shape[d] for d in range(3)):
                    continue

                key = (config_id, origin)

                if key not in pending:
                    queue.append(key)
                    pending.add(key)

    with console.status("[cyan]Resolving remaining critical configurations..."):
        while queue:
            config_id, origin = queue.popleft()
            pending.remove((config_id, origin))

            offsets, _ = CONFIGS[config_id]
            pattern = configuration_pattern(volume, origin, offsets)

            is_critical = is_critical_c1(pattern) if len(offsets) == 4 else is_critical_c2(pattern)

            # The configuration may have been resolved by an earlier flip.
            if not is_critical:
                continue

            # Only allow background -> foreground.
            candidates = [
                tuple(origin[d] + offset[d] for d in range(3))
                for offset in offsets
                if not volume[tuple(origin[d] + offset[d] for d in range(3))]
            ]

            # Prefer the flip with the greatest immediate topological improvement.
            best = max(candidates, key=lambda p: compute_flip_score(volume, p))

            volume[best] = True
            n_flips += 1

            enqueue_affected(best)

    return n_flips


# --------------------------------------------------
# 6. Complete well-composedness correction
# --------------------------------------------------

def make_well_composed(volume: np.ndarray) -> tuple[np.ndarray, int]:
    volume = np.asarray(volume, dtype=bool).copy()

    if volume.ndim != 3:
        raise ValueError("Expected a 3D binary volume.")

    with console.status("[cyan]Computing initial flip scores..."):
        scores = compute_flip_scores(volume)
        n_initial = count_critical_configurations(volume)

    versions = np.zeros(volume.shape, dtype=np.int32)
    flat_scores = scores.ravel()

    heap = [(-int(flat_scores[i]), int(i), 0) for i in np.flatnonzero(flat_scores > 0)]
    heapify(heap)

    n_greedy = 0
    n_remaining = n_initial

    # Phase 1: Greedy correction
    with Progress(
        TextColumn("[cyan]Enforcing well-composedness — Critical configurations resolved:"),
        BarColumn(),
        TextColumn("{task.completed:,.0f}/{task.total:,.0f} resolved"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:

        task = progress.add_task("correct", total=n_initial)

        while heap:
            neg_score, index, version = heappop(heap)
            p = tuple(int(x) for x in np.unravel_index(index, volume.shape))

            # Ignore outdated heap entries.
            if version != versions[p] or -neg_score != scores[p]:
                continue

            improvement = int(scores[p])

            volume[p] = ~volume[p]
            n_greedy += 1
            n_remaining -= improvement

            progress.advance(task, improvement)

            # Update only the affected 3x3x3 neighborhood.
            for offset in product((-1, 0, 1), repeat=3):
                q = tuple(p[d] + offset[d] for d in range(3))

                if any(q[d] < 0 or q[d] >= volume.shape[d] for d in range(3)):
                    continue

                scores[q] = compute_flip_score(volume, q)
                versions[q] += 1

                if scores[q] > 0:
                    q_index = int(np.ravel_multi_index(q, volume.shape))
                    heappush(heap, (-int(scores[q]), q_index, int(versions[q])))

    # Phase 2: Guaranteed addition-only correction
    n_fallback = 0

    if n_remaining:
        with console.status("[cyan]Applying guaranteed repair..."):
            n_fallback = finish_well_composedness(volume)

    # Final validation
    with console.status("[cyan]Validating voxel topology..."):
        remaining = count_critical_configurations(volume)

    if remaining:
        raise RuntimeError(f"Well-composedness repair failed: {remaining} critical configurations remain.")

    n_flips = n_greedy + n_fallback

    console.print(
        f"[green]✔ Well-composedness achieved in {n_flips:,} voxel flips[/green] "
        f"[white]({n_greedy:,} greedy + {n_fallback:,} fallback)[/white]"
    )

    return volume, n_flips