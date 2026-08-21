from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CrosswalkRegion:
    x1: float  # normalized [0, 1]
    y1: float
    x2: float
    y2: float
    confidence: float
    method: str  # "classical"


def iou(box_a: CrosswalkRegion, box_b: CrosswalkRegion) -> float:
    ix1 = max(box_a.x1, box_b.x1)
    iy1 = max(box_a.y1, box_b.y1)
    ix2 = min(box_a.x2, box_b.x2)
    iy2 = min(box_a.y2, box_b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (box_a.x2 - box_a.x1) * (box_a.y2 - box_a.y1)
    area_b = (box_b.x2 - box_b.x1) * (box_b.y2 - box_b.y1)
    return inter / (area_a + area_b - inter)


def is_inside_crosswalk(cx: float, cy: float, regions: List[CrosswalkRegion]) -> bool:
    return any(r.x1 <= cx <= r.x2 and r.y1 <= cy <= r.y2 for r in regions)


def merge_regions(regions: List[CrosswalkRegion], iou_thresh: float = 0.5) -> List[CrosswalkRegion]:
    merged: List[CrosswalkRegion] = []
    used = [False] * len(regions)
    for i, r in enumerate(regions):
        if used[i]:
            continue
        group = [r]
        for j in range(i + 1, len(regions)):
            if not used[j] and iou(r, regions[j]) >= iou_thresh:
                group.append(regions[j])
                used[j] = True
        best = max(group, key=lambda x: x.confidence)
        merged.append(best)
        used[i] = True
    return merged
