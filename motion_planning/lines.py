# lines.py
from typing import List, Tuple

def line_points_8(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int,int]]:
    """Bresenham-style 8-connected line sampler (returns list of (r,c))."""
    x0, y0 = c0, r0
    x1, y1 = c1, r1
    pts = []
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        pts.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return pts

def line_points_4(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int,int]]:
    """4-connected line sampler (Manhattan steps with splits)."""
    x0, y0 = c0, r0
    x1, y1 = c1, r1
    pts = [(y0, x0)]
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx >= dy:
        err = dx // 2
        while x0 != x1:
            x0 += sx; err -= dy
            pts.append((y0, x0))
            if err < 0:
                y0 += sy; err += dx
                pts.append((y0, x0))
    else:
        err = dy // 2
        while y0 != y1:
            y0 += sy; err -= dx
            pts.append((y0, x0))
            if err < 0:
                x0 += sx; err += dy
                pts.append((y0, x0))
    return pts

def line_points_pixels(p0_xy: Tuple[int,int], p1_xy: Tuple[int,int], use8: bool) -> List[Tuple[int,int]]:
    (x0, y0), (x1, y1) = p0_xy, p1_xy
    return (line_points_8 if use8 else line_points_4)(y0, x0, y1, x1)

def line_is_free(mask, p0_xy: Tuple[int,int], p1_xy: Tuple[int,int], use8: bool) -> bool:
    """Check if pixel-line between two points is entirely free on a boolean mask."""
    H, W = mask.shape
    for (r, c) in line_points_pixels(p0_xy, p1_xy, use8):
        if r < 0 or c < 0 or r >= H or c >= W or mask[r, c]:
            return False
    return True
