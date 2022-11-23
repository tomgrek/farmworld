import math

from farmworld.const import SKY

def get_covered_area(bitmap, screen_size, grid_size):
    """Return fraction of area (screen_size) that's fields.
    Assume 32-bit RGBA"""
    buf = bitmap.get_buffer().raw
    bgcol = 0
    for px in range(0, len(buf), 4):
        b, g, r, a = buf[px : px + 4]
        if (r, g, b) == SKY:
            bgcol += 1
    return ((math.prod(screen_size) - bgcol) / math.prod(screen_size)) * (grid_size**2)