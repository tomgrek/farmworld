import math

def get_covered_area(bitmap, soil_color, screen_size):
    """Return fraction of area (screen_size) that's fields.
    Assume 32-bit RGBA"""
    buf = bitmap.get_buffer().raw
    bgcol = 0
    for px in range(0, len(buf), 4):
        b, g, r, a = buf[px : px + 4]
        if (r, g, b) == (190, 200, 255):
            bgcol += 1
    return (math.prod(screen_size) - bgcol) / math.prod(screen_size)