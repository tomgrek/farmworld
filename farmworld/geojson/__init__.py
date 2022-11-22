import json
import math

import polygenerator

from farmworld.geojson import util

def get_geojson(name=None, num_fields=None):
    if name:
        assert num_fields is None, "Can only set num_fields if they're randomly generated."
    if name:
        try:
            with open(name, "r") as j:
                geojson = json.load(j)
        except:
            with open(f'{__file__.split("__init__.py")[0]}{name}', "r") as j:
                geojson = json.load(j)
    else:
        num_fields = num_fields or 1
        geojson = {"features": [{"geometry": {"type": "Polygon", "coordinates": [polygenerator.random_convex_polygon(10)]}} for _ in range(num_fields)]}
    return geojson


def poly_and_scaling_from_geojson(geojson, screen_size):
    """Returns list of scaled polys, plus grid_size and cell_size"""
    num_fields = len(geojson)
    grid_size = math.ceil(math.sqrt(num_fields))
    cell_size_x, cell_size_y = screen_size[0] / grid_size, screen_size[1] / grid_size
    
    coords = []
    for feature in geojson:
        if feature.get("geometry", {}).get("type", "") == "Polygon":
            coords.append(feature["geometry"]["coordinates"][0])

    if len(coords) == 0:
        raise Exception("GeoJSON needs at least one geometry feature of type polygon, with at least one coordinates set.")

    fields = []
    # flip it along y axis
    for field in coords:
        field_coords = []
        for x, y in field:
            field_coords.append([x, y * -1])
        fields.append(field_coords)
    
    render_infos = []
    
    for i, field in enumerate(fields):
        xmin, xmax = min(x[0] for x in field), max(x[0] for x in field)
        ymin, ymax = min(x[1] for x in field), max(x[1] for x in field)
        xmin, xmax = abs(xmin), abs(xmax)
        ymin, ymax = abs(ymin), abs(ymax)
        newfield = []
        for x, y in field:
            newfield.append([x + xmin, y + ymin])
        field = newfield
        xmin, xmax = min(x[0] for x in field), max(x[0] for x in field)
        ymin, ymax = min(x[1] for x in field), max(x[1] for x in field)
        xmin, xmax = abs(xmin), abs(xmax)
        ymin, ymax = abs(ymin), abs(ymax)

        if xmin > 0:
            newfield = []
            for x, y in field:
                newfield.append([x - xmin, y])
            field = newfield
        if ymin > 0:
            newfield = []
            for x, y in field:
                newfield.append([x, y - ymin])
            field = newfield

        xmin, xmax = min(x[0] for x in field), max(x[0] for x in field)
        ymin, ymax = min(x[1] for x in field), max(x[1] for x in field)
        xmin, xmax = abs(xmin), abs(xmax)
        ymin, ymax = abs(ymin), abs(ymax)

        newfield = []
        sfx, sfy = cell_size_x / xmax, cell_size_y / ymax

        row = math.floor(i / grid_size)
        start_x = (i % grid_size) * cell_size_x
        start_y = row * cell_size_y
        render_infos.append({"start_x": start_x, "start_y": start_y, "end_x": start_x + cell_size_x, "end_y": start_y + cell_size_y})
        for coord in field:
            x, y = coord
            newfield.append([(x * sfx) + start_x, (y * sfy) + start_y])
        fields[i] = newfield

    
    return fields, render_infos, grid_size, (cell_size_x, cell_size_y)
