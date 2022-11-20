import json

import polygenerator

import farmworld.geojson.util as util

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


def poly_from_geojson(geojson, screen_size):
    coords = []
    for feature in geojson["features"]:
        if feature.get("geometry", {}).get("type", "") == "Polygon":
            coords.append(feature["geometry"]["coordinates"][0])

    if len(coords) == 0:
        raise Exception("GeoJSON needs at least one geometry feature of type polygon, with at least one coordinates set.")

    fields = []
    for field in coords:
        field_coords = []
        for x, y in field:
            field_coords.append([x, y * -1])
        fields.append(field_coords)
    
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
        sfx, sfy = screen_size[0] / xmax, screen_size[1] / ymax
        for x, y in field:
            newfield.append([x * sfx, y * sfy])
        fields[i] = newfield
    
    return fields[0]
