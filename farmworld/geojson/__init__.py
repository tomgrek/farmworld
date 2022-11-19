import json

import polygenerator


def get_geojson(name=None):
    if name:
        try:
            with open(name, "r") as j:
                geojson = json.load(j)
        except:
            with open(f'{__file__.split("__init__.py")[0]}{name}', "r") as j:
                geojson = json.load(j)
    else:
        geojson = {"features": [{"geometry": {"type": "Polygon", "coordinates": [polygenerator.random_convex_polygon(10)]}}]}
    return geojson


def poly_from_geojson(geojson, scale_x=500, scale_y=500):
    for feature in geojson["features"]:
        if feature.get("geometry", {}).get("type", "") == "Polygon":
            coords = feature["geometry"]["coordinates"][0]

    if not coords:
        raise Exception("GeoJSON needs at least one geometry feature of type polygon, with at least one coordinates set.")

    newcoords = []
    for x, y in coords:
        newcoords.append([x, y * -1])
    coords = newcoords

    xmin, xmax = min(x[0] for x in coords), max(x[0] for x in coords)
    ymin, ymax = min(x[1] for x in coords), max(x[1] for x in coords)
    xmin, xmax = abs(xmin), abs(xmax)
    ymin, ymax = abs(ymin), abs(ymax)
    newcoords = []
    for x, y in coords:
        newcoords.append([x + xmin, y + ymin])
    coords = newcoords
    xmin, xmax = min(x[0] for x in coords), max(x[0] for x in coords)
    ymin, ymax = min(x[1] for x in coords), max(x[1] for x in coords)
    xmin, xmax = abs(xmin), abs(xmax)
    ymin, ymax = abs(ymin), abs(ymax)

    if xmin > 0:
        newcoords = []
        for x, y in coords:
            newcoords.append([x - xmin, y])
        coords = newcoords
    if ymin > 0:
        newcoords = []
        for x, y in coords:
            newcoords.append([x, y - ymin])
        coords = newcoords

    xmin, xmax = min(x[0] for x in coords), max(x[0] for x in coords)
    ymin, ymax = min(x[1] for x in coords), max(x[1] for x in coords)
    xmin, xmax = abs(xmin), abs(xmax)
    ymin, ymax = abs(ymin), abs(ymax)

    newcoords = []
    sfx, sfy = scale_x / xmax, scale_y / ymax
    for x, y in coords:
        newcoords.append([x * sfx, y * sfy])
    return newcoords
