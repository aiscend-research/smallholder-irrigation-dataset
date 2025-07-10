from geopy.distance import distance
from shapely.geometry import Polygon

def bounding_box(center_lat, center_lon, half_side_km=0.5):
    """
    Returns a truly geodesic bounding box ~1 km wide/high,
    centered on (center_lat, center_lon).
pip 
    :param center_lat: Center latitude in decimal degrees.
    :param center_lon: Center longitude in decimal degrees.
    :return: (min_lat, min_lon, max_lat, max_lon)
    """
    # Move half_side_km km north, south, east, and west from the center
    north_point = distance(kilometers=half_side_km).destination((center_lat, center_lon), bearing=0)
    south_point = distance(kilometers=half_side_km).destination((center_lat, center_lon), bearing=180)
    east_point  = distance(kilometers=half_side_km).destination((center_lat, center_lon), bearing=90)
    west_point  = distance(kilometers=half_side_km).destination((center_lat, center_lon), bearing=270)

    min_lat = south_point.latitude
    max_lat = north_point.latitude
    min_lon = west_point.longitude
    max_lon = east_point.longitude

    return (min_lat, min_lon, max_lat, max_lon)

def survey_polygon(row):
    # Compute bounding box (min_lat, min_lon, max_lat, max_lon)
    bb = bounding_box(row["y"], row["x"])
    # Construct polygon in (lon, lat): SW, SE, NE, NW, then back to SW.
    return Polygon([
        (bb[1], bb[0]),  # SW
        (bb[3], bb[0]),  # SE
        (bb[3], bb[2]),  # NE
        (bb[1], bb[2]),  # NW
        (bb[1], bb[0])   # close polygon
    ])

# Test how well this function compares to what Earth Collect bounding boxes are caulated as
if __name__ == "__main__":
    
    # id_5345209
    lat_center = -17.146892667882046
    lon_center = 27.1071721513814
    bbox = bounding_box(lat_center, lon_center, 0.5) # 500 meters in each direction
    print("Bounding box (geodesic 1km square):", bbox)

    earth_collect_bbox = (
        -17.15141029504564,   # south; min_lat
        27.1024730149089,     # west; min_lon
        -17.1423747110346,    # north; max_lat
        27.1118710606301      # east; max_lon
    )

    labels = ["South", "West", "North", "East"]

    for label, comp, truth in zip(labels, bbox, earth_collect_bbox):
        diff = abs(comp - truth)
        print(f"{label}: computed={comp}, truth={truth}, diff={diff}") 
    
    # We find that the borders are only ~0-3 cm off ^^

    # id_5200403
    lat_center = -16.15829617864069
    lon_center = 28.18225451770237
    bbox = bounding_box(lat_center, lon_center, 0.5) # 500 meters in each direction

    earth_collect_bbox = (
        -16.16281425251669,   # south; min_lat
        28.17757942318091,    # west; min_lon
        -16.15377779511203,   # north; max_lat
        28.18692939993126     # east; max_lon
    )

    for label, comp, truth in zip(labels, bbox, earth_collect_bbox):
        diff = abs(comp - truth)
        print(f"{label}: computed={comp}, truth={truth}, diff={diff}") 

    # This is also about 3 cm off. 