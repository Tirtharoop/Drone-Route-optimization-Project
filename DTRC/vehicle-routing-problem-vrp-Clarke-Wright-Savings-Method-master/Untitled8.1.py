import googlemaps
import gmplot
import polyline

API_KEY = "AIzaSyBUAALV-_QY-CAJHYJw51VuWrGCE8QcL90"
gmaps = googlemaps.Client(key=API_KEY)

# Use coordinates instead of place names
start_coords = "22.300501,88.140303"
end_coords = "22.567990,88.415120"

# Get driving directions using coordinates
directions_result = gmaps.directions(start_coords, end_coords, mode="driving")

# Decode all step-level polylines to follow actual roads
steps = directions_result[0]['legs'][0]['steps']
full_path = []

for step in steps:
    step_polyline = step['polyline']['points']
    points = polyline.decode(step_polyline)
    full_path.extend(points)

# Separate latitudes and longitudes
route_lats, route_lons = zip(*full_path)

# Plot on Google Map
gmap = gmplot.GoogleMapPlotter(route_lats[0], route_lons[0], 12, apikey=API_KEY)
gmap.plot(route_lats, route_lons, 'blue', edge_width=4)

# Optional: Add start and end markers
gmap.marker(route_lats[0], route_lons[0], color='green', title="Start")
gmap.marker(route_lats[-1], route_lons[-1], color='red', title="End")

# Save the map to HTML
gmap.draw("truck_route_by_coordinates.html")
