import requests
import json

from src.coordinate import Coordinate
from src.delivery import Delivery


class Depot:
    def __init__(self, json_url: str):
        self.json_url = json_url
        self.raw_data = self.get_raw_data()
        self.name = self.raw_data["name"]
        self.region = self.raw_data["region"]
        self.origin = Coordinate(str(self.raw_data["origin"]["lat"]), str(self.raw_data["origin"]["lng"]))
        self.vehicle_capacity = self.raw_data["vehicle_capacity"]
        self.deliveries = [Delivery(self.raw_data["deliveries"][i], i) for i in range(len(self.raw_data["deliveries"]))]

    def get_raw_data(self):
        r = requests.get(self.json_url, allow_redirects=True)
        return r.json()

    def f(self, route: list, save: bool = False):
        # construct string for route
        route_list = [f"{self.origin.lng},{self.origin.lat}"]
        weight = 0
        for delivery in route:
            weight += delivery.size
            if weight <= self.vehicle_capacity:
                route_list.append(f"{delivery.point.lng},{delivery.point.lat}")
            else:
                route_list.append(f"{self.origin.lng},{self.origin.lat}")
                route_list.append(f"{delivery.point.lng},{delivery.point.lat}")
                weight = 0

        # last return to origin
        route_list.append(f"{self.origin.lng},{self.origin.lat}")

        # set query string for API request
        query = f"https://router.project-osrm.org/route/v1/driving/{';'.join(route_list)}?overview=false"

        # get the response
        response = requests.get(query)

        # extract json data
        data = response.json()

        if save:
            waypoints = data["waypoints"]
            origin = waypoints[0]['location']
            geometrys = [waypoints[i]['location'] for i in range(len(waypoints))]
            indexs_origin = [i for i in range(len(geometrys)) if geometrys[i] == origin]
            for i in range(len(indexs_origin)):
                waypoints[indexs_origin[i]]["hint"] = 0
            indexs_hint = [i for i in range(len(waypoints)) if waypoints[i]["hint"] != 0]
            for i in range(len(indexs_hint)):
                waypoints[indexs_hint[i]]["hint"] = route[i].id

            with open('results/best_route_cvrp.json', 'w') as outfile:
                json.dump(data, outfile)

        return data["routes"][0]["distance"]

    def __str__(self):
        return f"Name:\t\t\t\t{self.name}\n" \
               f"Region:\t\t\t\t{self.region}\n" \
               f"Origin:\t\t\t\t{self.origin}\n" \
               f"Vehicle Capacity:\t{self.vehicle_capacity}\n" \
               f"No. of deliveries:\t{len(self.deliveries)}"
