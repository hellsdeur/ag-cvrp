import pandas as pd
import requests
import json
import numpy as np

from src.coordinate import Coordinate
from src.delivery import Delivery


class Depot:
    def __init__(self, json_url: str, distances_path: str = None):
        self.json_url = json_url
        self.raw_data = self.get_raw_data()
        self.name = self.raw_data["name"]
        self.region = self.raw_data["region"]
        self.origin = Coordinate(str(self.raw_data["origin"]["lat"]), str(self.raw_data["origin"]["lng"]))
        self.vehicle_capacity = self.raw_data["vehicle_capacity"]
        self.deliveries = [Delivery(self.raw_data["deliveries"][i], i) for i in range(len(self.raw_data["deliveries"]))]
        if distances_path:
            self.distances_matrix = pd.read_csv(distances_path)
        else:
            self.distances_matrix = self.calculate_distances(save=True)

    def get_raw_data(self):
        r = requests.get(self.json_url, allow_redirects=True)
        return r.json()

    def f(self, route: list, save: bool = False):
        # route with weight condition
        weight = 0

        # fitness starts as the distance between origin and first point
        y = self.distances_matrix.iloc[0, route[0].id]

        for i in range(len(route)):
            delivery = route[i]
            weight += delivery.size

            if i < len(route)-1:
                if weight <= self.vehicle_capacity:
                    y += self.distances_matrix.iloc[route[i].id, route[i+1].id]
                else:
                    y += self.distances_matrix.iloc[route[i].id, 0]
                    y += self.distances_matrix.iloc[0, route[i+1].id]
                    weight = 0

        # last return to origin
        y += self.distances_matrix.iloc[route[len(route) - 1].id, 0]

        if save:
            route_list = [f"{self.origin.lng},{self.origin.lat}"]

            for delivery in route:
                weight += delivery.size

                if weight <= self.vehicle_capacity:
                    route_list.append(f"{delivery.point.lng},{delivery.point.lat}")
                else:
                    route_list.append(f"{self.origin.lng},{self.origin.lat}")
                    route_list.append(f"{delivery.point.lng},{delivery.point.lat}")
                    weight = 0

            # set query string for API request
            query = f"https://router.project-osrm.org/route/v1/driving/{';'.join(route_list)}?overview=false"

            # get the response
            response = requests.get(query)

            # extract json data
            data = response.json()

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

        return y

    def calculate_distances(self, save=False):
        coordinates = [self.origin]
        for delivery in self.deliveries:
            coordinates.append(delivery.point)

        n_coordinates = len(coordinates)
        distances = np.zeros((n_coordinates, n_coordinates))

        for i in range(n_coordinates):
            for j in range(n_coordinates):
                if i != j:
                    origin = f"{coordinates[i].lng},{coordinates[i].lat}"
                    destination = f"{coordinates[j].lng},{coordinates[j].lat}"
                    route_str = ";".join([origin, destination])
                    query = f"https://router.project-osrm.org/route/v1/driving/{route_str}?overview=false"
                    response = requests.get(query)
                    data = response.json()
                    distances[i][j] = data["routes"][0]["distance"]

        df = pd.DataFrame(distances)

        if save:
            df.to_csv('distances_matrix.csv', index=False)

        return df


    def __str__(self):
        return f"Name:\t\t\t\t{self.name}\n" \
               f"Region:\t\t\t\t{self.region}\n" \
               f"Origin:\t\t\t\t{self.origin}\n" \
               f"Vehicle Capacity:\t{self.vehicle_capacity}\n" \
               f"No. of deliveries:\t{len(self.deliveries)}"
