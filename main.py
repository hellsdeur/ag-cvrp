import json
import folium

from src.depot import Depot
from src.ga import GeneticAlgorithm, MeanBehavior

depot = Depot("https://raw.githubusercontent.com/ronaldcuri/meta-cvrpy/main/data/cvrp_castanhal.json")

ga = GeneticAlgorithm(
    f=depot.f,
    individuals=depot.deliveries,
    k=5,
    minmax="min",
    selection="steady"
)

mb = MeanBehavior(
    ga=ga,
    n_run=1,
    mutation_rate=0.03,
    generations=1
)
mb.process()
description = mb.describe()
mb.plot(description)

best = mb.run_list[0].gbest_list[-1]
for i in range(1, len(mb.run_list)):
    if mb.run_list[i].gbest_list[-1].fitness < best.fitness:
        best = mb.run_list[i].gbest_list[-1]

depot.f(best.chromosome, save=True)

with open('results/best_route_cvrp.json', 'r') as outfile:
    data = json.load(outfile)
    #pontos de rota
    waypoints = data['waypoints']
    hints = [waypoints[i]['hint'] for i in range(len(waypoints))]
    geometrys = [waypoints[i]['location'] for i in range(len(waypoints))]
    geometrys = [[geometrys[i][1], geometrys[i][0]] for i in range(len(geometrys))]
    #distâncias entre cada ponto de rota
    legs = data['routes'][0]['legs']
    distances_legs = [legs[i]['distance'] for i in range(len(legs))]
#pontos da rota em que são o depósito
indexs_origin = [i for i in range(len(waypoints)) if waypoints[i]['hint'] == 0]


print(f'Quantidades de rotas calculadas: {len(indexs_origin)-1}')
for i in range(len(indexs_origin)-1):
    peso = 0
    rota = ''
    for hint in hints[indexs_origin[i]+1:indexs_origin[i+1]]:
        peso += depot.deliveries[hint-1].size
        rota += f' -> {hint}'
    print(f'\nRota {i+1}: 0' + rota + f' -> 0')
    print(f'Peso máximo atingido: {peso}kg')
    distance_route = sum(distances_legs[indexs_origin[i]:indexs_origin[i+1]])
    print(f'Distância: {distance_route:.2f}km')

#intanciando mapa
tiles = ['OpenStreetMap','CartoDB positron','Stamen toner','Stamen Terrain']
mapa = folium.Map(location=geometrys[0], zoom_start=12, tiles=tiles[1])

#marcador de cada ponto da rota
for i in range(len(geometrys)):
    folium.Marker(geometrys[i], popup=f'Ponto {hints[i]}').add_to(mapa)

#marcador do depósito, ponto incial/final
folium.CircleMarker(geometrys[0], radius=10, popup='Depósito', color='red', fill=True).add_to(mapa)

color = ['red', 'blue', 'green', 'yellow', 'black', 'purple', 'orange', 'brown', 'pink', 'gray']
#criar linhas poligonais entre as rotas
for i in range(len(indexs_origin)-1):
    folium.PolyLine(geometrys[indexs_origin[i]:indexs_origin[i+1]+1], color=color[i], weight=2, opacity=0.7).add_to(mapa)

#salvar mapa
mapa.save('results/mapa_cvrp.html')
