#!/usr/bin/env python

from geographiclib.geodesic import Geodesic

Geo = Geodesic.WGS84

lat1, lon1 = -7.313341167341917, 10.65583081724002
lat2, lon2 = -7.313340663909912, 10.655830383300781

lat1 = float(input('start point Latitude: '))
lon1 = float(input('start point Longitude: '))
lat2 = float(input('end point Latitude: '))
lon2 = float(input('end point Longitude: '))

d = Geo.Inverse(lat1, lon1,  lat2, lon2)

# print(d)

print('From point ({lat1:.9f}, {lon1:.9f}) to ({lat2:.9f}, {lon2:.9f})'.format(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2))
print('   Distance = {dist:13.3f}m'.format(dist=d['s12']))
print('    Azimuth = {azim:13.6f}d'.format(azim=d['azi1']))
