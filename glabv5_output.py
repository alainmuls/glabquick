#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd


def wavg(group: dict, avg_name: str, weight_name: str) -> float:
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    coordinate = group[avg_name]
    invVariance = 1 / np.square(group[weight_name])

    try:
        return (coordinate * invVariance).sum() / invVariance.sum()
    except ZeroDivisionError:
        return coordinate.mean()


def stddev(crd: pd.Series, avgCrd: float) -> float:
    """
    stddev calculates the standard deviation of series
    """
    dCrd = crd.subtract(avgCrd)

    return dCrd.std()


ECEF = ['Rx_X', 'Rx_Y', 'Rx_Z']
DeltaECEF = ['Rx_DeltaX', 'Rx_DeltaY', 'Rx_DeltaZ']
dECEF = ['Rx_dX', 'Rx_dY', 'Rx_dZ']
LLH = ['Rx_lat', 'Rx_lon', 'Rx_ellh']
DeltaNEU = ['Rx_DeltaN', 'Rx_DeltaE', 'Rx_DeltaU']
dNEU = ['Rx_dN', 'Rx_dE', 'Rx_dU']
XDOP = ['GDOP', 'PDOP', 'TDOP', 'HDOP', 'VDOP']
Tropo = ['inc', 'exc', 'err']

col_names = ['OUTPUT', 'Year', 'Doy', 'sod', 'convergence'] +  ECEF + DeltaECEF + dECEF + LLH + DeltaNEU + dNEU + XDOP + Tropo + ['#SVs', 'ProcMode']
# print(col_names)
# sys.exit(5)

# Get name of gLAB v2 processed OUTPUT only file
cvs_output_name = input("Enter name of CSV glab v5 MESSAGE OUTPUT file: ")

try:
    df_output = pd.read_csv(cvs_output_name, names=col_names, header=0, delim_whitespace=True)
except IOError:
    print('Could not find {file:s}.'.format(cvs_output_name))
    sys.exit(1)

# remove column 'OUTPUT'
df_output.drop(col_names[0], axis=1, inplace=True)

# print(df_output[DeltaNEU + dNEU])

# List unique values in the ProcMode column "0 -> SPP, 1 -> PPP, 2 -> SBAS, 3 -> DGNSS"
dProcModes = {0: 'SPP', 1: 'PPP', 2: 'SBAS', 3: 'DGNSS'}
# proc_modes = df_output.ProcMode.unique()
print('Processing modes observed:')
proc_modes = df_output[col_names[-1]].value_counts()

# calculate the weighted average of cartesian, geodetic and NEU data for each observed mode
for mode, count in proc_modes.iteritems():
    print('   Mode {mode:5s}: {epochs:7d} epochs'.format(mode=dProcModes[mode], epochs=count))

    # get df-indices for rows corresponding to the selected mode
    mode_idx = df_output.index[df_output['ProcMode'] == mode]

    # store the waighted averages in a dict
    dwavg = {}
    dstddev = {}

    for ecef, decef in zip(ECEF, dECEF):
        dwavg[ecef] = wavg(group=df_output.loc[mode_idx], avg_name=ecef, weight_name=decef)
        dstddev[decef] = stddev(df_output.loc[mode_idx][decef], dwavg[(ecef)])

    for llh, dneu in zip(LLH, dNEU):
        dwavg[llh] = wavg(group=df_output.loc[mode_idx], avg_name=llh, weight_name=dneu)
        dstddev[dneu] = stddev(df_output.loc[mode_idx][dneu], dwavg[(llh)])

    for i, (ecef, decef, llh, dneu) in enumerate(zip(ECEF, dECEF, LLH, dNEU)):
        if i < 2:
            print('      {ecef:s} = {wavg:13.3f} +-{stddev:.3f}   {llh:7s} = {crd:15.9f} +-{ecart:.3f}'.format(ecef=ecef, wavg=dwavg[ecef], stddev=dstddev[decef], llh=llh, crd=dwavg[llh], ecart=dstddev[dneu]))
        else:
            print('      {ecef:s} = {wavg:13.3f} +-{stddev:.3f}   {llh:7s} = {crd:15.3f} +-{ecart:.3f}'.format(ecef=ecef, wavg=dwavg[ecef], stddev=dstddev[decef], llh=llh, crd=dwavg[llh], ecart=dstddev[dneu]))
