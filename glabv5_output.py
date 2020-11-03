#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import utm


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


def make_datetime(year: float, doy: float, sod: float) -> dt.datetime:
    """
    converts the YYYY, DoY and Time to a datetime
    """
    return dt.datetime.strptime('{:d} {:d} {!s}'.format(int(year), int(doy), pd.to_datetime(sod, unit='s', errors='coerce').time()), '%Y %j %H:%M:%S')



ECEF = ['X', 'Y', 'Z']
DeltaECEF = ['DeltaX', 'DeltaY', 'DeltaZ']
dECEF = ['dX', 'dY', 'dZ']
LLH = ['lat', 'lon', 'ellh']
DeltaNEU = ['DeltaN', 'DeltaE', 'DeltaU']
dNEU = ['dN', 'dE', 'dU']
XDOP = ['GDOP', 'PDOP', 'TDOP', 'HDOP', 'VDOP']
Tropo = ['inc', 'exc', 'err']
UTM = ['UTM.N', 'UTM.E', 'Zone']

col_names = ['OUTPUT', 'year', 'doy', 'sod', 'convergence'] +  ECEF + DeltaECEF + dECEF + LLH + DeltaNEU + dNEU + XDOP + Tropo + ['#SVs', 'ProcMode']
# print(col_names)
# sys.exit(5)

# Get name of gLAB v2 processed OUTPUT only file
cvs_output_name = input("Enter name of CSV glab v5 MESSAGE OUTPUT file: ")

try:
    df_output = pd.read_csv(cvs_output_name, names=col_names, header=0, delim_whitespace=True)

    # remove columns
    df_output.drop('OUTPUT', axis=1, inplace=True)
    df_output.drop(DeltaECEF, axis=1, inplace=True)
    df_output.drop(DeltaNEU, axis=1, inplace=True)

    # add datetime column
    df_output['DT'] = df_output.apply(lambda x: make_datetime(x['year'], x['doy'], x['sod']), axis=1)

    # add UTM coordinates
    df_output['UTM.E'], df_output['UTM.N'], Zone, Letter = utm.from_latlon(df_output['lat'].to_numpy(), df_output['lon'].to_numpy())
    df_output['Zone'] = '{!s}{!s}'.format(Zone, Letter)

    print(df_output.head(n=10))
    print(df_output.tail(n=10))
except IOError:
    print('Could not find {file:s}.'.format(cvs_output_name))
    sys.exit(1)

# List unique values in the ProcMode column "0 -> SPP, 1 -> PPP, 2 -> SBAS, 3 -> DGNSS"
dProcModes = {0: 'SPP', 1: 'PPP', 2: 'SBAS', 3: 'DGNSS'}
# proc_modes = df_output.ProcMode.unique()
print('Processing modes observed:')
proc_modes = df_output['ProcMode'].value_counts()

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

    for utmcrd, dcrd in zip(UTM[:2], dNEU[:2]):
        dwavg[utmcrd] = wavg(group=df_output.loc[mode_idx], avg_name=utmcrd, weight_name=dcrd)
        dstddev[dcrd] = stddev(df_output.loc[mode_idx][dcrd], dwavg[(utmcrd)])

    for i, (ecef, decef, llh, dllh, utmcrd, dcrd) in enumerate(zip(ECEF, dECEF, LLH, dNEU, UTM, dNEU)):
        if i < 2:
            print('      {ecef:s} = {wavg:13.3f} +-{stddev:.3f}   {llh:7s} = {crd:15.9f} +-{ecart:.3f}   {utm:s} = {utmavg:13.3f} +- {utmdev:.3f}'.format(ecef=ecef, wavg=dwavg[ecef], stddev=dstddev[decef], llh=llh, crd=dwavg[llh], ecart=dstddev[dllh], utm=utmcrd, utmavg=dwavg[utmcrd], utmdev=dstddev[dcrd]))
        else:
            print('      {ecef:s} = {wavg:13.3f} +-{stddev:.3f}   {llh:7s} = {crd:15.3f} +-{ecart:.3f}'.format(ecef=ecef, wavg=dwavg[ecef], stddev=dstddev[decef], llh=llh, crd=dwavg[llh], ecart=dstddev[dllh]))

    # create columns for difference wrt average UTM values
    for crd in UTM[:2] + ['ellh']:
        df_output['d{:s}'.format(crd)] = df_output[crd] - dwavg[crd]

    # plot the results
    plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    plt.suptitle('Processing mode: {mode:s} (#obs:{obs:d})'.format(mode=dProcModes[mode], obs=count))

    # plot the coordinates UTM/ellh (difference with weighted average) vs time
    ax1.set_title('UTM coordinates')
    ax1.set_xlabel('')
    ax1.set_ylabel('Coordinate difference [m]')
    df_output.loc[mode_idx].plot(x='DT', y=['dUTM.N', 'dUTM.E', 'dellh'], ax=ax1)

    # plot the XDOP & #SVs vs DT
    ax2.set_title('XDOP & #SVs')
    ax2.set_xlabel('Date-Time')
    ax2.set_ylabel('xDOP or #SVs [-]')
    df_output.loc[mode_idx].plot(x="DT", y=XDOP, ax=ax2)

    # plot number of SV on second y-axis
    # ax2v = ax2.twinx()
    # ax2v.set_ylim([0, 12])
    # ax2v.set_ylabel('#SVs [-]')  # , fontsize='large', color='grey')
    ax2.fill_between(x=df_output.loc[mode_idx]['DT'].values, y1=0, y2=df_output.loc[mode_idx]['#SVs'].values, alpha=0.15, linestyle='-', linewidth=3, color='green', label='#SVs', interpolate=False)
    # ax2v.fill_between(df_output.loc[mode_idx]['DT'].values, 0, df_output.loc[mode_idx]['#SVs'].values, facecolor='#0079a3', alpha=0.4)

    # display / save the plot
    plt.show()

    # save the plot in subdir png of GNSSSystem
    png_name = '{out:s}-{mode:s}.png'.format(out=cvs_output_name.replace('.', '-'), mode=dProcModes[mode])
    fig.savefig(png_name, dpi=fig.dpi)
