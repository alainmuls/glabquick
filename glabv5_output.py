#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
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


# font for th elegend
legend_font = font_manager.FontProperties(family='monospace', weight='bold', style='normal', size='small')

ECEF = ['X', 'Y', 'Z']
DeltaECEF = ['DeltaX', 'DeltaY', 'DeltaZ']
dECEF = ['dX', 'dY', 'dZ']
LLH = ['lat', 'lon', 'ellh']
DeltaNEU = ['DeltaN', 'DeltaE', 'DeltaU']
dNEU = ['dN', 'dE', 'dU']
XDOP = ['GDOP', 'PDOP', 'TDOP', 'HDOP', 'VDOP']
Tropo = ['inc', 'exc', 'err']
UTM = ['UTM.N', 'UTM.E', 'Zone']
dUTM = ['dUTM.N', 'dUTM.E', 'dEllH']

col_names = ['OUTPUT', 'year', 'doy', 'sod', 'convergence'] +  ECEF + DeltaECEF + dECEF + LLH + DeltaNEU + dNEU + XDOP + Tropo + ['#SVs', 'ProcMode']
# print(col_names)
# sys.exit(5)

# Get name of gLAB v2 processed OUTPUT only file
cvs_output_name = input("Enter name of CSV glab v5 MESSAGE OUTPUT file: ")

try:
    df_pos = pd.read_csv(cvs_output_name, names=col_names, header=0, delim_whitespace=True)

    # remove columns
    df_pos.drop('OUTPUT', axis=1, inplace=True)
    df_pos.drop(DeltaECEF, axis=1, inplace=True)
    df_pos.drop(DeltaNEU, axis=1, inplace=True)

    # add datetime column
    df_pos['DT'] = df_pos.apply(lambda x: make_datetime(x['year'], x['doy'], x['sod']), axis=1)

    # add UTM coordinates
    df_pos['UTM.E'], df_pos['UTM.N'], Zone, Letter = utm.from_latlon(df_pos['lat'].to_numpy(), df_pos['lon'].to_numpy())
    df_pos['Zone'] = '{!s}{!s}'.format(Zone, Letter)

    print(df_pos.head(n=10))
    print(df_pos.tail(n=10))
except IOError:
    print('Could not find {file:s}.'.format(cvs_output_name))
    sys.exit(1)

# List unique values in the ProcMode column "0 -> SPP, 1 -> PPP, 2 -> SBAS, 3 -> DGNSS"
dProcModes = {0: 'SPP', 1: 'PPP', 2: 'SBAS', 3: 'DGNSS'}
# proc_modes = df_pos.ProcMode.unique()
print('Processing modes observed:')
proc_modes = df_pos['ProcMode'].value_counts()

# calculate the weighted average of cartesian, geodetic and NEU data for each observed mode
for mode, count in proc_modes.iteritems():
    print('   Mode {mode:5s}: {epochs:7d} epochs'.format(mode=dProcModes[mode], epochs=count))

    # get df-indices for rows corresponding to the selected mode
    mode_idx = df_pos.index[df_pos['ProcMode'] == mode]

    # store the waighted averages in a dict
    dwavg = {}
    dstddev = {}

    for ecef, decef in zip(ECEF, dECEF):
        dwavg[ecef] = wavg(group=df_pos.loc[mode_idx], avg_name=ecef, weight_name=decef)
        dstddev[ecef] = stddev(df_pos.loc[mode_idx][decef], dwavg[ecef])

    for llh, dneu in zip(LLH, dNEU):
        dwavg[llh] = wavg(group=df_pos.loc[mode_idx], avg_name=llh, weight_name=dneu)
        dstddev[llh] = stddev(df_pos.loc[mode_idx][dneu], dwavg[llh])

    for utm_h, dcrd in zip(UTM[:2], dNEU[:2]):
        dwavg[utm_h] = wavg(group=df_pos.loc[mode_idx], avg_name=utm_h, weight_name=dcrd)
        dstddev[utm_h] = stddev(df_pos.loc[mode_idx][dcrd], dwavg[utm_h])

    for i, (ecef, llh, utm_h) in enumerate(zip(ECEF, LLH, UTM)):
        if i < 2:
            print('      {ecef:s} = {wavg:13.3f} +-{stddev:.3f}   {llh:7s} = {crd:15.9f} +-{ecart:.3f}   {utm:s} = {utmavg:13.3f} +- {utmdev:.3f}'.format(ecef=ecef, wavg=dwavg[ecef], stddev=dstddev[ecef], llh=llh, crd=dwavg[llh], ecart=dstddev[llh], utm=utm_h, utmavg=dwavg[utm_h], utmdev=dstddev[utm_h]))
        else:
            print('      {ecef:s} = {wavg:13.3f} +-{stddev:.3f}   {llh:7s} = {crd:15.3f} +-{ecart:.3f}'.format(ecef=ecef, wavg=dwavg[ecef], stddev=dstddev[ecef], llh=llh, crd=dwavg[llh], ecart=dstddev[llh]))

    # create columns for difference wrt average UTM values used for plotting
    df_tmp = pd.DataFrame()
    df_tmp['DT'] = df_pos.loc[mode_idx]['DT']
    for crd, dutm in zip(UTM[:2] + LLH[-1:], dUTM):
        print(crd)
        df_tmp[dutm] = df_pos.loc[mode_idx][crd] - dwavg[crd]
    df_tmp[dNEU] = df_pos.loc[mode_idx][dNEU]
    df_tmp[XDOP] = df_pos.loc[mode_idx][XDOP]
    df_tmp["#SVs"] = df_pos.loc[mode_idx]["#SVs"]

    print('df_tmp =\n{!s}'.format(df_tmp))

    # plot the results
    plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    plt.suptitle('Processing mode: {mode:s} (#obs:{obs:d})'.format(mode=dProcModes[mode], obs=count))

    # plot the coordinates UTM/ellh (difference with weighted average) vs time
    # df_pos.loc[mode_idx].plot(x='DT', y=['dUTM.N', 'dUTM.E', 'dellh'], ax=ax1)

    # name the titles and so on
    ax1.set_title('UTM & EllH')
    ax1.set_xlabel('')
    ax1.set_ylabel('Coordinate difference [m]')
    # ax1.legend(prop=legend_font, bbox_to_anchor=(1.01, 1), loc='upper left')

    # # annotate with average & stddev of coordinates
    # for _, crd in enumerate(UTM[:2] + LLH[-1:]):
    #     crd_txt = r'{crd:s}: {wavg:14.3f} $\pm$ {sdcrd:.3f}'.format(crd=crd, wavg=dwavg[crd], sdcrd=dstddev[crd])

    #     ax1.annotate(crd_txt, xy=(1, 1), xycoords='axes fraction', xytext=(0, 35 - i * 15), textcoords='offset pixels', horizontalalignment='right', verticalalignment='bottom', fontweight='bold', fontsize='small', fontname='Courier', family='monospace')

    for ii, (crd, crd_sd, dutm) in enumerate(zip(UTM[:2] + [LLH[-1]], dNEU, dUTM)):
        # ax1.errorbar(x=dfUTM.loc[idx]['DT'], y=dfCrd.loc[idx][crd], yerr=dfUTM.loc[idx][stdDev2Plot[i]], linestyle='None', fmt='o', ecolor=rgb_new, capthick=2, markersize=2, color=colors[key], label=value)
        # print('_' * 25)
        print("ii = {!s}".format(ii))
        print("crd = {:s}".format(crd))
        print("crd_sd = {:s}".format(crd_sd))
        print("dutm = {:s}".format(dutm))
        # print("df_pos.loc[mode_idx]['DT'][:5] = {!s}".format(df_pos.loc[mode_idx]['DT'][:5]))
        # print("df_tmp = {!s}".format(df_tmp))
        # print("df_tmp.loc[mode_idx] =\n{!s}".format(df_tmp.loc[mode_idx]))

        # print('#' * 25)
        # print("crd = {:s}".format(crd))
        # print("type(crd) = {!s}".format(type(crd)))
        # print("df_tmp.loc[mode_idx][dutm][:5] =\n{!s}".format(df_tmp.loc[mode_idx][dutm][:5]))

        # print("len(df_pos.loc[mode_idx]['DT']) = {!s}".format(len(df_pos.loc[mode_idx]['DT'])))
        # print("len(df_tmp.loc[mode_idx][dutm]) = {!s}".format(len(df_tmp.loc[mode_idx][dutm])))
        # print("len(df_pos.loc[mode_idx][crd_sd]) = {!s}".format(len(df_pos.loc[mode_idx][crd_sd])))
        # print('_' * 50)

        # df_tmp.loc[mode_idx].plot(x='DT', y=dutm, ax=ax1)

        # upper and lower coordinates
        # df_tmp['up'] = df_tmp.loc[mode_idx][dutm] + df_pos.loc[mode_idx][crd_sd]
        # df_tmp['down'] = df_tmp.loc[mode_idx][dutm] - df_pos.loc[mode_idx][crd_sd]
        # print(df_tmp.head(n=10))
        # print(df_tmp.tail(n=10))

        # print("len(df_tmp['DT']) = {!s}".format(len(df_tmp['DT'])))
        # print("len(df_tmp[dUTM])= {!s}".format(len(df_tmp[dUTM])))
        # print("len(df_tmp[crd_sd])= {!s}".format(len(df_tmp[crd_sd])))

        # ax1.errorbar(x=df_tmp.loc[mode_idx]['DT'], y=df_tmp.loc[mode_idx][dUTM], yerr=df_tmp.loc[mode_idx][crd_sd], linestyle='None', fmt='o', capthick=2, markersize=2, label=crd)

        # crd_txt = r'{crd:s}: {wavg:14.3f} $\pm$ {sdcrd:.3f}'.format(crd=crd, wavg=dwavg[crd], sdcrd=dstddev[crd])

        # ax1.annotate(crd_txt, xy=(1, 1), xycoords='axes fraction', xytext=(0, 35 - i * 15), textcoords='offset pixels', horizontalalignment='right', verticalalignment='bottom', fontweight='bold', fontsize='small', fontname='Courier', family='monospace')

        ax1.errorbar(x=df_tmp['DT'], y=df_tmp[dutm], yerr=df_tmp[crd_sd], linestyle='none', fmt='.', capthick=1, markersize=2)

    # sys.exit(6)


    # plot the XDOP & #SVs vs DT
    # df_tmp.plot(x='DT', y=XDOP, ax=ax2)
    for xdop in XDOP:
        ax2.plot(df_tmp['DT'], df_tmp[xdop])

    # plot number of SV on second y-axis
    # ax2v = ax2.twinx()
    # ax2v.set_ylim([0, 12])
    # ax2v.set_ylabel('#SVs [-]')  # , fontsize='large', color='grey')
    ax2.fill_between(x=df_tmp['DT'].values, y1=0, y2=df_tmp['#SVs'].values, alpha=0.15, linestyle='-', linewidth=3, color='green', label='#SVs', interpolate=False)
    # ax2v.fill_between(df_pos.loc[mode_idx]['DT'].values, 0, df_pos.loc[mode_idx]['#SVs'].values, facecolor='#0079a3', alpha=0.4)

    # name the axis
    ax2.set_title('XDOP & #SVs')
    ax2.set_xlabel('Date-Time')
    ax2.set_ylabel('xDOP or #SVs [-]')
    ax2.legend(prop=legend_font, bbox_to_anchor=(1.01, 1), loc='upper left')

    # copyright this
    ax2.annotate(r'$\copyright$ Alain Muls (alain.muls@mil.be)', xy=(1, 0), xycoords='axes fraction', xytext=(0, -30), textcoords='offset pixels', horizontalalignment='right', verticalalignment='top', weight='strong', fontsize='small', family='monospace')

    # display / save the plot
    plt.show()

    # save the plot in subdir png of GNSSSystem
    png_name = '{out:s}-{mode:s}.png'.format(out=cvs_output_name.replace('.', '-'), mode=dProcModes[mode])
    fig.savefig(png_name, dpi=fig.dpi)
