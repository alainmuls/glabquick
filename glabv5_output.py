#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import colors as mpcolors
import datetime as dt
import utm
from geoidheight import geoid


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


def geoid_undulation(lat: float, lon: float) -> float:
    """
    calculates the geoid undulation N at specified position
    """
    return gh.get(lat, lon)


def make_rgb_transparent(rgb, bg_rgb, alpha):
    """
    make a color transparent
    """
    return [alpha * c1 + (1 - alpha) * c2
            for (c1, c2) in zip(rgb, bg_rgb)]


def table_header_str(col_headers: list) -> str:
    """
    table_header_str creates a hedaer line t
    """
    ret_str = ''
    for header in col_headers:
        ret_str += '| __{hdr:s}__ '.format(hdr=header)
    ret_str += '|\n'
    ret_str += '| ---: ' * len(col_headers)
    ret_str += '|\n'

    return ret_str


def table_row_str(df: pd.DataFrame, columns: list, idx_start: int, idx_end:int, width: list, prec: list) -> str:
    """
    table_row_str creates a row into the table
    """
    print('{:{width}.{prec}f}'.format(2.7182, width=15, prec=6))
    ret_str = ''

    print('\n\ncolumns = {!s}'.format(columns))
    for idx in idx_start:
        for mlist, mwidth, mprec in zip(columns, width, prec):
            # print('mlist {!s}'.format(mlist))
            # print('mwidth {!s}'.format(mwidth))
            # print('mprec {!s}'.format(mprec))
            for col in mlist:
                ret_str += '| {:{width}.{prec}f} '.format(df[col][idx], width=mwidth, prec=mprec)
        ret_str += '|\n'

    return ret_str


def markdown_report(mode: str):
    """
    create a markdown report for the selected mode
    """
    md_name = '{out:s}-{mode:s}.markdown'.format(out=cvs_output_name.replace('.', '-'), mode=dProcModes[mode])
    with open(md_name, 'w') as fout:
        fout.write(table_header_str(DATES + ECEF + dECEF))

        # determine indices of first / last 10 rows
        start_idx = df_pos.loc[mode_idx][:10].index
        end_idx = df_pos.loc[mode_idx][-10:].index

        # write cartesian information
        for i in start_idx:
            for col in DATES:
                fout.write('| {!s} '.format(df_pos.loc[mode_idx][col][i]))
            for col in ECEF + dECEF:
                fout.write('| {:.4f} '.format(df_pos.loc[mode_idx][col][i]))
            fout.write('|\n')
        fout.write('| ---: ' * 9)
        fout.write('|\n')

        print('end_idx = {!s}'.format(end_idx))
        for i in start_idx:
            for col in DATES:
                fout.write('| {!s} '.format(df_pos.loc[mode_idx][col][i]))
            for col in ECEF + dECEF:
                fout.write('| {:.4f} '.format(df_pos.loc[mode_idx][col][i]))
            fout.write('|\n')
        # fout.write('| ---: ' * 9)
        fout.write('\n')

        print(table_row_str(df=df_pos.loc[mode_idx], columns=[ECEF, dECEF], idx_start=start_idx, idx_end=end_idx, width=[0, 0], prec=[4, 3]))


# font for th elegend
legend_font = font_manager.FontProperties(family='monospace', weight='bold', style='normal', size='small')

DATES = ['year', 'doy', 'sod', 'DT']
ECEF = ['X', 'Y', 'Z']
DeltaECEF = ['DeltaX', 'DeltaY', 'DeltaZ']
dECEF = ['dX', 'dY', 'dZ']
LLH = ['lat', 'lon', 'ellh']
DeltaNEU = ['DeltaN', 'DeltaE', 'DeltaU']
dNEU = ['dN', 'dE', 'dU']
XDOP = ['GDOP', 'PDOP', 'TDOP', 'HDOP', 'VDOP']
Tropo = ['inc', 'exc', 'err']
UTM = ['UTM.N', 'UTM.E', 'ortoH']
dUTM = ['Delta_UTM.N', 'Delta_UTM.E', 'Delta_ortoH']

col_names = ['OUTPUT', 'year', 'doy', 'sod', 'convergence'] +  ECEF + DeltaECEF + dECEF + LLH + DeltaNEU + dNEU + XDOP + Tropo + ['#SVs', 'ProcMode']

utm_colors = ['tab:green', 'tab:blue', 'tab:brown']
dop_colors = ['tab:green', 'tab:orange', 'tab:blue', 'tab:purple', 'tab:red', 'tab:brown']

rgb_colors = [mpcolors.colorConverter.to_rgb(color) for color in utm_colors]
rgb_error_colors = [make_rgb_transparent(rgb, (1, 1, 1), 0.4 - i * 0.1) for i, rgb in enumerate(rgb_colors)]

# initialise the geodheight class
gh = geoid.GeoidHeight('/usr/share/GeographicLib/geoids/egm2008-1.pgm')

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

    # df_pos['OrtoH'] = df_pos.apply(geoid_undulation(df_pos['lat'], df_pos['lon']))
    df_pos['ortoH'] = [ellh - geoid_undulation(lat, lon) for lat, lon, ellh in zip(df_pos['lat'], df_pos['lon'], df_pos['ellh'])]

    print(df_pos.head(n=10))
    print(df_pos.tail(n=10))
except IOError:
    print('Could not find {file:s}.'.format(cvs_output_name))
    sys.exit(1)

# List unique values in the ProcMode column "0 -> SPP, 1 -> PPP, 2 -> SBAS, 3 -> DGNSS"
dProcModes = {0: 'SPP', 1: 'PPP', 2: 'SBAS', 3: 'DGNSS'}
# proc_modes = df_pos.ProcMode.unique()
print('\nProcessing modes observed:')
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

    for utm, dcrd in zip(UTM, dNEU):
        dwavg[utm] = wavg(group=df_pos.loc[mode_idx], avg_name=utm, weight_name=dcrd)
        dstddev[utm] = stddev(df_pos.loc[mode_idx][dcrd], dwavg[utm])

    # print the results for this mode
    for i, (ecef, llh, utm) in enumerate(zip(ECEF, LLH, UTM)):
        print('{ecef:>9s} = {cart:13.3f} +-{cartSD:.3f}   {geographic:>7s} = {geod:15.9f} +-{geodSD:.3f}   {utm:>7s} = {utmavg:13.3f} +- {utmSD:.3f}'.format(ecef=ecef, cart=dwavg[ecef], cartSD=dstddev[ecef], geographic=llh, geod=dwavg[llh], geodSD=dstddev[llh], utm=utm, utmavg=dwavg[utm], utmSD=dstddev[utm]))

    # create columns for difference wrt average UTM values used for plotting
    df_tmp = pd.DataFrame()
    df_tmp['DT'] = df_pos.loc[mode_idx]['DT']
    for crd in UTM:
        df_tmp['Delta_{:s}'.format(crd)] = df_pos.loc[mode_idx][crd] - dwavg[crd]
    df_tmp[dNEU] = df_pos.loc[mode_idx][dNEU]
    df_tmp[XDOP] = df_pos.loc[mode_idx][XDOP]
    df_tmp["#SVs"] = df_pos.loc[mode_idx]["#SVs"]

    # print('df_tmp =\n{!s}'.format(df_tmp))

    # plot the results
    plt.style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    plt.suptitle('Processing mode: {mode:s} (#obs:{obs:d})'.format(mode=dProcModes[mode], obs=count))

    # PLOT THE COORDINATES UTM/ELLH (DIFFERENCE WITH WEIGHTED AVERAGE) VS TIME
    for i, (crd, crd_sd, dutm, crd_color, error_color) in enumerate(zip(UTM, dNEU, dUTM, rgb_colors, rgb_error_colors)):

        ax1.errorbar(x=df_tmp['DT'], y=df_tmp[dutm], yerr=df_tmp[crd_sd], linestyle='none', capthick=1, markersize=1, fmt='o', color=crd_color, ecolor=error_color, elinewidth=3, capsize=0, label=crd)

        crd_txt = r'{crd:s}: {wavg:14.3f} $\pm$ {sdcrd:.3f}'.format(crd=crd, wavg=dwavg[crd], sdcrd=dstddev[crd])

        ax1.annotate(crd_txt, xy=(1, 1), xycoords='axes fraction', xytext=(0, 35 - i * 15), textcoords='offset pixels', horizontalalignment='right', verticalalignment='bottom', fontweight='bold', fontsize='small', fontname='Courier', family='monospace')

    # name the titles and so on
    ax1.set_title('UTM & Orto H')
    ax1.set_xlabel('')
    ax1.set_ylabel('Coordinate difference [m]')
    ax1.legend(prop=legend_font, bbox_to_anchor=(1.02, 1), loc='upper left', markerscale=3)

    # PLOT THE XDOP & #SVS VS DT
    # df_tmp.plot(x='DT', y=XDOP, ax=ax2)
    for xdop, xdop_color in zip(XDOP, dop_colors):
        ax2.plot(df_tmp['DT'], df_tmp[xdop], label=xdop, color=xdop_color)

    # plot number of SV on second y-axis
    ax2v = ax2.twinx()
    ax2v.set_ylim([0, 12])
    ax2v.set_ylabel('#SVs [-]')  # , fontsize='large', color='grey')
    ax2v.fill_between(x=df_tmp['DT'].values, y1=0, y2=df_tmp['#SVs'].values, alpha=0.25, linestyle='-', linewidth=3, color='grey', label='#SVs', interpolate=False)
    # ax2v.fill_between(df_pos.loc[mode_idx]['DT'].values, 0, df_pos.loc[mode_idx]['#SVs'].values, facecolor='#0079a3', alpha=0.4)

    # name the axis
    ax2.set_title('XDOP & #SVs')
    ax2.set_xlabel('Date-Time')
    ax2.set_ylabel('xDOP [-]')
    ax2.legend(prop=legend_font, bbox_to_anchor=(1.02, 1), loc='upper left')

    # copyright this
    ax2.annotate(r'$\copyright$ Alain Muls (alain.muls@mil.be)', xy=(1, 0), xycoords='axes fraction', xytext=(0, -30), textcoords='offset pixels', horizontalalignment='right', verticalalignment='top', weight='strong', fontsize='small', family='monospace')

    # display / save the plot
    plt.show()

    # create markdown report
    markdown_report(mode=mode)

    # save the plot in subdir png of GNSSSystem
    png_name = '{out:s}-{mode:s}.png'.format(out=cvs_output_name.replace('.', '-'), mode=dProcModes[mode])
    fig.savefig(png_name, dpi=fig.dpi)
