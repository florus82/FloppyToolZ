from FloppyToolZ.Funci import *
from scipy import optimize
import joblib
from joblib import Parallel, Delayed

# double logistic function
def funci(x, p1, p2, p3, p4, p5, p6):
    return p1 +p2 * ((1 / (1 + np.exp((p3 - x) / p4))) - (1 / (1 + np.exp((p5 - x) / p6))))
# scale MODIS dates betwee 1 and 365/366
def ModTimtoInt(timecodelist):
    leap_years = [str(i) for i in range(1960, 2024, 4)]
    if str(timecodelist[0])[0:4] in leap_years or str(timecodelist[-1])[0:4] in leap_years:
        lp = 366
    else:
        lp = 365

    jdays1 = [int(str(d)[4:7]) for d in timecodelist if str(d)[3] == str(timecodelist[0])[3]]
    jdays2 = [int(str(d)[4:7])+lp for d in timecodelist if str(d)[3] != str(timecodelist[0])[3]]
    jdays  = jdays1 + jdays2
    jdays  = [i-(jdays[0]-1) for i in jdays]
    return [jdays, [i for i in range(1,lp+1, 1)]]
# get sav files for runs with min/max/median r2 performance
def getMinMaxMedianSAV(iterCSV, savfileConti):
    it_min = np.where(iterCSV['r2'] == np.min(iterCSV['r2']))[0][0]
    it_max = np.where(iterCSV['r2'] == np.max(iterCSV['r2']))[0][0]
    it_median = np.random.choice(np.asarray(np.where(round(iterCSV['r2'], 2) == round(np.median(iterCSV['r2']), 2)))[0])

    keys = [fil.split('_')[-1].split('.')[0] for fil in savfileConti]
    vals = [fil for fil in savfileConti]
    savi = dict(zip(keys, vals))

    sav_min    = ['Min_r2_' + savi[str(it_min)].split('/')[-1].split('.')[0], joblib.load(savi[str(it_min)])]
    sav_max    = ['Max_r2_' + savi[str(it_max)].split('/')[-1].split('.')[0], joblib.load(savi[str(it_max)])]
    sav_median = ['Median_r2_' + savi[str(it_median)].split('/')[-1].split('.')[0], joblib.load(savi[str(it_median)])]

    return [sav_min, sav_max, sav_median]
# erase NA-frame from numpy arrays
def shrinkNAframe(na_array):
    cols = na_array.shape[1]
    rows = na_array.shape[0]

    c = [col for col in range(cols) if np.nansum(np.isnan(na_array[:,col]))==rows] # passes i if all rows in col are nan
    r = [row for row in range(rows) if np.nansum(np.isnan(na_array[row,:]))==cols] # passes i if all cols in row are nan

    if (len(c)==0):
        xoff = 0
        xcount = cols
    elif c[0] == 0 and c[-1] != (cols-1): # that means the first column contains only nans and the last one doesn't --> that necessitates to cut of left-side of array only
        xoff = len(c)
        xcount = cols - xoff
    elif c[0] != 0 and c[-1] == (cols-1): # first column contains values and last one doesn't --> only the right side of the array needs to be cut off
        xoff = 0
        xcount = cols - len(c)
    elif c[0] ==0 and c[-1] == (cols-1): # first and last column contain nans --> only middle needs to be read as array
        seq = [i for i in range(len(c))]
        xoff = [i for i in range(len(c)) if seq[i] != c[i]]
        xcount = cols - len(c)

    if (len(r)==0):
        yoff = 0
        ycount = rows
    elif r[0] == 0 and r[-1] != (rows-1):
        yoff = len(r)
        ycount = rows - yoff
    elif r[0] != 0 and r[-1] == (rows-1):
        yoff = 0
        ycount = rows - len(r)
    elif r[0] ==0 and r[-1] == (rows-1):
        seq = [i for i in range(len(r))]
        yoff = [i for i in range(len(r)) if seq[i] != r[i]]
        ycount = rows - len(r)

    return [xoff, yoff, xcount, ycount]
# create growing seasons and frowing fits (jul-jul) combos
def time_seq(start_day, start_month, start_year, end_day, end_month, end_year):
    start = int(str(start_year) + str(getJulianDay(start_day, start_month, start_year)))
    end = int(str(end_year) + str(getJulianDay(end_day, end_month, end_year)))

    return [start, end]


def PixelBreaker_BoneStorm(x):
    # create seasonal container
    SoS_conti = []
    EoS_conti = []
    SeasMax_conti = []
    SeasMin_conti = []
    SeasInt_conti = []
    SeasLen_conti = []
    SeasAmp_conti = []

    all_len = sum([len(ii) for ii in timelini]) # this equal to the number of scenes for all timeframes
    counter_start = 0
    counter_end   = 0

    for seas in timelini:
        # get a timeframe subset
        counter_end += len(seas)
        sub_ndvi = x[counter_start + (all_len * 0) : counter_end + (all_len * 0)] # as NDVI,EVI,NBR scenes are stacked
        sub_evi  = x[counter_start + (all_len * 1) : counter_end + (all_len * 0)] # this way
        sub_nbr  = x[counter_start + (all_len * 2) : counter_end + (all_len * 0)]
        subby    = [sub_ndvi, sub_evi, sub_nbr]
        counter_start += len(seas)

        # ################################## fit function and derive seasonal parameter for each VI subset
        for sub in subby:
            m = sub
            # mask nan from m; mask also the values from the time object, which is passed on to optimize.curve_fit
            doys  = np.asarray(seas)[np.logical_not(np.isnan(m))]
            vivas = m[np.logical_not(np.isnan(m))]

            if len(doys) == 0 or len(vivas) == 0:
                SoS_conti.append(np.nan)
                EoS_conti.append(np.nan)
                SeasMax_conti.append(np.nan)
                SeasMin_conti.append(np.nan)
                SeasInt_conti.append(np.nan)
                SeasLen_conti.append(np.nan)
                SeasAmp_conti.append(np.nan)

            else:
                popt, pcov = optimize.curve_fit(funci,
                                                doys, vivas, p0=[0.1023, 0.8802, 108.2, 7.596, 311.4, 7.473],
                                                maxfev=100000000)
                pred = funci(dummi[iti], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
                dev1 = np.diff(pred)
                SoS     = np.argmax(dev1) + 1
                EoS     = np.argmin(dev1) + 1
                SeasMax = round(max(pred), 2)
                SeasMin = round(min(pred), 2)
                SeasInt = round(np.trapz(funci(np.arange(SoS, EoS + 1, 1),
                                               popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])), 2)
                SeasLen = EoS - SoS
                SeasAmp = SeasMax - SeasMin

                SoS_conti.append(SoS)
                EoS_conti.append(EoS)
                SeasMax_conti.append(SeasMax)
                SeasMin_conti.append(SeasMin)
                SeasInt_conti.append(SeasInt)
                SeasLen_conti.append(SeasLen)
                SeasAmp_conti.append(SeasAmp)

    resi = np.asarray([np.median(SoS_conti), np.median(EoS_conti), np.median(SeasMax_conti), np.median(SeasMin_conti),
                       np.median(SeasInt_conti), np.median(SeasLen_conti), np.median(SeasAmp_conti)])
    return resi

def PixelSmasher(tile_array, pixelbreakerFunci, timelini, dummi, storpath):
    timelini = timelini + timelini + timelini # as there are three stacked VIs
    dummi    = dummi + dummi + dummi # as there are three stacked VIs
    SP_arr3d = np.apply_along_axis(pixelbreakerFunci,2, tile_array)
    joblib.dump(SP_arr2d, storpath)
    return print('Tile ' + storpath.split('/')[-1].split('.')[0] + ' smashed')