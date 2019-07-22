import os
import ogr, osr
import gdal
import sys
import numpy as np
import math
import zipfile
import struct
import datetime as dt
# import pandas as pd

def getFilelist(originpath, ftyp):
    files = os.listdir(originpath)
    out   = []
    for i in files:
        if i.split('.')[-1] in ftyp:
            if originpath.endswith('/'):
                out.append(originpath + i)
            else:
                out.append(originpath + '/' + i)
        # else:
        #     print("non-matching file - {} - found".format(i.split('.')[-1]))
    return out

def getJulianDay(day, month, year):
    leap_years = [i for i in range(1960, 2024, 4)]
    if year in leap_years:
        keys = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        vals = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        lookUp = dict(zip( keys, vals))
        #print('leap year')
    else:
        keys = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        vals = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        lookUp = dict(zip(keys, vals))
        #print('no leap year')
    res = day + lookUp[month]
    return res

def getAttributesName(layer):

    # check the type of layer
    if type(layer) is ogr.Layer:
        lyr = layer

    elif type(layer) is ogr.DataSource:
        lyr = layer.GetLayer(0)

    elif type(layer) is str:
        lyrOpen = ogr.Open(layer)
        lyr = lyrOpen.GetLayer(0)

    # create empty dict and fill it
    header = dict.fromkeys(['Name', 'Type'])
    head   = [[lyr.GetLayerDefn().GetFieldDefn(n).GetName(),
             ogr.GetFieldTypeName(lyr.GetLayerDefn().GetFieldDefn(n).GetType())]
            for n in range(lyr.GetLayerDefn().GetFieldCount())]

    header['Name'], header['Type'] = zip(*head)

    return header

def getAttributesALL(layer):

    # check the type of layer
    if type(layer) is ogr.Layer:
        lyr = layer

    elif type(layer) is ogr.DataSource:
        lyr = layer.GetLayer(0)

    elif type(layer) is str:
        lyrOpen = ogr.Open(layer)
        lyr = lyrOpen.GetLayer(0)

    # create empty dict and fill it

    header = dict.fromkeys(['Name', 'Type'])

    head = [[lyr.GetLayerDefn().GetFieldDefn(n).GetName(),
             ogr.GetFieldTypeName(lyr.GetLayerDefn().GetFieldDefn(n).GetType())]
            for n in range(lyr.GetLayerDefn().GetFieldCount())]

    header['Name'], header['Type'] = zip(*head)

    attrib = dict.fromkeys(header['Name'])
    for i, j in enumerate(header['Name']):
        attrib[j] = [lyr.GetFeature(k).GetField(j) for k in range(lyr.GetFeatureCount())]

    return attrib

def getSpatRefRas(layer):
    # check type of layer
    if type(layer) is gdal.Dataset:
        SPRef = osr.SpatialReference()
        SPRef.ImportFromWkt(layer.GetProjection())

    elif type(layer) is str:
        lyr   = gdal.Open(layer)
        SPRef = osr.SpatialReference()
        SPRef.ImportFromWkt(lyr.GetProjection())

    #print(SPRef)
    return(SPRef)

def getSpatRefVec(layer):

    # check the type of layer
    if type(layer) is ogr.Geometry:
        SPRef   = layer.GetSpatialReference()

    elif type(layer) is ogr.Feature:
        lyrRef  = layer.GetGeometryRef()
        SPRef   = lyrRef.GetSpatialReference()

    elif type(layer) is ogr.Layer:
        SPRef   = layer.GetSpatialRef()

    elif type(layer) is ogr.DataSource:
        lyr     = layer.GetLayer(0)
        SPRef   = lyr.GetSpatialRef()

    elif type(layer) is str:
        lyrOpen = ogr.Open(layer)
        lyr     = lyrOpen.GetLayer(0)
        SPRef   = lyr.GetSpatialRef()

    #print(SPRef)
    return(SPRef)

def getHexType(raster):
    if type(raster) is str:
        ras    = gdal.Open(raster)
        rasti  = ras.GetRasterBand(1)
        ras_DT = rasti.DataType

    elif type(raster) is gdal.Dataset:
        rasti  = raster.GetRasterBand(1)
        ras_DT = rasti.DataType

    elif type(raster) is gdal.Band:
        ras_DT = raster.DataType

    gdals = [1,2,3,4,5,6,7]
    hexas = ['b', 'H', 'h', 'I', 'i', 'f', 'd']
    lookUp = dict(zip(gdals, hexas))

    hexa = []
    for k ,v in lookUp.items():
        if k == ras_DT:
            hexa = lookUp[k]
    return hexa

# #### have to be the same projection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def getRasCellFromXY(X, Y, raster):
    if type(raster) is str:
        ras = gdal.Open(raster)
    elif type(raster) is gdal.Dataset:
        ras = raster
    gt  = ras.GetGeoTransform()
    col = int((X - gt[0]) / gt[1])
    row = int((Y - gt[3]) / gt[5])
    return([col, row])

# function that extracts extent and corner coordinates of a raster file - checked only with UTM projection
def getcorners(single_ras):
    ds = gdal.Open(single_ras)
    gt = ds.GetGeoTransform()
    ext = {'Xmin': gt[0],
           'Xmax': gt[0] + (gt[1] * ds.RasterXSize),
           'Ymin': gt[3] + (gt[5] * ds.RasterYSize),
           'Ymax': gt[3]}
    coo = {'UpperLeftXY': [ext['Xmin'], ext['Ymax']],
           'UpperRightXY': [ext['Xmax'], ext['Ymax']],
           'LowerRightXY': [ext['Xmax'], ext['Ymin']],
           'LowerLeftXY': [ext['Xmin'], ext['Ymin']]}
    return ext, coo

# ## get minmax values of x,y of a raster(path) or a list with raster (paths)
def getExtentRas(raster):
    if type(raster) is str:
        ds = gdal.Open(raster)
    elif type(raster) is gdal.Dataset:
        ds = raster
    gt = ds.GetGeoTransform()
    ext = {'Xmin': gt[0],
            'Xmax': gt[0] + (gt[1] * ds.RasterXSize),
            'Ymin': gt[3] + (gt[5] * ds.RasterYSize),
            'Ymax': gt[3]}
    return ext

def getExtentVec(shape):
    if type(shape) is str:
        lyrOpen = ogr.Open(shape)
        lyr     = lyrOpen.GetLayer(0)
    elif type(shape) is ogr.DataSource:
        lyr = shape.GetLayer(0)
    elif type(shape) is ogr.Layer:
        lyr = shape
    xmin, xmax, ymin, ymax = lyr.GetExtent()
    ext = {'Xmin': xmin,
           'Xmax': xmax,
           'Ymin': ymin,
           'Ymax': ymax}
    return ext

# ## get common bounding box dimensions for list of Extent_dictionaries
def commonBoundsDim(extentList):
    # create empty dictionary with list slots for corner coordinates
    k = ['Xmin', 'Xmax', 'Ymin', 'Ymax']
    v = [[], [], [], []]
    res = dict(zip(k, v))

    # fill it with values of all raster files
    for i in extentList:
        for j in k:
            res[j].append(i[j])
    # determine min or max values per values' list to get common bounding box
    ff = [max, min, max, min]
    for i, j in enumerate(ff):
        res[k[i]] = j(res[k[i]])
    return res

# ## get common bounding box coordinates for Extent_dictionaries
def commonBoundsCoord(ext):
    if type(ext) is dict:
        ext = [ext]
    else:
        ext = ext
    cooL = []
    for i in ext:
        coo = {'UpperLeftXY': [i['Xmin'], i['Ymax']],
               'UpperRightXY': [i['Xmax'], i['Ymax']],
               'LowerRightXY': [i['Xmax'], i['Ymin']],
               'LowerLeftXY': [i['Xmin'], i['Ymin']]}
        cooL.append(coo)
    return cooL

# ## read in one or multiple raster as one or multiple subsets based on coordinates
# ## optional: write away subsets and set NoData manually : only for single raster tiles at the moment!!!!!
def rastersubbyCord(raster,ULx,ULy,LRx,LRy,storpath='none', nodata='fromimage'):
    # check storpath
    if storpath is not 'none':
        if storpath.endswith('/'):
            storpath = storpath
        else:
            storpath = storpath + '/'
    # check if raster is list
    if type(raster) is not list:
        raster = [raster]
    else:
         raster = raster
    k = ['Raster', 'ULx_off', 'ULy_off', 'LRx_off', 'LRy_off', 'Data']
    v = [[], [], [], [], [], []]
    res = dict(zip(k, v))

    for z, i in enumerate(raster):
        if type(i) is not gdal.Dataset:
            in_ds = gdal.Open(i)
        else:
            in_ds = i
        in_gt = in_ds.GetGeoTransform()
        inv_gt = gdal.InvGeoTransform(in_gt)
        # transform coordinates into offsets (in cells) and make them integer
        off_UpperLeft = gdal.ApplyGeoTransform(inv_gt, ULx,ULy)  # new UL * rastersize^-1  + original ul/rastersize(opposite sign
        off_LowerRight = gdal.ApplyGeoTransform(inv_gt, LRx, LRy)
        off_ULx, off_ULy = map(round, off_UpperLeft)  # or int????????????????
        off_LRx, off_LRy = map(round, off_LowerRight)

        in_band = in_ds.GetRasterBand(1)
        data = in_band.ReadAsArray(off_ULx, off_ULy, off_LRx - off_ULx, off_LRy - off_ULy)

        if storpath is not 'none':
            gtiff_driver = gdal.GetDriverByName('GTiff')
            out_ds = gtiff_driver.Create(storpath + (i.split('/')[-1]).split('.')[0] + '_subby.tif', off_LRx - off_ULx,
                                         off_LRy - off_ULy, 1, in_ds.GetRasterBand(1).DataType)
            out_gt = list(in_gt)
            out_gt[0], out_gt[3] = gdal.ApplyGeoTransform(in_gt, off_ULx, off_ULy)
            out_ds.SetGeoTransform(out_gt)
            out_ds.SetProjection(in_ds.GetProjection())

            out_ds.GetRasterBand(1).WriteArray(data)
            if nodata is 'fromimage':
                out_ds.GetRasterBand(1).SetNoDataValue(in_band.GetNoDataValue())
            else:
                out_ds.GetRasterBand(1).SetNoDataValue(nodata[z])
            del out_ds

        a = [i, off_ULx, off_ULy, off_LRx, off_LRy, np.where(data != nodata[z], data, np.nan)]
        for t, j in enumerate(k):
            res[j].append(a[t])

    cols = [res['LRx_off'][i] - res['ULx_off'][i] for i, j in enumerate(raster)]
    if len(set(cols)) > 1:
        print('')
        print("WARNING: subsets vary in x extent!!!!!!!!!!!!!!")
        print('')
    rows = [res['LRy_off'][i] - res['ULy_off'][i] for i, j in enumerate(raster)]
    if len(set(rows)) > 1:
        print('')
        print("WARNING: subsets vary in y extent!!!!!!!!!!!!!!")
        print('')

    return res

def reprojShapeEPSG(file, epsg):
    # create spatial reference object
    sref  = osr.SpatialReference()
    sref.ImportFromEPSG(epsg)
    # open the shapefile
    ds = ogr.Open(file, 1)
    driv = ogr.GetDriverByName('ESRI Shapefile')  # will select the driver foir our shp-file creation.

    shapeStor = driv.CreateDataSource('/'.join(file.split('/')[:-1]))
    # get first layer (assuming ESRI is standard) & and create empty output layer with spatial reference plus object type
    in_lyr = ds.GetLayer()
    out_lyr = shapeStor.CreateLayer(file.split('/')[-1].split('.')[0] + '_reproj_' + str(epsg), sref, in_lyr.GetGeomType())

# create attribute field
    out_lyr.CreateFields(in_lyr.schema)
    # with attributes characteristics
    out_feat = ogr.Feature(out_lyr.GetLayerDefn())

    for in_feat in in_lyr:
        geom = in_feat.geometry().Clone()
        geom.TransformTo(sref)
        out_feat.SetGeometry(geom)
        for i in range(in_feat.GetFieldCount()):
            out_feat.SetField(i, in_feat.GetField(i))
        out_lyr.CreateFeature(out_feat)
    shapeStor.Destroy()
    del ds
    return('reprojShape done :)')

def reprojShapeSpatRefVec(file, SpatRefVec):

    # check if file already exists; if yes, delete it
    if os.path.isfile('/'.join(file.split('/')[:-1]) + '/' + file.split('/')[-1].split('.')[0] + '_reproj_' + SpatRefVec.GetAttrValue('PROJCS') + '.shp'):
        ShapeKiller('/'.join(file.split('/')[:-1]) + '/' + file.split('/')[-1].split('.')[0] + '_reproj_' + SpatRefVec.GetAttrValue('PROJCS') + '.shp')
    # open the shapefile
    ds = ogr.Open(file, 1)
    driv = ogr.GetDriverByName('ESRI Shapefile')  # will select the driver foir our shp-file creation.
    shapeStor = driv.CreateDataSource('/'.join(file.split('/')[:-1]))
    # get first layer (assuming ESRI is standard) & and create empty output layer with spatial reference plus object type
    in_lyr = ds.GetLayer()
    out_lyr = shapeStor.CreateLayer(file.split('/')[-1].split('.')[0] + '_reproj_' +  SpatRefVec.GetAttrValue('PROJCS'),
                                    SpatRefVec, in_lyr.GetGeomType())

# create attribute field
    out_lyr.CreateFields(in_lyr.schema)
    # with attributes characteristics
    out_feat = ogr.Feature(out_lyr.GetLayerDefn())

    for in_feat in in_lyr:
        geom = in_feat.geometry().Clone()
        geom.TransformTo(SpatRefVec)
        out_feat.SetGeometry(geom)
        for i in range(in_feat.GetFieldCount()):
            out_feat.SetField(i, in_feat.GetField(i))
        out_lyr.CreateFeature(out_feat)
    shapeStor.Destroy()
    del ds
    return('reprojShape done :)')
# function for creating a list of functions that describe a 'snake' around center with order being the number of outer curls

def BuildSnake(order):
    def goleft(dicti, gs):
        dicti['X'].append(dicti['X'][-1] - gs)
        dicti['Y'].append(dicti['Y'][-1])
        return dicti

    def goup(dicti, gs):
        dicti['X'].append(dicti['X'][-1])
        dicti['Y'].append(dicti['Y'][-1] + gs)
        return dicti

    def goright(dicti, gs):
        dicti['X'].append(dicti['X'][-1] + gs)
        dicti['Y'].append(dicti['Y'][-1])
        return dicti

    def godown(dicti, gs):
        dicti['X'].append(dicti['X'][-1])
        dicti['Y'].append(dicti['Y'][-1] - gs)
        return dicti

    fL = list()
    for i in range(order):
        ordi = i + 1
        fdum = [[goleft], (2* ordi -1) * [goup], 2 * ordi * [goright], 2 * ordi * [godown], 2 * ordi * [goleft]]
        fL.append(fdum)
    ffL = [subitem for sublist in fL for item in sublist for subitem in item]
    return ffL

def ApplySnake(dict, gridsiz, snaki):
    for i in snaki:
        res = i(dict, gridsiz)
    return res

# ## function centroid to polygon
def boundingCentroidCoord(X, Y, celldim):
    if type(X) is not list:
        XX, YY = [X], [Y]
    else:
        XX, YY = X, Y
    k = ['ulX', 'ulY', 'urX', 'urY', 'lrX', 'lrY', 'llX', 'llY']
    v = [[], [], [], [], [], [], [], []]
    res = dict(zip(k, v))
    for i in XX:

        res['ulX'].append(i - celldim/2)
        res['urX'].append(i + celldim/2)
        res['lrX'].append(i + celldim/2)
        res['llX'].append(i - celldim/2)

    for j in YY:
        res['ulY'].append(j + celldim / 2)
        res['urY'].append(j + celldim / 2)
        res['lrY'].append(j - celldim / 2)
        res['llY'].append(j - celldim / 2)

    return res

def movinWinni(rasterlist, win_names, funci, win_dim = 'none', storpath = 'none'):
    print(rasterlist)
    print(win_names)
    print(funci)
    print(win_dim)
    print(storpath)
    # check storpath
    if storpath is not 'none':
        if storpath.endswith('/'):
            storpath = storpath
        else:
            storpath = storpath + '/'

    # create list to store output
    res = []

    # loop through files and
    for wc, w in enumerate(win_names):
        for img in rasterlist:
            # get raster info and data
            in_ras = gdal.Open(img)
            in_ras_band = in_ras.GetRasterBand(1)
            in_ras_data = in_ras_band.ReadAsArray()
            in_rows = in_ras_data.shape[0]  # rows of image
            in_cols = in_ras_data.shape[1]  # cols of image
            in_ras_gt = in_ras.GetGeoTransform()
            ras_res = in_ras_gt[1]

            # check for dimensions; if none given, the radius is converted to cells for window
            print(win_dim)
            if win_dim is 'none':
                ww = int((w * 2 + ras_res) / ras_res)
                win = (ww, ww)
            else:
                win = (win_dim[0], win_dim[1])
            print(win[0])
            rows = in_rows - win[0] + 1
            cols = in_cols - win[1] + 1

            # slice the data times the product of window dimensions
            slici = []
            for i in range(win[0]):
                for j in range(win[1]):
                    slici.append(in_ras_data[i:rows+i,j:cols+j])
            # stack slices and fill empty array with funci returns
            stacki = np.dstack(slici)
            print(stacki.shape)
            out_data = np.zeros(in_ras_data.shape, np.float32)
            out_data[math.floor(win[0]/2):-(math.floor(win[0]/2)), math.floor(win[1]/2):-(math.floor(win[1]/2))] = np.apply_along_axis(Shanni, 2, stacki)
            res.append(out_data)

            #write in raster (optional)
            if storpath is 'none':
                print("did you assign this function's output to an object? - no copy on drive")
            else:
                print('copy on drive')
                gtiff_driver = gdal.GetDriverByName('GTiff')
                out_ds = gtiff_driver.Create(storpath + '_' + img.split('/')[-1].split('.')[0] + '_' +
                                             str(win_names[wc]) + '_' + funci.__name__ + '.tif', in_cols,
                                             in_rows, 1, eType=gdal.GDT_Float32)
                out_ds.SetGeoTransform(in_ras.GetGeoTransform())
                out_ds.SetProjection(in_ras.GetProjection())
                out_ds.GetRasterBand(1).WriteArray(out_data)
                del out_ds
    return(res)

def Shanni(a):
    b = len(a)
    unique, counts = np.unique(a, return_counts=True) # np.count_nonzero
    res = np.sum((counts / b) * np.log(counts / b)) * -1
    return res

def XYtoShape(XYdict, attributes, epsg, storpath, name, Stype):
    #get the keys of the attributes dictonary
    attribs = []
    for k, v in attributes.items():
        attribs.append(k)

    #get the keys from the coordinates dictonary
    XYkeys = []
    for k, v in XYdict.items():
        XYkeys.append(k)

    # create SpatialReference from given epsg
    sref = osr.SpatialReference()
    sref.ImportFromEPSG(epsg)

    # create the empty shapefile with given name at given location; the Stype can be point or poly so far
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shapeStor = driver.CreateDataSource(storpath)
    if Stype == 'point':
        out_lyr = shapeStor.CreateLayer(name, sref, ogr.wkbPoint)
    elif Stype == 'poly':
        out_lyr = shapeStor.CreateLayer(name, sref, ogr.wkbPolygon)

    # create the attributes from dictonary (all as strings) on the feature
    for k, v in attributes.items():
        nam_fld = ogr.FieldDefn(k, ogr.OFTString)
        nam_fld.SetWidth(50)
        out_lyr.CreateField(nam_fld)

    out_feat = ogr.Feature(out_lyr.GetLayerDefn())
    if Stype == 'point':
        for i in range(len(XYdict[XYkeys[0]])):
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(XYdict[XYkeys[0]][i], XYdict[XYkeys[1]][i])
            out_feat.SetGeometry(point)

            for j in range(len(attributes.keys())):
                out_feat.SetField(j, attributes[attribs[j]][i])
            out_lyr.CreateFeature(out_feat)
    elif Stype == 'poly': # note the hard coded keys!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in range(len(XYdict['ulX'])):
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(XYdict['ulX'][i], XYdict['ulY'][i])
            ring.AddPoint(XYdict['urX'][i], XYdict['urY'][i])
            ring.AddPoint(XYdict['lrX'][i], XYdict['lrY'][i])
            ring.AddPoint(XYdict['llX'][i], XYdict['llY'][i])

            square = ogr.Geometry(ogr.wkbPolygon)
            square.AddGeometry(ring)
            square.CloseRings()
            out_feat.SetGeometry(square)

            for j in range(len(attributes.keys())):
                out_feat.SetField(j, attributes[attribs[j]][i])
            out_lyr.CreateFeature(out_feat)
    shapeStor.Destroy()
    return(print('XYtoShape done :)'))

def getXYfromShape(shapefile):
    coo = dict.fromkeys(['X', 'Y'])
    coo['X'] = []
    coo['Y'] = []

    sh   = ogr.Open(shapefile)
    sha  = sh.GetLayer()
    shap = sha.GetNextFeature()

    while shap:
        geom = shap.geometry()
        coo['X'].append(geom.GetX())
        coo['Y'].append(geom.GetY())
        shap = sha.GetNextFeature()
    sha.ResetReading()

    return(coo)

def ShapeKiller(shape_path):
    if os.path.isfile(shape_path):
       os.remove(shape_path)
       os.remove(shape_path.split('.')[0] + '.shx')
       os.remove(shape_path.split('.')[0] + '.prj')
       os.remove(shape_path.split('.')[0] + '.dbf')

def RasterKiller(raster_path):
    if os.path.isfile(raster_path):
        os.remove(raster_path)

def warpMODIS(hdfConti, storpath, epsg):
    if storpath.endswith('/'):
        storpath = storpath
    else:
        storpath = storpath + '/'

    hdf = gdal.Open(hdfConti)
    sdsdict = hdf.GetMetadata('SUBDATASETS')
    sdslist = [sdsdict[k] for k in sdsdict.keys() if '_NAME' in k]

    for i, file in enumerate(sdslist):
        in_ds  = gdal.Open(file)
        out_ds = storpath + '_'.join(hdfConti.split('/')[-1].split('.')[:-1]) + '_' + sdslist[i].split('"')[2].split(':')[-1] + '.tif'
        gdal.Warp(out_ds, in_ds, dstSRS = 'EPSG:' + str(epsg))

def points_to_Center(points, refimage):

    # check if file already exists; if yes, delete it
    if os.path.isfile('/'.join(points.split('/')[:-1]) + '/' + points.split('/')[-1].split('.')[0] + '_centerCoord' + '.shp'):
        ShapeKiller('/'.join(points.split('/')[:-1]) + '/' + points.split('/')[-1].split('.')[0] + '_centerCoord' + '.shp')
    # open the shapefile
    ds = ogr.Open(points, 0)
    driv = ogr.GetDriverByName('ESRI Shapefile')  # will select the driver foir our shp-file creation.
    shapeStor = driv.CreateDataSource('/'.join(points.split('/')[:-1]))
    # get first layer (assuming ESRI is standard) & and create empty output layer with
    in_lyr = ds.GetLayer()
    out_lyr = shapeStor.CreateLayer(points.split('/')[-1].split('.')[0] + '_centerCoord',
                                    getSpatRefVec(points), in_lyr.GetGeomType())

    # open reference image
    rasti = gdal.Open(refimage)
    rast = rasti.GetGeoTransform()
    refX = rast[0]
    refY = rast[3]
    grid_size = rast[1]

    # create attribute field
    out_lyr.CreateFields(in_lyr.schema)
    # with attributes characteristics
    out_feat = ogr.Feature(out_lyr.GetLayerDefn())

    # iterate over features
    for in_feat in in_lyr:
        geomX = in_feat.geometry().GetX()
        geomY = in_feat.geometry().GetY()

        # finding close center coordinate
        Xstart = refX - (math.floor((refX - geomX) / grid_size) * grid_size) - 15
        Ystart = refY - (math.floor((refY - geomY) / grid_size) * grid_size) - 15

        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint(Xstart, Ystart)

        out_feat.SetGeometry(geom)

        for i in range(in_feat.GetFieldCount()):
            out_feat.SetField(i, in_feat.GetField(i))
        out_lyr.CreateFeature(out_feat)
    shapeStor.Destroy()
    del ds
    return('CenterCoords calculated :)')