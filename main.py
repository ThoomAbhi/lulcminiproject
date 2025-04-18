import streamlit as st
# Standard imports
import os
from tqdm.notebook import tqdm
import requests
import json
import time
import pandas as pd
import numpy as np
from PIL import Image

# Geospatial processing packages
import geopandas as gpd
import geojson
import asyncio
import shapely
import rasterio as rio
from rasterio.plot import show
import rasterio.mask
from shapely.geometry import box

# Mapping and plotting libraries
#import matplotlib.pyplot as plt
#import matplotlib.colors as cl
import ee
import eeconvert as eec
import geemap
import geemap.eefolium as emap
import folium

# Deep learning libraries
import torch
from torchvision import datasets, models, transforms
from streamlit_folium import folium_static

from google.colab import drive
from zmq import Flag



shape_name = ''



# print("Data dimensions: {}".format(geoboundary.shape))
# geoboundary.sample(3)


# LULC Classes
classes = [
  'AnnualCrop',
  'Forest',
  'HerbaceousVegetation',
  'Highway',
  'Industrial',
  'Pasture',
  'PermanentCrop',
  'Residential',
  'River',
  'SeaLake'
]

 # Change this to your file destination folder in Google drive


# Get the shape geometry for Kreis Borken



def generate_tiles(image_file, output_file, area_str, size=64):
  """Generates 64 x 64 polygon tiles.

  Args:
    image_file (str): Image file path (.tif)
    output_file (str): Output file path (.geojson)
    area_str (str): Name of the region
    size(int): Window size

  Returns:
    GeoPandas DataFrame: Contains 64 x 64 polygon tiles
  """

  # Open the raster image using rasterio
  raster = rio.open(image_file)
  width, height = raster.shape

  # Create a dictionary which will contain our 64 x 64 px polygon tiles
  # Later we'll convert this dict into a GeoPandas DataFrame.
  geo_dict = { 'id' : [], 'geometry' : []}
  index = 0

  # Do a sliding window across the raster image
  with tqdm(total=width*height) as pbar:
    for w in range(0, width, size):
        for h in range(0, height, size):
            # Create a Window of your desired size
            window = rio.windows.Window(h, w, size, size)
            # Get the georeferenced window bounds
            bbox = rio.windows.bounds(window, raster.transform)
            # Create a shapely geometry from the bounding box
            bbox = box(*bbox)

            # Create a unique id for each geometry
            uid = '{}-{}'.format(area_str.lower().replace(' ', '_'), index)

            # Update dictionary
            geo_dict['id'].append(uid)
            geo_dict['geometry'].append(bbox)

            index += 1
            pbar.update(size*size)

  # Cast dictionary as a GeoPandas DataFrame
  results = gpd.GeoDataFrame(pd.DataFrame(geo_dict))
  # Set CRS to EPSG:4326
  results.crs = {'init' :'epsg:4326'}
  # Save file as GeoJSON
  results.to_file(output_file, driver="GeoJSON")

  raster.close()
  return results

def export_image(image, filename, region, folder):
  """Export Image to Google Drive.

  Args:
    image (ee.image.Image): Generated Sentinel-2 image
    filename (str): Name of image, without the file extension
    geometry (ee.geometry.Geometry): The geometry of the area of
      interest to filter to.
    folder (str): The destination folder in your Google Drive.

  Returns:
    ee.batch.Task: A task instance
  """

  print('Exporting to {}.tif ...'.format(filename))

  task = ee.batch.Export.image.toDrive(
    image=image,
    driveFolder=folder,
    scale=10,
    region=region.geometry(),
    description=filename,
    fileFormat='GeoTIFF',
    crs='EPSG:4326',
    maxPixels=900000000
  )
  task.start()

  return task


def predict_crop(transform,image, shape, classes, model, show=False):
  """Generates model prediction using trained model

  Args:
    image (str): Image file path (.tiff)
    shape (geometry): The tile with which to crop the image
    classes (list): List of LULC classes

  Return
    str: Predicted label
  """

  with rio.open(image) as src:
      # Crop source image using polygon shape
      # See more information here:
      # https://rasterio.readthedocs.io/en/latest/api/rasterio.mask.html#rasterio.mask.mask
      out_image, out_transform = rio.mask.mask(src, shape, crop=True)
      # Crop out black (zero) border
      _, x_nonzero, y_nonzero = np.nonzero(out_image)
      out_image = out_image[
        :,
        np.min(x_nonzero):np.max(x_nonzero),
        np.min(y_nonzero):np.max(y_nonzero)
      ]

      # Get the metadata of the source image and update it
      # with the width, height, and transform of the cropped image
      out_meta = src.meta
      out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
      })

      # Save the cropped image as a temporary TIFF file.
      temp_tif = 'temp.tif'
      with rio.open(temp_tif, "w", **out_meta) as dest:
        dest.write(out_image)

      # Open the cropped image and generated prediction
      # using the trained Pytorch model
      image = Image.open(temp_tif)
      input = transform(image)
      output = model(input.unsqueeze(0))
      _, pred = torch.max(output, 1)
      label = str(classes[int(pred[0])])

      if show:
        out_image.show(title=label)

      return label

  return None

def generate_image(
  region,
  product='COPERNICUS/S2',
  min_date='2018-01-01',
  max_date='2020-01-01',
  range_min=0,
  range_max=2000,
  cloud_pct=10
):

  """Generates cloud-filtered, median-aggregated
  Sentinel-2 image from Google Earth Engine using the
  Pythin Earth Engine API.

  Args:
    region (ee.Geometry): The geometry of the area of interest to filter to.
    product (str): Earth Engine asset ID
      You can find the full list of ImageCollection IDs
      at https://developers.google.com/earth-engine/datasets
    min_date (str): Minimum date to acquire collection of satellite images
    max_date (str): Maximum date to acquire collection of satellite images
    range_min (int): Minimum value for visalization range
    range_max (int): Maximum value for visualization range
    cloud_pct (float): The cloud cover percent to filter by (default 10)

  Returns:
    ee.image.Image: Generated Sentinel-2 image clipped to the region of interest
  """

  # Generate median aggregated composite
  image = ee.ImageCollection(product)\
      .filterBounds(region)\
      .filterDate(str(min_date), str(max_date))\
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))\
      .median()

  # Get RGB bands
  image = image.visualize(bands=['B4', 'B3', 'B2'], min=range_min, max=range_max)
  # Note that the max value of the RGB bands is set to 65535
  # because the bands of Sentinel-2 are 16-bit integers
  # with a full numerical range of [0, 65535] (max is 2^16 - 1);
  # however, the actual values are much smaller than the max value.
  # Source: https://stackoverflow.com/a/63912278/4777141

  return image.clip(region)

# async def initialize_ee():
#     try:
#         # Replace 'path/to/your/service-account-key.json' with your actual JSON key file path
#         key_file = './service-account-key.json'
#         credentials = await ee.ServiceAccountCredentials(None, key_file)
#         ee.Initialize(credentials)
#         return True
#     except Exception as e:
#         st.error(f"Error initializing Earth Engine: {str(e)}")
#         return False

def wait_for_ee_task(task, check_interval=10):
  print("Waiting for Earth Engine task to complete...")
  with st.spinner("Exporting image, please wait..."):
    while True:
        status = task.status()
        state = status['state']
        if state in ['COMPLETED', 'FAILED', 'CANCELLED']:
            break

        time.sleep(20)
    if state == 'COMPLETED':
      st.success("✅ Export complete!")
    else:
      st.error(f"❌ Export failed: {state}")

async def fun():
  # drive.mount('/content/drive', force_remount=True)
  ee.Authenticate()
  ee.Initialize(project="helical-apricot-454313-u0")
  ISO = 'IND' # "DEU" is the ISO code for Germany
  ADM = 'ADM3' # Equivalent to administrative districts

  # Query geoBoundaries
  url = f"https://www.geoboundaries.org/api/current/gbOpen/{ISO}/{ADM}"
  r = requests.get(url)
  download_path = r.json()["gjDownloadURL"]

  # Save the result as a GeoJSON
  filename = 'geoboundary.geojson'
  geoboundary = requests.get(download_path).json()
  with open(filename, 'w') as file:
    geojson.dump(geoboundary, file)

  # Read data using GeoPandas
  geoboundary = gpd.read_file(filename)


  if not (geoboundary['shapeName'] == shape_name).any():
    st.error("City is wrong")
    return ""


  region  = geoboundary.loc[geoboundary.shapeName == shape_name]
  centroid = region.iloc[0].geometry.centroid.coords[0]
  region = eec.gdfToFc(region) #geodataframe to feature collection

  # Generate RGB image using GEE
  # image = generate_image(
  #     region,
  #     product='COPERNICUS/S2', # Sentinel-2A
  #     min_date='2021-01-01', # Get all images within
  #     max_date='2021-12-31', # the year 2021
  #     cloud_pct=10, # Filter out images with cloud cover >= 10.0%
  # )

  # folder = 'Colab Notebooks'
  # task = export_image(image, shape_name, region, folder)
  # wait_for_ee_task(task)


  # Change this to your image file path
  cwd = './drive/My Drive/Colab Notebooks/'
  tif_file = os.path.join(cwd, '{}.tif'.format(shape_name))

  # Uncomment this to download the TIF file
  #if not os.path.isfile(tif_file):
    #tif_file = '{}.tif'.format(shape_name)
    #!gdown "12VJQBht4n544OXh4dmugqMESXXxRlBcU"

  # Open image file using Rasterio

  boundary = geoboundary[geoboundary.shapeName == shape_name]

  output_file = os.path.join(cwd, '{}.geojson'.format(shape_name))
  tiles = generate_tiles(tif_file, output_file, shape_name, size=64)




  # Geopandas sjoin function
  tiles = gpd.sjoin(tiles, boundary, predicate='within')

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_file = cwd+'/models/best_model.pth'

  # Uncomment this to download the model file
  #if not os.path.isfile(model_file):
    #model_file = 'best_model.pth'
    #!gdown "13AFOESwxKmexCoOeAbPSX_wr-hGOb9YY"

  model = models.resnet50(pretrained=True)
  num_ftrs = model.fc.in_features
  model.fc = torch.nn.Linear(num_ftrs, 10)
  model.load_state_dict(torch.load(model_file, map_location=device))
  model.eval()

  imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

  transform = transforms.Compose([
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(imagenet_mean, imagenet_std)
  ])

  # Commence model prediction
  labels = [] # Store predictions
  for index in tqdm(range(len(tiles)), total=len(tiles)):
    label = predict_crop(transform,tif_file, [tiles.iloc[index]['geometry']], classes, model)
    labels.append(label)
  tiles['pred'] = labels

  filepath = os.path.join(cwd, "{}_preds.geojson".format(shape_name))
  tiles.to_file(filepath, driver="GeoJSON")

  tiles = gpd.read_file(filepath)

  # We map each class to a corresponding color
  colors = {
    'AnnualCrop' : 'lightgreen',
    'Forest' : 'forestgreen',
    'HerbaceousVegetation' : 'yellowgreen',
    'Highway' : 'gray',
    'Industrial' : 'red',
    'Pasture' : 'mediumseagreen',
    'PermanentCrop' : 'chartreuse',
    'Residential' : 'magenta',
    'River' : 'dodgerblue',
    'SeaLake' : 'blue'
  }
  tiles['color'] = tiles["pred"].apply(
    lambda x: cl.to_hex(colors.get(x))
  )

  # Instantiate map centered on the centroid
  map = folium.Map(location=[centroid[1], centroid[0]], zoom_start=10)

  # Add Google Satellite basemap
  folium.TileLayer(
        tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr = 'Google',
        name = 'Google Satellite',
        overlay = True,
        control = True
  ).add_to(map)

  # Add LULC Map with legend
  legend_txt = '<span style="color: {col};">{txt}</span>'
  for label, color in colors.items():

    # Specify the legend color
    name = legend_txt.format(txt=label, col=color)
    feat_group = folium.FeatureGroup(name=name)

    # Add GeoJSON to feature group
    subtiles = tiles[tiles.pred==label]
    if len(subtiles) > 0:
      folium.GeoJson(
          subtiles,
          style_function=lambda feature: {
            'fillColor': feature['properties']['color'],
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5,
          },
          name='LULC Map'
      ).add_to(feat_group)
      map.add_child(feat_group)

  folium.LayerControl().add_to(map)
  return map

st.title("City Input")
cities = ["Karimnagar", "Los Angeles", "Chicago", "Houston", "Phoenix"]

# Dropdown to select city
city = st.selectbox("Select a City", cities)
if st.button("Submit"):
    shape_name = city
    map_obj = asyncio.run(fun())
    if(map_obj != ""):
      folium_static(map_obj, height=500)
