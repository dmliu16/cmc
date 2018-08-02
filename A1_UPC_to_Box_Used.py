
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import ipywidgets as widgets
from ipywidgets import interact

# Plotly plotting support
import plotly.offline as py
py.init_notebook_mode()
import plotly.graph_objs as go
import plotly.figure_factory as ff
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')

# this file joins the MCO shipping sorter scanner message with the packorder information and UPC
# our goal is to figure out for each UPC, what is the typical boxsize people use

# load the mco shipping sorter scanner message and the packorder info
mco_scan = pd.read_csv('MCO_Scan.csv')
wms_packorder = pd.read_csv('wms_packorderid.csv')

# load stockitem
stock_item = pd.read_csv('wms_stockitem.csv')

# change the name
mco_scan = mco_scan.rename(index=str, columns={"Shipping Unit Barcode": "SHIPPINGLUBARCODE"})

# convert to floats
mco_scan['SHIPPINGLUBARCODE'] = pd.to_numeric(mco_scan['SHIPPINGLUBARCODE'], errors='coerce')
wms_packorder['SHIPPINGLUBARCODE'] = pd.to_numeric(wms_packorder['SHIPPINGLUBARCODE'], errors='coerce')

# merge the two table together on shippinglubarcode
packorder_boxsize = pd.merge(mco_scan, wms_packorder, on='SHIPPINGLUBARCODE', how='inner')

# delete the unnecessary cols
del packorder_boxsize['Scanner Id']
del packorder_boxsize['Disposition Code']
del packorder_boxsize['ID']
del packorder_boxsize['CREATED']

# remove the duplicates from wms_stockitem
stock_item = stock_item.drop_duplicates(subset=['PACKORDERID'])

# join the two table
upc_boxsize = pd.merge(packorder_boxsize, stock_item, on='PACKORDERID', how='inner')

# delete cols
del upc_boxsize['Unnamed: 0']
del upc_boxsize['ID']
del upc_boxsize['CREATED']

# change the name
upc_boxsize = upc_boxsize.rename(index=str, columns={"MATERIALUPC": "UPC"})

from scipy import stats # will use this library to help find the mode

# group by upc
group_upc = upc_boxsize.groupby('UPC')

# define the cols
upc = []
size = []

# find the upc and size of pack
for name,group in group_upc:
    # put all the upc's in the list
    upc.append(name)
    # get all the packaging barcode as a list
    boxsize = group['Packaging Barcode'].tolist()
    # find the mode
    m = stats.mode(boxsize)
    size.append(m[0][0])

# create a dataframe with upc and boxsize
find_boxsize = {'UPC': upc, 'Packaging Barcode': size}
find_boxsize_by_upc = pd.DataFrame(data=find_boxsize)

#find_boxsize_by_upc.to_csv('find_boxsize_by_upc.csv')


# In[51]:




