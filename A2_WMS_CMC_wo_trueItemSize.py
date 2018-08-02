
# coding: utf-8

# In[31]:


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

# this file aims to create a table with all the information we need later for analysis

# import data
cw1000_out = pd.read_csv('cw1000_out_good.csv')
wms_packorder = pd.read_csv('wms_packorderid.csv')
stock_item = pd.read_csv('wms_stockitem.csv')
upc_dim = pd.read_csv('upc_dim.csv')

# merge the two for easy comparison 
wms_cw1000 = pd.merge(wms_packorder, cw1000_out, on='PACKORDERID', how='inner')

# delete unnecessary cols
del wms_cw1000['ID']
del wms_cw1000['Unnamed: 0']


# height is a reliable argument
# according to observation, all the items going into cw1000 is bigger than the min_h

# minimum dimesions
min_w = 250
min_l_5_1 = 230
min_l_5_2 = 160
min_l = 170
min_h = 30

# we can also see from the data that 
# 1. b4 5/1/2017, the min length is 230 bc of the labeling
# 2. we are able to adjust the labeling to 160 for length for one day
# 3. min length changed to 170 after 

wms_cw1000['date'] = False

# in order to have a better estimate for the future, we only keep our observation from 5/2/18 onwards
for i in range(len(wms_cw1000)):
    error = 1
    interval = (3,6)
    if (wms_cw1000['CREATED'][i][interval[0]:interval[1]] == 'MAY' or wms_cw1000['CREATED'][i][interval[0]:interval[1]] == 'JUN'):
        if (wms_cw1000['SizeL'][i] >= min_l and not (min_l_5_1-error <= wms_cw1000['SizeL'][i] <= min_l_5_1+error)):
            wms_cw1000['date'][i] = True

            
# only keep the one with good dates
wms_cw1000 = wms_cw1000.where(wms_cw1000['date'] == True).dropna()
# delete the col
del wms_cw1000['date']


# merge to get the chart with upc
stock_item = stock_item.rename(index=str, columns={"MATERIALUPC": "UPC"})


# we can just keep the first bc they would all be the same
stock_item = stock_item.drop_duplicates(subset=['PACKORDERID'], keep='first')
wms_cw1000_upc = pd.merge(stock_item, wms_cw1000, on='PACKORDERID', how='inner')

del wms_cw1000_upc['CREATED_y']
wms_cw1000_upc = wms_cw1000_upc.rename(index=str, columns={"CREATED_x": "CREATED"})

# Only select the cols that we need
upc_dim = upc_dim[['UPC', 'DEPTH', 'HEIGHT', 'WIDTH', 'WEIGHT']]
# drop duplicates
upc_dim = upc_dim.drop_duplicates(subset=['UPC'])

# merge to get all the items with UPC and the recommended walmart dimensions
wms_cw1000_good = pd.merge(upc_dim, wms_cw1000_upc, on='UPC', how='inner')

# reorganize the dataframe to be more clean
wms_cw1000_perfect_wo_itemSize = wms_cw1000_good[['CREATED','SHIPPINGLUBARCODE','PACKORDERID','UPC','DEPTH','HEIGHT','WIDTH','WEIGHT','SHIPPINGLUTYPE','SizeH','SizeL','SizeW','Weight']]

# rename
wms_cw1000_perfect_wo_itemSize = wms_cw1000_perfect_wo_itemSize.rename(index=str, columns={"DEPTH": "wal_l"})
wms_cw1000_perfect_wo_itemSize = wms_cw1000_perfect_wo_itemSize.rename(index=str, columns={"HEIGHT": "wal_h"})
wms_cw1000_perfect_wo_itemSize = wms_cw1000_perfect_wo_itemSize.rename(index=str, columns={"WIDTH": "wal_w"})
wms_cw1000_perfect_wo_itemSize = wms_cw1000_perfect_wo_itemSize.rename(index=str, columns={"WEIGHT": "wal_weight"})

# make sure the units are all in mm
wms_cw1000_perfect_wo_itemSize['wal_l'] = wms_cw1000_perfect_wo_itemSize['wal_l']/10
wms_cw1000_perfect_wo_itemSize['wal_h'] = wms_cw1000_perfect_wo_itemSize['wal_h']/10
wms_cw1000_perfect_wo_itemSize['wal_w'] = wms_cw1000_perfect_wo_itemSize['wal_w']/10

# save the file
#wms_cw1000_perfect_wo_itemSize.to_csv('wms_cw1000_perfect_wo_itemSize.csv')


# when I was writing this code for the first time, I made a mistake by joining the stock_item file directly
# with the wms_cw1000 file without removing the duplicated packorderid in stock_item
# it was challenging bc it is hard to come to the realization that after I remove all the duplicated 
# packorder, there is only a few left for me to merge with the wms_cw1000

# it is actually okay to remove the duplicates in stock_item and keep any one of the duplicates bc they actually
# are duplicating with the same UPC


# In[39]:





# In[41]:





# In[27]:


a = wms_cw1000.sample(n=1000)
sns.distplot(a['SizeW'], rug=True)


# In[26]:


b = wms_cw1000_good.sample(n=1000)
sns.distplot(b['SizeW'], rug=True)


# In[29]:


print(len(wms_cw1000))

