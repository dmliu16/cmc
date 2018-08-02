
# coding: utf-8

# In[13]:


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

# import all the successfully packed boxes
cmc_info = pd.read_csv('wms_cw1000_waste.csv')

# figure out the actual box used by walmart
box_type = {'Walmart_Box': ['S1-15', 'S2-18', 'S3-18', 'S4-21', 'S5-24', 'S6-24', 'M1-27', 'M2-27', 'M3-33',
                    'M4-30', 'M5-30', 'M6-36', 'L1-33', 'L2-33', 'L3-45', 'L4-39', 'L5-45', 'L6-45'],
            #'height': [4,   5.25, 6.25,  7.5,  5,   11,  8.5,  11.5,  5.25,  10,  12.25,  9,  13,  16,  12.25,  15,  14.125,  18], 
            #'length': [9.0, 11,   13,    14,   17,  15,  19,   17,    26,    23,  22,     29, 25,  23,  35,     29,  36,      36], 
            #'width' : [6,   8.0,  9,     10,   13,  12,  12,   13,    19,    15,  16,     19, 17,  19,  18,     20,  21,      21],
            'wal_cost'  : [0.14,0.23, 0.29,  0.37, 0.47,0.54,0.55, 0.78,  0.99,  1.01,1.16,   1.45,1.35,1.72,1.86,1.98,  2.68,    2.92] 
           }
type_cost = pd.DataFrame(data=box_type)

wal_cost_box = pd.merge(cmc_info, type_cost, on='Walmart_Box',how='left')

num_order = len(wal_cost_box)
total_wal_box_cost = sum(wal_cost_box['wal_cost'])
average_wal_box_cost = round(total_wal_box_cost/num_order,3)

mm2_ft2 = 1.07639e-5

# cost per square foot of cmc cardboard
cmc_cardboard_cost = 0.05

total_cmc_box_cost = (sum(cmc_info['waste']) + sum(cmc_info['cmc_cardboard']))*mm2_ft2*cmc_cardboard_cost
average_cmc_box_cost = round(total_cmc_box_cost/num_order,3)


print('Material Cost')
print('The weighted average for the walmart box is ', average_wal_box_cost, '$')
print('The average cost for the cmc boxes is ', average_cmc_box_cost, '$')
cmc_savings = average_cmc_box_cost-average_wal_box_cost
print('On average, we spend ', cmc_savings, '$ more on single unit orders for cmc')

# this part of the code helps you figure out the dunnage cost
num_airbags_pack = 5900*12/5
num_packs = 13680
total_airbags = num_airbags_pack*num_packs
total_cost_airbags = 1700000
cost_airbag = round(total_cost_airbags/total_airbags,2)

print('One airbag is ', cost_airbag, '$')
print('Average saving on airbags is ', round(cost_airbag*5,2), '$ per single unit order\n')

print('Shipping Cost')
print('The estimated savings on shipping cost is 0.08$ per order\n')

print('Total Savings')
print('Total savings on cmc is ', -round(cmc_savings,2)+round(cost_airbag*5,2)+0.08, '$ per single unit order')

