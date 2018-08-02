
# coding: utf-8

# In[2]:


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

# the goal of this file is to figure out the waste and rejection waste of the cmc machine

# import all the successfully packed boxes
cmc_info = pd.read_csv('wms_cw1000_perfectperfect.csv')

# will use this to figure out the rejection rate
good_rejected = pd.read_csv('good_rejected_boxes.csv')

#############################################################################################
# This part of the code figure out the rejection rate base on number of good boxes produced #
#############################################################################################

# only keep date after 5/1/2018
for i in range(len(good_rejected)):
    dates = good_rejected['Date'][i]
    month = dates[3:5]
    date = dates[0:2]

    # must be in may or june and cant be 5/1
    if (month != '05' and month != '06' or (month == '05' and date == '01')):     
        good_rejected = good_rejected.drop([i])


# get total good boxes and total bad boxes
good_total = sum(good_rejected['Goodbox Produced'])

# get total rej
rej_total = sum(good_rejected['Box Rejected'])

# rej_good ratio
rej_good_ratio = rej_total/good_total
        
##########################
# End of rejection rate  #
##########################

    
# the original one are all good boxes
cmc_info['Rejected'] = False

# good boxes produced
num_good = len(cmc_info)
num_rej = round(num_good*rej_good_ratio, 0)

# I graphed the distributions of the sample of n = 426, and the distribution is very close to the distribution
# of all the box sizes
#rejected_boxes = cmc_info.sample(n = int(num_rej))

# need to change Rejected to True
rejected_boxes['Rejected'] = True

# concat the two
cmc_info = pd.concat([cmc_info, rejected_boxes], sort=True)

# figure out all the rejected cardboard
rejected_cardboard = 0
for i, row in rejected_boxes.iterrows():
    rejected_cardboard += rejected_boxes['cmc_cardboard'][i]


# del unneccesary cols
del cmc_info['Unnamed: 0']

# add one more col for waste
cmc_info['waste'] = 0
cmc_info['cmc_waste'] = 0


for i, row in cmc_info.iterrows():
    # now we need to figure out the waste for each one
    cmc_info['waste'][i] = cmc_waste_cal(cmc_info['SizeH'][i], cmc_info['SizeW'][i], cmc_info['SizeL'][i])
    cmc_info['cmc_waste'][i] = cmc_only_waste_cal(cmc_info['SizeH'][i], cmc_info['SizeW'][i], cmc_info['SizeL'][i])
    
    
cmc_info.to_csv('wms_cw1000_waste.csv')
    
# walmart and cmc density
cmc_cardboard_weight = 0.0608 # pounds / ft2
wal_cardboard_weight = 0.0765 # pounds / ft2

    
total_wasted_cardboard = sum(cmc_info['waste'])
total_wasted_cardboard_cmc = sum(cmc_info['cmc_waste'])

mm2_ft2 = 1.07639e-5

print('Rejected Cardboard')
print('Total rejected cardboard is ', round(rejected_cardboard*mm2_ft2, 3), 'ft2 or ', round(rejected_cardboard*mm2_ft2*cmc_cardboard_weight, 3), ' pound')
print('We can average out the rejection waste to ', round(rejected_cardboard*mm2_ft2/num_good,3), 'ft2  or ', round(rejected_cardboard*mm2_ft2/num_good*cmc_cardboard_weight,3), ' pound per good order\n')

print('Wasted Cardboard')
print('Total wasted cardboard is ', round(total_wasted_cardboard*mm2_ft2, 3), 'ft2 or ', round(total_wasted_cardboard*mm2_ft2*cmc_cardboard_weight, 3) ,' pound')
print('This is roughly ', round(total_wasted_cardboard*mm2_ft2/num_good, 3),'ft2 or ', round(total_wasted_cardboard*mm2_ft2/num_good*cmc_cardboard_weight, 3) ,' per good order\n')

print('Wasted Cardboard_CMC')
print('Total wasted cardboard is ', round(total_wasted_cardboard_cmc*mm2_ft2, 3), 'ft2 or ', round(total_wasted_cardboard_cmc*mm2_ft2*cmc_cardboard_weight, 3), ' pound')
print('This is roughly ', round(total_wasted_cardboard_cmc*mm2_ft2/num_good, 3),'ft2 or ', round(total_wasted_cardboard_cmc*mm2_ft2/num_good*cmc_cardboard_weight, 3), ' pound per good order')


# In[21]:


# unit weight for airbags is 3.1g for 3 airbags => 1 per airbag
# unit weight for airbags .97g/cm3 = 60.5551, 0.014467592592592593

# walmart
# base on A3, we know that per order from Walmart packaging
# we use 4.8 airbags or 0.011 pounds of airbags, 4.353 ft^2 or  0.333 pound of cardboard
# the recycled content of the cardboard would be the max = 87% according to data published by Georgia Pacific
# and the recycle rate for a typical consumer is about 25% according to US News
# thus the overall recycle content would be an estimated 21.75%
# biorenewable content would be 30% from GP thus 7.5%
# airbags are made of HDPE which is non-biodegradable
# recycled content for airbags is currently UNKNOWN, and consumer recycling percentage is an estimated 5% (already being conservative)

# cmc
# per order of cmc packaging
# we use 4.776 ft^2 or  0.29 pound of cardboard and 0.116 ft2  or  0.007  pound rejected, 
# and 2.024 ft2 or  0.123  pound wasted
# 
# 


# In[17]:


# this function finds the waste of the box
# given the dimensions of the box
def cmc_waste_cal(SizeH, SizeW, SizeL):
    dent = 20
    constant = 70
    extra = 35
    gap = 10
    cardboard_width = 1000
    gap_waste = (constant+SizeH+dent)*gap*6 # there are 6 gaps
    vertical_waste = (cardboard_width - SizeW - 2*dent - 2*SizeH - 2*constant)*(2*SizeH+2*SizeL+3*gap+extra)
    side_waste = extra*(2*dent+2*SizeH+2*constant)
    waste = gap_waste + vertical_waste + side_waste
    return waste

# this ignores the small gaps we have for each box
# this is the waste which only cmc would have 
# when georgia pacific makes their cardboards this waste is not there
def cmc_only_waste_cal(SizeH, SizeW, SizeL):
    dent = 20
    constant = 70
    gap = 10
    cardboard_width = 1000
    vertical_waste = (cardboard_width - SizeW - 2*dent - 2*SizeH - 2*constant)*(2*SizeH+2*SizeL+3*gap)
    return vertical_waste


# In[7]:


a = cmc_info.sample(n=2000)
sns.distplot(a['SizeL'], rug=True)


# In[26]:


0.625*5*8/144/12*60.5551


# In[30]:


0.625*5*8/144/12*60.5551217589


# In[4]:


print(len(good_rejected))

