
# coding: utf-8

# In[23]:


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

# the goal of this file is to find all the boxes that are packed by CMC and also recorded in the Schafaer database

# import files
wms_cw1000 = pd.read_csv('wms_cw1000_perfect_wo_itemSize.csv')
upc_boxsize = pd.read_csv('find_boxsize_by_upc.csv')

# merge to find the actual box used for some of them
wms_cw1000_new = pd.merge(wms_cw1000, upc_boxsize, on='UPC', how='left')

# delete cols we dont need
del wms_cw1000_new['Unnamed: 0_x']
del wms_cw1000_new['Unnamed: 0_y']

# figure out the actual box used by walmart
box_type = {'BOXTYPE': ['S1-15', 'S2-18', 'S3-18', 'S4-21', 'S5-24', 'S6-24', 'M1-27', 'M2-27', 'M3-33',
                    'M4-30', 'M5-30', 'M6-36', 'L1-33', 'L2-33', 'L3-45', 'L4-39', 'L5-45', 'L6-45'],
            'height': [4,   5.25, 6.25,  7.5,  5,   11,  8.5,  11.5,  5.25,  10,  12.25,  9,  13,  16,  12.25,  15,  14.125,  18], 
            'length': [9.0, 11,   13,    14,   17,  15,  19,   17,    26,    23,  22,     29, 25,  23,  35,     29,  36,      36], 
            'width' : [6,   8.0,  9,     10,   13,  12,  12,   13,    19,    15,  16,     19, 17,  19,  18,     20,  21,      21]}
type_dimension = pd.DataFrame(data=box_type)

# make sure we talking about the same dimensions
# rearrange the order of wal_l, wal_h, and wal_w to match that of itemW, itemL, itemH
wms_cw1000_new = standardize(wms_cw1000_new)

# find the three dimensions of the box
wms_cw1000_new['walbox_l'], wms_cw1000_new['walbox_h'], wms_cw1000_new['walbox_w'] = 0,0,0
wms_cw1000_new['Walmart_Box'] = 0

for i in range(len(wms_cw1000_new)):
    code = wms_cw1000_new['Packaging Barcode'][i]
    # make sure it's not a NaN
    if (type(code) == float):
        wms_cw1000_new['walbox_l'][0], wms_cw1000_new['walbox_h'][0], wms_cw1000_new['walbox_w'][0] = 0,0,0
    # this is -- or an actual pack order
    else:
        # --
        if (len(code) < 10 or code[2] != '0' or code[5] != '0'):
            print(len(code))
            wms_cw1000_new['walbox_l'][0], wms_cw1000_new['walbox_h'][0], wms_cw1000_new['walbox_w'][0] = 0,0,0
        # actual packorder
        else:
            # figure out the length aka longest dimension
            length = float(code[0:2])
            # only the length 9 will cuz this
            if (length == 90):
                length = 9
            # figure out the width aka second longest
            width = float(code[3:5])
            # only 6, 8, 9 will cuz this
            if (width > 30):
                width = width/10
            # figure out the height aka shortest dimension
            height = float(code[6:8])
            if (height > 20):
                height = height/10
            
            print(i)
            # match it with the right boxes
            boxes = type_dimension.where(np.logical_and(type_dimension['length']==length,type_dimension['width']==width)).dropna()
            # find the box
            box = boxes.where(abs(boxes['height'] - height) < 1).dropna()['BOXTYPE'].tolist()[0]
            # assign the box
            wms_cw1000_new['Walmart_Box'][i] = box
            
            # assign the dimensions
            wms_cw1000_new['walbox_l'][i] = length*25.4
            wms_cw1000_new['walbox_w'][i] = width*25.4
            wms_cw1000_new['walbox_h'][i] = float(boxes.where(abs(boxes['height'] - height) < 1).dropna()['height'].tolist()[0])*25.4

# delete cols
del wms_cw1000_new['Packaging Barcode']
del wms_cw1000_new['wal_weight']

# only keep the boxed ones
wms_cw1000_new['boxed'] = True

# find the not boxed ones
for index, row in wms_cw1000_new.iterrows():
    if (wms_cw1000_new['SHIPPINGLUTYPE'][index][0:3] != 'BOX' and wms_cw1000_new['Walmart_Box'][index] == 0):
        wms_cw1000_new['boxed'][index] = False
        
wms_cw1000_new = wms_cw1000_new.where(wms_cw1000_new['boxed'] == True).dropna()

# minimum dimesions
min_w = 250
min_l_5_1 = 230
min_l_5_2 = 160
min_l = 170
min_h = 30

# get rid of small items only left with normal sized items
cw1000_normal = wms_cw1000_new.where(np.logical_and(np.logical_and(wms_cw1000_new['SizeW'] > min_w, wms_cw1000_new['SizeH'] > min_h), wms_cw1000_new['SizeL'] > min_l))
cw1000_normal = cw1000_normal.dropna()

# an estimated space that the item can move around in each direction
freeSpace = 13 # this is about half an inch
boxThickness = 3

# the extra width for the built of the box 
extra_width = 20*2 # 2cm on each side
cw1000_normal['itemW'] = cw1000_normal['SizeW'] - freeSpace - 2*boxThickness - extra_width
cw1000_normal['itemH'] = cw1000_normal['SizeH'] - freeSpace - 2*boxThickness
cw1000_normal['itemL'] = cw1000_normal['SizeL'] - freeSpace - 2*boxThickness

# this gets us the items of normal size
#cw1000_normal.to_csv('cw1000_normal.csv')

# find small items
cw1000_small = wms_cw1000_new.where(np.logical_or(np.logical_or(wms_cw1000_new['SizeW'] <= min_w, wms_cw1000_new['SizeH'] <= min_h), wms_cw1000_new['SizeL'] <= min_l))
cw1000_small = cw1000_small.dropna()

# these are the default item size
cw1000_small['itemW'], cw1000_small['itemL'], cw1000_small['itemH'] = cw1000_small['wal_w'], cw1000_small['wal_l'], cw1000_small['wal_h']

# we know the height is nvr minimized bc min of the table is 43 which is bigger than the min dim cmc can make
cw1000_small['itemH'] = cw1000_small['SizeH'] - freeSpace - 2*boxThickness
for index, row in cw1000_small.iterrows():
    # the max width and length we can have
    max_width = cw1000_small['SizeW'][index] - freeSpace - 2*boxThickness - extra_width
    max_length = cw1000_small['SizeL'][index] - freeSpace - 2*boxThickness
    # if the width is bigger than min
    if cw1000_small['SizeW'][index] != min_w:
        cw1000_small['itemW'][index] = max_width
    # see if the walmart measurement is bigger than the max
    elif (cw1000_small['itemW'][index] > max_width):
        cw1000_small['itemW'][index] = max_width

    # if the length is bigger than min
    if cw1000_small['SizeL'][index] != min_l:
        cw1000_small['itemL'][index] = max_length
    # see if the walmart length measurement is bigger than the max length
    elif (cw1000_small['itemL'][index] > max_length):
        cw1000_small['itemL'][index] = max_length

# get the info we need
wms_cw1000_good = pd.concat([cw1000_normal, cw1000_small], sort=True)

# delete unnecessary cols
del wms_cw1000_good['SHIPPINGLUBARCODE']
del wms_cw1000_good['boxed']
del wms_cw1000_good['Weight']
del wms_cw1000_good['wal_h']
del wms_cw1000_good['wal_l']
del wms_cw1000_good['wal_w']

# choose a box for the ones that we can't retrieve information from 
wms_cw1000_good = get_right_size(wms_cw1000_good, type_dimension)

type_dimension = type_dimension.rename(index=str, columns={"BOXTYPE": "Walmart_Box"})
wms_cw1000_good = pd.merge(wms_cw1000_good, type_dimension, on='Walmart_Box', how='left')

del wms_cw1000_good['walbox_h']
del wms_cw1000_good['walbox_l']
del wms_cw1000_good['walbox_w']

# rename
wms_cw1000_good = wms_cw1000_good.rename(index=str, columns={"height": "walbox_h"})
wms_cw1000_good = wms_cw1000_good.rename(index=str, columns={"length": "walbox_l"})
wms_cw1000_good = wms_cw1000_good.rename(index=str, columns={"width": "walbox_w"})

# get to the right units mm
wms_cw1000_good['walbox_h'] = wms_cw1000_good['walbox_h']*25.4
wms_cw1000_good['walbox_w'] = wms_cw1000_good['walbox_w']*25.4
wms_cw1000_good['walbox_l'] = wms_cw1000_good['walbox_l']*25.4

# export this to a csv
wms_cw1000_good.to_csv('wms_cmc_perfect.csv')

# figure out the vol of items
wms_cw1000_good['item_vol'] = wms_cw1000_good['itemH']*wms_cw1000_good['itemW']*wms_cw1000_good['itemL']

# figure out vol of walmart boxes
wms_cw1000_good['walmart_vol'] = wms_cw1000_good['walbox_h']*wms_cw1000_good['walbox_l']*wms_cw1000_good['walbox_w']

# figure out vol of cmc boxes
wms_cw1000_good['cmc_vol'] = wms_cw1000_good['SizeL']*wms_cw1000_good['SizeW']*wms_cw1000_good['SizeH']

wms_cw1000_good['walmart_cardboard'] = 0
wms_cw1000_good['cmc_cardboard'] = 0

# figure out cardboard used by walmart box and cmc box
for i, row in wms_cw1000_good.iterrows():
    print(i)
    wms_cw1000_good['walmart_cardboard'][i] = wal_cardboard(wms_cw1000_good['walbox_h'][i],wms_cw1000_good['walbox_l'][i],wms_cw1000_good['walbox_w'][i])
    wms_cw1000_good['cmc_cardboard'][i] = cmc_cardboard(wms_cw1000_good['SizeL'][i],wms_cw1000_good['SizeW'][i],wms_cw1000_good['SizeH'][i])
    
# this is the final version of the project
wms_cw1000_good.to_csv('wms_cw1000_perfectperfect.csv')

# unit conversion
mm3_ft3 = 3.531467e-8
mm2_ft2 = 1.07639e-5
in3_ft3 = 0.000578704

# get the total values
total_item_vol = round(sum(wms_cw1000_good['item_vol'])*mm3_ft3,3)
total_wal_box_vol = round(sum(wms_cw1000_good['walmart_vol'])*mm3_ft3,3)
total_cmc_box_vol = round(sum(wms_cw1000_good['cmc_vol'])*mm3_ft3,3)

# figure out the dunnage and vol savings
order_num = len(wms_cw1000_good)
dunnage = round(total_wal_box_vol-total_item_vol,3)
airbag_vol = 4.2 * 8.2 * 2.6 * in3_ft3
airbags = round(dunnage/airbag_vol,1)


# walmart and cmc density
cmc_cardboard_weight = 0.0608 # pounds / ft2
wal_cardboard_weight = 0.0765 # pounds / ft2

# total corrugate usage
corrugate_wal = round(sum(wms_cw1000_good['walmart_cardboard'])*mm2_ft2,3)
corrugate_cmc = round(sum(wms_cw1000_good['cmc_cardboard'])*mm2_ft2,3)
wal_weight = round(corrugate_wal*wal_cardboard_weight,3)
cmc_weight = round(corrugate_cmc*cmc_cardboard_weight,3)

print('Number of orders: ', order_num, '\n')

print('Volume')
print('Total volume of the walmart boxes is: ', total_wal_box_vol, ' ft^3')
print('Total volume of the cmc boxes is: ', total_cmc_box_vol, ' ft^3')
print('Total voumne saved by cmc is ', total_wal_box_vol-total_cmc_box_vol, ' ft^3')
print('Save ', (total_wal_box_vol-total_cmc_box_vol)/total_wal_box_vol*100, '% of volume\n')

print('Dunnage')
print('Total dunnage used by walmart is: ', dunnage, ' ft^3')
print('Total dunage used by cmc is: 0 ft^3')
print('That is ', round(dunnage/order_num,3), ' ft^3 per order')
print('This is roughly ', round(dunnage/order_num/airbag_vol,1), ' airbags per order or ', airbags, " airbags totally\n")

print('Corrugate')
print('Total cardboard used by walmart is: ', corrugate_wal, ' ft^2 or ', wal_weight, ' pound')
print('Total cardboard used by cmc is: ', corrugate_cmc, 'ft^2 or ', cmc_weight, ' pound')
print('Total cardboard saved by cmc is: ', corrugate_wal-corrugate_cmc, ' ft^2 or ', wal_weight-cmc_weight, ' pound')
print('Walmart uses ', round(corrugate_wal/17303,3), 'ft^2 or ', round(wal_weight/17303,3), 'pound of cardboard per order')
print('CMC uses ', round(corrugate_cmc/17303,3), 'ft^2 or ', round(cmc_weight/17303,3), 'pound of cardboard per order')
print('Use ', (corrugate_cmc-corrugate_wal)/corrugate_wal*100, '% more corrugate area or ',(cmc_weight-wal_weight)/wal_weight*100, '% more corrugate weight using the CMC Machines')

# Why are the numbers different in this once comparing to the previous analysis
# changes I made to this file
# 1. 70% more cardboard x <=> walmart box folding assumption wrong
# 2. change calculation to only consider boxes after 5/2/2018 bc
#        a) according to observation the min box width was decreased from 440mm to 250mm after April 15
#        b) according to observation the min box length was decreased from 230mm to 160mm on May 1 by putting
#            the lable horizontally; and changed to 170mm since May 2 onward
# 3. pull actual data from Ant for boxsize
#        a) since they don't scan items coming out from cmc, we can only do that base on the upc. I found the upc of 
#           each order (details can be found in A1 analysis) 
#        b) and find the mode of the boxes used for each upc
# 4. changed the estimation of item size to decrease a couple errors I realized
#        a) some cmc boxes only have one or two dimensions smaller than the cmc min, so i can give a direct estimate
#           for some dimensions of the small items
# 
# After measurement, we can figure out the unit weight of the cmc and walmart boxes
# Walmart: 
# CMC: 
#
# This gives us the result we below
# This is the final version of this analysis unless I realize any major errors


# In[10]:


sum(wms_cw1000_good['item_vol'])*mm3_ft3
sum(wms_cw1000_good['item_vol'])*mm3_ft3/4056


# In[3]:


# this alg finds the carboard used of a walmart box given the three dimensions
def wal_cardboard(dim1, dim2, dim3):
    gaps = 10 # this is just 1cm
    # extra is the final part that connects the cardboards
    # they glue around this
    extra = 40 # this is 4cm
    
    dims = sorted([dim1, dim2, dim3])
    h, w, l = dims[0], dims[1], dims[2]
    
    cardboard = h*(w*2+l*2+gaps*3+extra) + w*(2*w+2*l)
    
    return cardboard

# this alg finds the cardboard used by a cm machine given the three dimensions
def cmc_cardboard(l,w,h):
    gaps = 10
    # dent is the extra on width
    dent = 20 # 2cm
    # constant is the constant extra from the height
    constant = 70 # 7cm
    # this is the closing fold on the final part
    extra = 35 #3.5cm
    
    cardboard = w*(2*h+2*l+extra+3*gaps)+2*(h+constant+dent)*(h*2+l*2)
    
    return cardboard


# In[4]:


# this help us to get the right box to use
def get_right_size(df, box_dim):
    for i,row in df.iterrows():
        print(i)
        if (df['Walmart_Box'][i] == 0):
            # identify the ones that dont fit in the box
            # find the right box we should use
            df['Walmart_Box'][i] = find_right_size(df['itemW'][i], df['itemH'][i], df['itemL'][i], box_dim)
    return df
    
    
def find_right_size(itemW, itemH, itemL, type_dimension):
    for i,row in type_dimension.iterrows():
        box = type_dimension['BOXTYPE'][i]
        height, width, length = type_dimension['height'][i]*25.4, type_dimension['width'][i]*25.4, type_dimension['length'][i]*25.4
        info = {'BOXTYPE': [box], 'itemW': [itemW], 'itemH': [itemH], 'itemL': [itemL], 'width': [width], 'height': [height], 'length': [length]}
        df = pd.DataFrame(data=info)
        if fit(df)['fit'][0] == True:
            return box
    print('The item is too big for all boxes')
    return 'BOX L6-45'


# In[5]:


import math

# this fuction adds one more col for the dataframe to see if the item would if in the recommended box
def fit(df):
    error = 10
    df['fit'] = True
    # finds the vol of the item and the box
    df['vol_item'] = (df['itemL']+error)*(df['itemH']+error)*(df['itemW']+error)
    df['vol_box'] = ((df['height'])*(df['width'])*(df['length']))
    for i in range(len(df)):
        # sort them so it's easy to rank the dimensions
        item_size = sorted([df['itemW'][i]+error,df['itemH'][i]+error,df['itemL'][i]+error])
        box_size = sorted([(df['height'][i]),(df['width'][i]),(df['length'][i])])
        
        # first do the simpliest check, compare vol
        if (df['vol_item'][i] > df['vol_box'][i]):
            df['fit'][i] = False
            continue
        
        # now we case the problem into three cases
        # we try to lay the biggest side of the item on a side of the box
        # we case on the three different sides of the box
        
        # This way we reduce the 3 dimensional problem to a 2 dimensional one
        
        # because we are definitely laying a side of the item flat
        # the area of that side has to be smaller than the biggest side of the area
        if (b_in_a(box_size[2],box_size[1],item_size[2],item_size[1])):
            if (item_size[0] <= box_size[0]):

                continue
                
            
        # see if the biggest side of the item is smaller than the smallest side of the box or second smallest side
        # aka lay smallest side
        if (b_in_a(box_size[1],box_size[0],item_size[2],item_size[1])):
            if (box_size[2] >= item_size[0]):
                
                continue

        # aka lay second smallest side
        if (b_in_a(box_size[2],box_size[0],item_size[2],item_size[1])):
            if (box_size[1] >= item_size[0]):
                continue

        
        
        
        df['fit'][i] = False
    
        
    return df


# according to a theorm I found online
# suppose an a1*a2 rectangle T is given, with the notation arranged so that a1 >= a2
# then a b1*b2 rectangle R given b1>=b2 fits into T if and only if
# 1. b1 <= a1 and b2 <= a2, or
# 2. b1 > a1, b2 <= a2, and ((a1+a2)/(b1+b2))^2 + ((a1-a2)/(b1-b2))^2 >= 2

# the second part of the theorm comes from sin and xos of the rotating angle theta

def b_in_a (a1, a2, b1, b2):
    statement1 = (b1 <= a1 and b2 <= a2)
    statement2 = (b1 > a1 and b2 <= a2 and ((a1+a2)/(b1+b2))**2 + ((a1-a2)/(b1-b2))**2 >= 2)
    if (statement1 or statement2):
        return True
    return False


# In[6]:


# this fuction matches the height, len and width with the correct dimension
# given the datafram
def standardize(df):
    #df = df.rename(index=str, columns={"DEPTH_FT": "LENGTH_FT"})
    # this standardizes the len, width, and height
    # made sure that they are measuring the same dimension
    for i,row in df.iterrows():
        width, length = 0, 0
        l_ft = df['wal_l'][i]
        w_ft = df['wal_w'][i]
        h_ft = df['wal_h'][i]
        iW = df['SizeW'][i]-40 # minus 40 for the dent
        iL = df['SizeL'][i]
        iH = df['SizeH'][i]
        fts = sorted([l_ft, w_ft, h_ft])
        i_s = sorted([iW, iL, iH])
        # min values
        min_ft = fts[0]
        min_i = i_s[0]
        # second smallest
        min2_ft = fts[1]
        min2_i = i_s[1]
        # third smallest aka largest
        min3_ft = fts[2]
        min3_i = i_s[2]
        # rank the dimensions and match them
        # smallest ft is width
        if (min_i == iW):
            df['wal_w'][i] = min_ft
            if (min2_i == iL):
                df['wal_l'][i] = min2_ft
                df['wal_h'][i] = min3_ft
            else:
                df['wal_h'][i] = min2_ft
                df['wal_l'][i] = min3_ft
        # smallest ft is length
        elif (min_i == iL):
            df['wal_l'][i] = min_ft
            if (min2_i == iW):
                df['wal_w'][i] = min2_ft
                df['wal_h'][i] = min3_ft
            else:
                df['wal_h'][i] = min2_ft
                df['wal_w'][i] = min3_ft
        # smallest ft is height
        else:
            df['wal_h'][i] = min_ft
            if (min2_i == iW):
                df['wal_w'][i] = min2_ft
                df['wal_l'][i] = min3_ft
            else:
                df['wal_l'][i] = min2_ft
                df['wal_w'][i] = min3_ft
                
    return df


# In[224]:


print(len(cw1000_normal))
print(len(cw1000_small))
print(len(cw1000_normal)+len(cw1000_small))
print(len(wms_cw1000_new))


# In[19]:


cmc_cardboard(217.932,392.938,177.8)*mm2_ft2
# 8.185 ft^2 <=> 226g = 0.498245 pounds (cmc) => 0.060873 pounds/ft2
cmc_cardboard(219.8,372.7,89.1)*mm2_ft2
# 5.1212 ft^2 <=> 141g = 0.310852 pounds (cmc) => 0.607 pounds/ft2

in2_ft2 = 0.00694444
wmt_cardboard = 10.12*4.92*in2_ft2
# 0.345766445376 ft2 <=> 12g = 0.0264555 pounds (walmart) => 0.0765 pounds /ft2


# In[263]:


wms_cw1000_good


# In[ ]:



sns.distplot(normal_item['pctheight'], rug=True)

