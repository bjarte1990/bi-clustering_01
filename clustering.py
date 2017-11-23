import re
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy import spatial

#filename = 'data_sample'
filename = 'magia_cluster_data'
with open(filename) as f:
    lines = f.readlines()[1:] # dont need header

links = []
for line in lines:
    matching = re.match('([0-9]+),\"(.*)\"', line) # match item_id and the json, then separate
    links.append((matching.group(1), json.loads(matching.group(2).replace('\"\"', '\"'))))

#from-to:value
edge_list = []

# unpack values
for link in links:
    for attribute in link[1]['attributes']:
        #print(attribute)
        a_id = attribute['attr_id']
        values = attribute['values']
        if len(values) == 1:
            v = values[0]['v']
            edge_list.append((link[0], str(a_id),v))
        else:
            values_dict = {}
            # for value in values:
            # values_dict[value['node_id']] = value['v']
            #             item_attribute.append((link[0], a_id, values_dict))

            #trick: we dont need a list for nodes, handle nodes as attributes
            for value in values:
                edge_list.append((link[0], str(value['node_id']), value['v']))

items,attributes,values = zip(*edge_list)

# create mapping for items and attributes
item_mapping = list(set(items))
item_mapping.sort()
attribute_mapping = list(set(attributes))
attribute_mapping.sort()

#there is no id overlapping between items and attributes
#print(set(attribute_mapping).intersection(set(item_mapping)))

# little bit slow
new_edge_list = []
for edge in edge_list:
    new_item = item_mapping.index(edge[0])
    new_attribute = attribute_mapping.index(edge[1])
    new_edge_list.append((new_item, new_attribute, edge[2]))

new_items, new_attributes, new_values = zip(*new_edge_list)

connectivity_m = coo_matrix((new_values, (new_items, new_attributes)), shape=(len(item_mapping),
                                                             len(attribute_mapping))).toarray()

#print(sparse_m.toarray())
#similarity : cosine
similarity_matrix = np.zeros((len(item_mapping), len(item_mapping)))
np.fill_diagonal(similarity_matrix, 1)
for i in range(len(item_mapping) - 1):
    ss = 1- spatial.distance.cosine(connectivity_m[i,:],
                           connectivity_m[i+1,:])
    similarity_matrix[i][i+1] = ss
    similarity_matrix[i+1][i] = ss

print(similarity_matrix)