import re
import json
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from scipy import spatial
from collections import Counter

class Clustering:

    edge_list = None
    items = None
    attributes = None
    values = None
    item_mapping = None
    attribute_mapping = None
    connectivity_m = None
    similarity_m = None
    clusters = None

    def __parse_lines(self, line_list):
        links = []
        for line in line_list:
            matching = re.match('([0-9]+),\"(.*)\"',
                                line)  # match item_id and the json, then separate
            links.append(
                (matching.group(1), json.loads(matching.group(2).replace('\"\"', '\"'))))
        return links

    def __generate_edge_list(self, links):
        # from-to:value
        self.edge_list = []
        # unpack values
        for link in links:
            for attribute in link[1]['attributes']:
                # print(attribute)
                a_id = attribute['attr_id']
                values = attribute['values']
                if len(values) == 1:
                    v = values[0]['v']
                    self.edge_list.append((link[0], str(a_id), v))
                else:
                    values_dict = {}
                    # for value in values:
                    # values_dict[value['node_id']] = value['v']
                    #             item_attribute.append((link[0], a_id, values_dict))

                    # trick: we dont need a list for nodes, handle nodes as attributes
                    for value in values:
                        self.edge_list.append((link[0], str(value['node_id']), value['v']))

        self.items, self.attributes, self.values = zip(*self.edge_list)

    def __generate_mappings(self):
        self.item_mapping = list(set(self.items))
        self.item_mapping.sort()
        self.attribute_mapping = list(set(self.attributes))
        self.attribute_mapping.sort()

    def __map_edges(self):
        # there is no id overlapping between items and attributes
        # print(set(attribute_mapping).intersection(set(item_mapping)))

        # little bit slow
        new_edge_list = []
        for edge in self.edge_list:
            new_item = self.item_mapping.index(edge[0])
            new_attribute = self.attribute_mapping.index(edge[1])
            new_edge_list.append((new_item, new_attribute, edge[2]))

        self.edge_list = new_edge_list
        self.items, self.attributes, self.values = zip(*self.edge_list)

    def __generate_connectivity_m(self):
        self.connectivity_m = csr_matrix((self.values, (self.items, self.attributes)),
                                    shape=(len(self.item_mapping),
                                           len(self.attribute_mapping))).toarray()

    def __generate_similarity_m(self):
        #similarity : cosine
        self.similarity_matrix = np.zeros((len(self.item_mapping), len(self.item_mapping)))
        np.fill_diagonal(self.similarity_matrix, 1)
        for i in range(len(self.item_mapping) - 1):
            ss = 1 - spatial.distance.cosine(self.connectivity_m[i,:],
                                   self.connectivity_m[i+1,:])
            self.similarity_matrix[i][i+1] = ss
            self.similarity_matrix[i+1][i] = ss

    def generate_clusters(self, eps=0.3, min_samples=5):
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', algorithm='brute').\
            fit(self.connectivity_m)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('%d cluster found...' % n_clusters)
        print('Cluster mapping: ')
        self.clusters = {}
        for c_num in set(labels):
            cluster_indexes = list(np.where(np.array(labels) == c_num)[0])
            print('Cluster #%d:' % c_num)
            print(np.array(self.item_mapping)[cluster_indexes])
            self.clusters[c_num] = list(np.array(self.item_mapping)[cluster_indexes])
        print('Number of items in clusters: ')
        print(Counter(labels))

    def get_clusters(self):
        return self.clusters

    def get_similarity_m(self):
        return self.similarity_matrix

    def recommend_n(self, product_id, n):
        idx = self.item_mapping.index(str(product_id))
        item_line = self.similarity_matrix[idx,:]
        item_line[idx] = 0
        sorted_idxs = np.argsort(item_line)[::-1]
        sorted_line = np.array(self.item_mapping)[sorted_idxs][:n]
        print('Recommendations for item %s' % str(product_id))
        print(sorted_line)

    def __init__(self, filename='magia_cluster_data'):
        self.source_file = filename
        # read file and create link list
        with open(self.source_file) as f:
            lines = f.readlines()[1:]  # dont need header

        links = self.__parse_lines(lines)
        self.__generate_edge_list(links)
        # create mapping for items and attributes
        self.__generate_mappings()
        self.__map_edges()
        self.__generate_connectivity_m()
        self.__generate_similarity_m()
