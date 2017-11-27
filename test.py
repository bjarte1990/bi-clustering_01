from clustering import Clustering

cl = Clustering('magia_cluster_data')
#cl = Clustering('data_sample')
#cl.generate_clusters(eps=0.5, min_samples=5)

#cl.recommend_n(5931298, 4)

cl.spectral(8)