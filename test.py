from clustering import Clustering

cl = Clustering('magia_cluster_data')

cl.generate_clusters(eps=0.3, min_samples=5)

cl.recommend_n(5931298, 4)