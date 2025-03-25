import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_prototypes(prototypes, queries, query_labels, dataset):
    prototypes = prototypes.detach().cpu() # 5 dim
    query = queries.detach().cpu() # queries dim
    query_labels = query_labels.detach().cpu()  # queries 1
    
    pca = PCA(n_components=2)
    all_data = torch.cat([prototypes, query], dim=0)  # 프로토타입과 쿼리를 합쳐서 변환
    all_data_pca = pca.fit_transform(all_data)

    num_prototypes = prototypes.shape[0]
    prototypes_pca = all_data_pca[:num_prototypes]
    query_pca = all_data_pca[num_prototypes:]

    plt.figure(figsize=(8, 6))
    colormap = plt.get_cmap('viridis', num_prototypes)  # 클래스별 색상 설정

    # 프로토타입: 원형 (circle)
    for i in range(num_prototypes):
        plt.scatter(prototypes_pca[i, 0], prototypes_pca[i, 1], 
                    color=colormap(i), s=100, marker='o', label=f"Prototype {i}")

    # 쿼리: 삼각형 (triangle), 같은 클래스면 같은 색상
    query_labels = [query_labels]
    for i in range(len(query_labels)):
        label = query_labels[i].item()  # 해당 쿼리 샘플의 클래스
        plt.scatter(query_pca[i, 0], query_pca[i, 1], 
                    color=colormap(label), s=100, marker='^', label=f"Query {label}")

    for i in range(num_prototypes):
        plt.annotate(str(i), (prototypes_pca[i, 0], prototypes_pca[i, 1]), fontsize=12, ha='right')

    plt.title('PCA Projection of Prototypes and Query')
    plt.legend()
    
    plt.savefig(f"img/{dataset}_prototypes_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"저장 완료: img/{dataset}_prototypes_visualization.png")
