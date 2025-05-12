import random
import dgl
import networkx as nx

def BOS(dif_threshold, network_num_dif_threshold):
    def new_graph():
        values = [0, 1]
        weights = [0.2, 0.8]       # 30% 选择 0   70% 选择 1
        node_gene = []

        for i in range(9):
            conv_gene_0_used = False  # 追踪 cbam 是否已使用
            layer_genes = []
            for j in range(4):
                gene = conv_gene[random.randint(0, len(conv_gene) - 1)]
                # 检查是否是 cbam
                if gene == 'cbam':
                    if not conv_gene_0_used:
                        layer_genes.append(gene)
                        conv_gene_0_used = True  # 标记为已使用
                    else:
                        # 如果已经使用过，则选择其他基因
                        while gene == 'cbam':
                            gene = conv_gene[random.randint(0, len(conv_gene) - 1)]
                        layer_genes.append(gene)
                else:
                    layer_genes.append(gene)
            node_gene.append(layer_genes)

        node_connect = []

        for i in range(9):
            while True:
                random_numbers = random.choices(values, weights=weights, k=6)

                # 检查是否全为 0
                if random_numbers != [0, 0, 0, 0, 0, 0]:
                    node_connect.append(random_numbers)  # 仅在不是全 0 时添加
                    break  # 退出循环
        return node_gene, node_connect

    def three_path_len(connect):

        graph = dgl.graph([])

        # 添加 6 个节点，节点编号为 0 到 5
        num_nodes = 6
        graph.add_nodes(num_nodes)

        if connect[0] == 1:
            graph.add_edges(1, 2)
        if connect[1] == 1:
            graph.add_edges(1, 3)
        if connect[2] == 1:
            graph.add_edges(1, 4)
        if connect[3] == 1:
            graph.add_edges(2, 3)
        if connect[4] == 1:
            graph.add_edges(2, 4)
        if connect[5] == 1:
            graph.add_edges(3, 4)

        if connect[0] == 1 or connect[1] == 1 or connect[2] == 1:
            graph.add_edges(0, 1)
        if connect[0] == 0 and (connect[3] == 1 or connect[4] == 1):
            graph.add_edges(0, 2)
        if connect[1] == 0 and connect[3] == 0 and connect[5] == 1:
            graph.add_edges(0, 3)

        if connect[0] == 1 and connect[3] == 0 and connect[4] == 0:
            graph.add_edges(2, 5)
        if(connect[1] == 1 or connect[3] == 1) and connect[5] == 0:
            graph.add_edges(3, 5)
        if connect[2] == 1 or connect[4] == 1 or connect[5] == 1:
            graph.add_edges(4, 5)

        # 计算从节点 0 到节点 5 的所有路径
        nx_graph = graph.to_networkx()

        try:
            all_paths = list(nx.all_simple_paths(nx_graph, source=0, target=5))
            if not all_paths:
                return float('inf')  # 如果没有路径，则返回无穷大

            # 计算路径长度
            path_lengths = [len(path) - 1 for path in all_paths]  # 减一以获得边的数量
            short_path = min(path_lengths)  # 最短路径长度
            long_path = max(path_lengths)  # 最长路径长度
            avg_length = sum(path_lengths) / len(path_lengths)  # 平均路径长度

        except nx.NetworkXNoPath:
            return float('inf')  # 如果没有路径，则返回无穷大

        # 返回三个路径长度的平均值
        return (short_path + long_path + avg_length) / 3



    def count_volume(node_gene, node_connect):
        num = 0
        if node_connect[0] == 1 or node_connect[1] == 1 or node_connect[2] == 1:
            num += 1
        if node_connect[0] == 1 or node_connect[3] == 1 or node_connect[4] == 1:
            num += 1
        if node_connect[1] == 1 or node_connect[3] == 1 or node_connect[5] == 1:
            num += 1
        if node_connect[2] == 1 or node_connect[4] == 1 or node_connect[5] == 1:
            num += 1
        for gene in node_gene:
            if gene == 'cbam':
                num -= 1
        return num

    dif_gene = [
        [0,   1,   1,   1,     1,   1],
        [1,   0, 0.3, 0.3,  0.3, 0.6],
        [1, 0.3,   0, 0.3,  0.3,  0.6],
        [1, 0.3, 0.3,   0,  0.3,  0.6],
        [1, 0.3, 0.3, 0.3,    0, 0.6],
        [1, 0.6, 0.6, 0.6,  0.6,    0]
    ]

    conv_gene = [
                 "cbam",
                 "3×3conv",
                 "BN+3×3conv",
                 "ReLU+3×3conv",
                 # "3×3conv+BN",
                 "3×3conv+ReLU",
                 # "BN+ReLU+3×3conv",
                 # "BN+3×3conv+ReLU",
                 # "3×3conv+BN+ReLU",
                 "3×3conv+ReLU+BN"
                 ]

    #初始化第一个架构
    network_num = 1
    all_node_gene = []
    all_node_connect = []
    new_gene, new_connet = new_graph()
    all_node_gene.append(new_gene)
    all_node_connect.append(new_connet)


    while network_num < network_num_dif_threshold:
        num = 0
        new_gene, new_connet = new_graph()

        while num < network_num:
            pathlen = 0
            gene_dif = 0
            cql = 0
            old_cql = []
            new_cql = []
            old_node_gene = all_node_gene[num]
            old_node_connect = all_node_connect[num]

            for cnt1, cnt2 in zip(old_node_connect, new_connet):
                pathlen += abs(three_path_len(cnt1) - three_path_len(cnt2))

            for ong, onc in zip(old_node_gene, old_node_connect):
                old_cql.append(count_volume(ong, onc))
            for nng, nnc in zip(new_gene, new_connet):
                new_cql.append(count_volume(nng, nnc))
            for num1, num2 in zip(old_cql, new_cql):
                cql += abs(num1 - num2)

            for gene1, gene2 in zip(old_node_gene, new_gene):
                for gen1, gen2 in zip(gene1, gene2):
                    index1 = conv_gene.index(gen1)
                    index2 = conv_gene.index(gen2)
                    gene_dif += dif_gene[index1][index2]

            pathlen = (pathlen - 0.3)/11.7
            cql = cql/14
            gene_dif = (gene_dif - 12)/20

            dif = pathlen+cql+gene_dif

            if dif > dif_threshold:
                num += 1
            else:
                break

        if num == network_num:
            all_node_gene.append(new_gene)
            all_node_connect.append(new_connet)
            network_num = network_num + 1
            print("node:", new_gene)
            print("connet:", new_connet)
        else:
            continue

    return all_node_gene, all_node_connect
def GenerateRandomModel(seach_time, dif_threshold, model_number):
    for i in range(1, seach_time+1):
        #差异阈值 0.5  生成架构数量 50
        all_node_gene, all_node_connect = BOS(dif_threshold, model_number)

        with open('search{}.txt'.format(i), 'w', encoding='utf-8') as f:
            f.write("all_node_gene:\n")
            for inner_list in all_node_gene:
                f.write(f"{inner_list},")

            f.write("\nall_node_connect:\n")
            for inner_list in all_node_connect:
                f.write(f"{inner_list},")


if __name__ == '__main__':
    GenerateRandomModel(20, 0.5, 100)
