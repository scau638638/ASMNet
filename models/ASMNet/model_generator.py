import random
import dgl
import networkx as nx
import torch
from models.bo.BO_Net_dif_low_chan_vo import BONet_dif_low
from thop import profile
import sys
import os

def calculate_max_matching_score(old_ops, new_ops):
    def parse_ops(ops_str):
        if '+' in ops_str:
            return ops_str.split('+')
        else:
            return [ops_str]

    old_components = parse_ops(old_ops)
    new_components = parse_ops(new_ops)

    #权重
    scores = {
        '3×3conv': 0.4,
        'ReLU': 0.3,
        'BN': 0.3,
        'cbam': 1
    }

    # 计算最长公共子串（必须连续）的分数
    max_score = 0
    for i in range(len(old_components)):
        for j in range(len(new_components)):
            k = 0
            current_score = 0
            while (i + k < len(old_components) and
                   j + k < len(new_components) and
                   old_components[i + k] == new_components[j + k]):
                current_score += scores[old_components[i + k]]
                k += 1
            max_score = max(max_score, current_score)

    return max_score

def model_generator(dif_threshold, network_num_dif_threshold):

    #  basic operation
    conv_gene = [
                 "cbam",
                 "3×3conv",
                 "ReLU",
                 "BN",
                 "BN+3×3conv",
                 "ReLU+3×3conv",
                 "3×3conv+BN",
                 "3×3conv+ReLU",
                 "ReLU+BN",
                 "BN+ReLU",
                 "BN+ReLU+3×3conv",
                 "BN+3×3conv+ReLU",
                 "ReLU+3×3conv+BN",
                 "ReLU+BN+3×3conv",
                 "3×3conv+BN+ReLU",
                 "3×3conv+ReLU+BN"
                 ]

    def new_model():
        #使用二进制存储节点连接    values 0：未连接   1：连接     weights权重
        values = [0, 1]
        weights = [0.2, 0.8]
        node_gene = []

        for i in range(9):
            conv_gene_0_used = False
            layer_genes = []
            for j in range(4):
                gene = conv_gene[random.randint(0, len(conv_gene) - 1)]
                # 检查是否是 cbam
                if gene == 'cbam':
                    if not conv_gene_0_used:
                        layer_genes.append(gene)
                        conv_gene_0_used = True
                    else:
                        # 限制cbam至多使用一次
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
                    node_connect.append(random_numbers)
                    break
        return node_gene, node_connect

    #距离差异性
    def three_path_len(connect):
        graph = dgl.graph([])
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
                return float('inf')

            path_lengths = [len(path) - 1 for path in all_paths]
            short_path = min(path_lengths)  # 最短路径长度
            long_path = max(path_lengths)  # 最长路径长度
            avg_length = sum(path_lengths) / len(path_lengths)  # 平均路径长度

        except nx.NetworkXNoPath:
            return float('inf')  # 如果没有路径，则返回无穷大

        return (short_path + long_path + avg_length) / 3

    #节点差异性
    def calculate_gene_difference(old_node_gene, new_gene):
        total_diff = 0

        min_len = min(len(old_node_gene), len(new_gene))

        for i in range(min_len):
            old_op = old_node_gene[i]
            new_op = new_gene[i]

            score = calculate_max_matching_score(old_op, new_op)
            total_diff += 1 - score

        return total_diff

    # 计算量差异性
    def params_count(old_node_gene, old_node_connect, new_node_gene, new_node_connect):

        # 将标准输出重定向到空文件（解决标准输出问题）
        original_stdout = sys.stdout
        null_output = open(os.devnull, 'w')
        sys.stdout = null_output

        try:
            input_tensor = torch.randn(1, 3, 512, 512)

            model1 = BONet_dif_low(node_gene=old_node_gene, node_connect=old_node_connect)
            model2 = BONet_dif_low(node_gene=new_node_gene, node_connect=new_node_connect)

            flops1, params1 = profile(model1, inputs=(input_tensor,))
            flops2, params2 = profile(model2, inputs=(input_tensor,))

            result = abs(flops1 - flops2)
        finally:
            sys.stdout = original_stdout
            null_output.close()

        return result

    #初始化第一个模型
    network_num = 1
    all_node_gene = []
    all_node_connect = []
    new_gene, new_connet = new_model()
    all_node_gene.append(new_gene)
    all_node_connect.append(new_connet)

    while network_num < network_num_dif_threshold:
        num = 0
        new_gene, new_connet = new_model()

        while num < network_num:
            pathlen = 0
            gene_dif = 0
            old_node_gene = all_node_gene[num]
            old_node_connect = all_node_connect[num]

            for cnt1, cnt2 in zip(old_node_connect, new_connet):
                pathlen += abs(three_path_len(cnt1) - three_path_len(cnt2))

            cql = params_count(old_node_gene, old_node_connect, new_gene, new_connet)

            for gene1, gene2 in zip(old_node_gene, new_gene):
                dif_gene = calculate_gene_difference(gene1, gene2)
                gene_dif += dif_gene

            #  通道数：16-256    多次试验得最大值最小值    （换通道数得重新测量）
            #  gene_dif     19             26.5
            #  pathlen      0.6            6.8
            #  cql          8320.0   7900176448.0
            gene_dif = (gene_dif-19)/(26.5-19)
            pathlen = (pathlen - 0.6)/(6.8-0.6)
            cql = (cql-8320.0)/(7900176448.0-8320.0)

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

def Generater(seach_time, dif_threshold, model_number):
    for i in range(1, seach_time+1):
        all_node_gene, all_node_connect = model_generator(dif_threshold, model_number)

        with open('search{}.txt'.format(i), 'w', encoding='utf-8') as f:
            f.write("all_node_gene:\n")
            for inner_list in all_node_gene:
                f.write(f"{inner_list},")

            f.write("\nall_node_connect:\n")
            for inner_list in all_node_connect:
                f.write(f"{inner_list},")

if __name__ == '__main__':
    Generater(1, 1, 50)
