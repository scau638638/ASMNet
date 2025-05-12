import ast


def get_gene_cnt():
    with open(r'C:\Users\chenrui\Desktop\BONet-5\models\bo\search.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    # 提取all_node_gene
    start_index = content.find('all_node_gene:')
    end_index = content.find('all_node_connect:')
    all_node_gene_str = content[start_index + len('all_node_gene:'):end_index].strip()
    all_node_gene = ast.literal_eval(all_node_gene_str)

    # 提取all_node_connect
    start_index = content.find('all_node_connect:')
    all_node_connect_str = content[start_index + len('all_node_connect:'):].strip()
    all_node_connect = ast.literal_eval(all_node_connect_str)

    return all_node_gene, all_node_connect

def get_gene_cnt_i(i):
    with open(r'C:\Users\chenrui\Desktop\BONet-5\models\bo\search{}.txt'.format(i), 'r', encoding='utf-8') as file:
        content = file.read()

    # 提取all_node_gene
    start_index = content.find('all_node_gene:')
    end_index = content.find('all_node_connect:')
    all_node_gene_str = content[start_index + len('all_node_gene:'):end_index].strip()
    all_node_gene = ast.literal_eval(all_node_gene_str)

    # 提取all_node_connect
    start_index = content.find('all_node_connect:')
    all_node_connect_str = content[start_index + len('all_node_connect:'):].strip()
    all_node_connect = ast.literal_eval(all_node_connect_str)

    return all_node_gene, all_node_connect
