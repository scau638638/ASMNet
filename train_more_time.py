
import subprocess
import csv
from collections import Counter
from models.bo.util import get_gene_cnt,get_gene_cnt_i
def count_all_gene(all_gene):
    #统计模型每种基因的数量
    keywords = [
        "cbam", "3×3conv", "BN+3×3conv", "ReLU+3×3conv",
        "3×3conv+ReLU",  "3×3conv+ReLU+BN"
    ]
    all_items = [item for sublist in all_gene for item in sublist]
    counter = Counter(all_items)
    return [counter[keyword] for keyword in keywords]
def count_all_connect(all_connect):
    #统计连接数
    return sum(sum(sublist) for sublist in all_connect)
def get_f1_iou(i,n):
    # 读取CSV文件
    with open(r'C:\Users\chenrui\Desktop\BONet-5\exps\BONet-CFD\csv\performance_{}_{}.csv'.format(i,n), 'r') as file:
        csv_reader = csv.DictReader(file)
        data = list(csv_reader)

    # 提取epoch、f1_score和iou的值
    epoch = int(data[0]['epoch'])
    f1_score = float(data[0]['f1_score'])
    iou = float(data[0]['iou'])

    return epoch, f1_score, iou
def run_script_multiple_times(script_path, n):
    for _ in range(n):
        # 启动并等待脚本完成
        process = subprocess.run(['python', script_path])
        if process.returncode != 0:
            print("Script failed with return code:", process.returncode)
        else:
            print("Script completed successfully.")

if __name__ == "__main__":

        # best 5-13  0.922
        #  batch_size = 4
        # lr = 0.0001
        # weight_decay = 0.0004
        # momentum = 0.9
        # optimizer = 'Adam'
        # loss_func = 'wbce_dice_loss'

    # best = [
    #     [],
    #     [79,20,39,62,69,40,37,18,83,9,75,44,84,64],
    #     [2,50,72,99,57,84,8,85],
    #     [47,43,55,38,77,34,87,5,70,64,73,65,22,26,62,12,48],
    #     [80,23,78,71,72,9,5,86,24,13,91,18,53,7],
    #     [59,94,10,49,95,54,74,44,0,79,60,69,24],
    # ]
    #
    # group_num = 1
    # for group in best:
    #     with open(r'C:\Users\chenrui\Desktop\BONet-5\models\bo\train_model_num.txt', 'w') as file:
    #         file.write(str(group_num))
    #     group_num = group_num + 1
    #     for n in group:
    #         with open(r'C:\Users\chenrui\Desktop\BONet-5\models\bo\temp_num.txt', 'w') as file:
    #             file.write(str(n))
    #         run_script_multiple_times('train_model.py', 1)



    for i in range(1, 6):
        with open(r'C:\Users\chenrui\Desktop\BONet-5\models\bo\train_model_num.txt', 'w') as file:
            file.write(str(i))
        for n in range(0, 100):
            with open(r'C:\Users\chenrui\Desktop\BONet-5\models\bo\temp_num.txt', 'w') as file:
                file.write(str(n))
            run_script_multiple_times('train_model.py', 1)

    # run_script_multiple_times('train_model.py', 1)
        all_gene, all_connect = get_gene_cnt_i(i)

        # 打开CSV文件并写入表头
        # with open(r'C:\Users\chenrui\Desktop\测试结果\test11\基因.csv', 'w', newline='') as file:
        with open(r'C:\Users\chenrui\Desktop\测试结果\dif_test_{}\total.csv'.format(i), 'w', newline='') as file:
            fieldnames = ['ID', 'cbam', '3×3conv', 'BN+3×3conv', 'ReLU+3×3conv', '3×3conv+ReLU',
                           '3×3conv+ReLU+BN', 'connect_num', 'F1-score', 'iou']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for n in range(0, 100):
                gene = all_gene[n]
                connect = all_connect[n]

                # 获取epoch、f1_score和iou值
                epoch, f1_score, iou = get_f1_iou(i, n)

                # 统计连接数
                connect_num = count_all_connect(connect)

                # 统计每种基因的数量
                gene_counts = count_all_gene(gene)

                # 将数据写入CSV文件
                writer.writerow({
                    'ID': n,
                    'cbam': gene_counts[0],
                    '3×3conv': gene_counts[1],
                    'BN+3×3conv': gene_counts[2],
                    'ReLU+3×3conv': gene_counts[3],
                    '3×3conv+ReLU': gene_counts[4],
                    '3×3conv+ReLU+BN': gene_counts[5],
                    'connect_num': connect_num,
                    'F1-score': f1_score,
                    'iou': iou
                })
