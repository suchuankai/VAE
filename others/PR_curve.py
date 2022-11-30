import matplotlib.pyplot as plt


class PR_curve():
    def __init__(self, path):

        estrus_data, normal_data = self.read_txt(path=path)
        recall , precision = self.cal_PR(estrus_data=estrus_data, normal_data=normal_data)
        self.plot_PR(recall=recall, precision=precision)

    def read_txt(self, path):
        mode = 0
        estrus_data = []
        normal_data = []
        with open(path, 'r') as f:
            for line in f.readlines():
                if "Normal" in line:
                    mode = 1
                if "estrus total" in line:
                    estrus_total = line.split('=')[1].strip()
                    print(estrus_total)
                if "normal_total" in line:
                    normal_total = line.split('=')[1].strip()
                    print(normal_total)
                if(mode == 0 and '%' in line):
                    g1 = line.split(':')[1].strip()
                    estrus_data.append(int(g1))
                if(mode == 1 and '%' in line): 
                    g1 = line.split(':')[1].strip()
                    normal_data.append(int(g1))
        return estrus_data, normal_data

    def cal_PR(self, estrus_data, normal_data):
        TP = [0] * 11
        FP = [0] * 11
        FN = [0] * 11
        TN = [0] * 11
        recall = [0] * 11
        precision = [0] * 11

        estrus_total = 53
        normal_total = 8280

        for i in range(11):     # caculate 11 value
            for j in range(i,10):
                TP[i] += estrus_data[j] 
                FP[i] += normal_data[j]
            TP[i] = TP[i] / estrus_total
            FP[i] = FP[i] / normal_total
            FN[i] = 1 - TP[i]
            TN[i] = 1 - FP[i]

        for i in range(11):
            if((TP[i]+FN[i])==0 and (TP[i]+FP[i])==0):
                recall[i] = 0
                precision[i] = 0
            elif((TP[i]+FN[i])==0 and (TP[i]+FP[i])!=0):
                recall[i] = 0
                precision[i] = TP[i] / (TP[i]+FP[i])
            elif((TP[i]+FN[i])!=0 and (TP[i]+FP[i])==0):
                recall[i] =    TP[i] / (TP[i]+FN[i])
                precision[i] = 1
            else:
                recall[i] =    TP[i] / (TP[i]+FN[i])
                precision[i] = TP[i] / (TP[i]+FP[i])
        return recall, precision

    def plot_PR(self, precision, recall):
        max = 0
        max_index = 0
        max_precision = 0
        max_recall = 0
        for i in range(len(precision)):
            if precision[i]+recall[i] > max:
                max = precision[i]+recall[i]
                max_precision = precision[i]
                max_recall    = recall[i]
                max_index     = i / 10

        print(max,max_precision,max_recall)

        plt.title("Testing data PR Curve")
        plt.plot(recall, precision, color = 'blue',marker='o')

        plt.plot([max_recall], [max_precision], 'o', color = 'red') 
        plt.annotate('threshold = '+ str(max_index), xy=(max_recall, max_precision), xytext=(max_recall-0.3, max_precision-0.1),
                    xycoords='data',
                    color = 'red',
                    arrowprops = dict(arrowstyle='->', color='red', linewidth=3, mutation_scale=25)
                    )
        plt.xlabel('recall') 
        plt.ylabel('precision') 
        plt.show()






