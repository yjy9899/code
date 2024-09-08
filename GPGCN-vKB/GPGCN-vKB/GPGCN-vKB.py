import csv
import numpy as np
from ampligraph.latent_features import ConvKB
from ampligraph.datasets import load_wn18
from ampligraph.evaluation import evaluate_performance

def get_data(path):
    flie = path
    f = open(flie, "r")
    l = []
    for line in f:
        data = line.strip()
        l.append(data)
    x = np.array(l)
    ll = []
    for i in x:
        res = i
        res = res.replace('[', '')
        res = res.replace(']', '')
        ll.append([res])
    lll = []
    for i in ll:
        res = i[0].replace('\'', '')
        lll.append(res.split(',')[:3])
    return lll

model = ConvKB(batches_count=2, seed=22, epochs=100, k=180, eta=1,
              embedding_model_params={'num_filters': 32, 'filter_sizes': [1],
                                       'dropout': 0.1},
               optimizer='adam', optimizer_params={'lr': 0.001},
               loss='pairwise', loss_params={}, verbose=True)

# X = load_wn18()
my_x = get_data("data8/traindata2.txt")

my_test_x = get_data("data8/testdata_fenlei_CVEandCAPEC.txt")

my_x = np.array(my_x)
my_test_x = np.array(my_test_x)

model.fit(my_x)
t_pre_before = model.predict(my_test_x)
# model.fit(X['train'])
print(t_pre_before)
ranks = evaluate_performance(my_test_x, model, filter_triples=my_test_x, use_default_protocol=True)

# 计算MR
mr = np.mean(ranks)
print(f"Mean Rank (MR): {mr}")

# 计算Hit@10
hits_at_10 = np.mean(ranks <= 10)
print(f"Hit@10: {hits_at_10}")
# writer.writerow(["{:.1f}".format(5), "{:.1f}".format(150),
#                 "{:.1f}".format(400),  "{:.4f}".format(mr),
#                 "{:.4f}".format(hits_at_10)])
# print(model.predict(X['test'][:5]))
# 计算Hit@3
hits_at_3 = np.mean(ranks <= 3)
print(f"Hit@3: {hits_at_3*10}")
