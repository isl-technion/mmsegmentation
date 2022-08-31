import matplotlib.pyplot as plt
import numpy as np

keys = [
    # 'background',  # 0
    'bicycle',  # 1
    'building',  # 2
    'fence',  # 3
    'other',  # 4
    'person',  # 5
    'pole',  # 6
    'rough terr.',  # 7
    'shed',  # 8
    'soft terr.',  # 9
    'stairs',  # 10
    'trans. terr.',  # 11
    'vegetation',  # 12
    'vehicle',  # 13
    'walking terr.',  # 14
    'water',  # 15
]
# values = np.array((268814, 229262856, 35228303, 20217782, 1150188, 3838590, 142051812, 15539169, 13295549, 3026753, 711029194, 215685325, 57806647, 467720714, 86985))
values = np.array((5021032,  5473128537,   608690312,   398167109,     9009185, 79645243,  3377697914,   454961409,  2272579525,    26250093, 11940611075,  5477815980,   809150593,  7480051017,     6935121))
indices = np.argsort(values)[::-1]
values = list(values)

keys = [keys[i] for i in indices]
values = [values[i] for i in indices]

plt.figure()
plt.bar(keys, values, color = 'maroon', width=0.8)
plt.xticks(range(len(keys)), keys, rotation=60)
plt.xlabel('Category Name', fontsize=12)
plt.ylabel('Population', fontsize=12)
# plt.show()
plt.savefig('./class_hist.png', bbox_inches='tight')
plt.close()

plt.figure()
small_classes_num = 5
plt.bar(keys[-small_classes_num:], values[-small_classes_num:], color = 'maroon', width=0.8)
plt.xticks(range(small_classes_num), keys[-small_classes_num:], rotation=60)
plt.xlabel('Category Name', fontsize=12)
plt.ylabel('Population', fontsize=12)
# plt.show()
plt.savefig('./small_class_hist.png', bbox_inches='tight')
plt.close()

aaa=1