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
values = np.array((268814, 229262856, 35228303, 20217782, 1150188, 3838590, 142051812, 15539169, 13295549, 3026753, 711029194, 215685325, 57806647, 467720714, 86985))
indices = np.argsort(values)[::-1]
values = list(values)

keys = [keys[i] for i in indices]
values = [values[i] for i in indices]
plt.bar(keys, values, color = 'maroon', width=0.8)
plt.xticks(range(len(keys)), keys, rotation=60)
plt.xlabel('Category Name', fontsize=12)
plt.ylabel('Population', fontsize=12)
# plt.show()

plt.savefig('./class_hist.png', bbox_inches='tight')

aaa=1