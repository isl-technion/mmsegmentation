import matplotlib.pyplot as plt
import numpy as np

keys = [
    'void',  # 0
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
num_classes=16
AgamimPathA_hist = [31522, 383468, 227730863, 35295341, 18005882, 1173707, 4244303, 125600525, 13837139, 11737163, 5778866, 714936537, 234801492, 58013721, 464674200, 93447, ]
AgamimPathB_hist = [112041, 117317, 164123392, 57421339, 20429650, 105651, 5048636, 328455851, 21792463, 96998734, 1634637, 781109580, 473067269, 55777244, 409130156, 60616, ]
AgamimPathC_hist = [99561, 97745, 199139015, 24103206, 21193658, 94631, 4143944, 109727685, 18196612, 308767480, 916901, 663199290, 565340325, 36926677, 203529072, 404646, ]
Descend_hist = [1428963, 4621211, 5100072355, 539554508, 401859215, 9077787, 81155457, 3257125188, 410461075, 2558784243, 17916206, 11521617835, 4973968734, 760943821, 7284456135, 6390867, 1428963, ]
values = np.array([(AgamimPathA_hist[i]+AgamimPathB_hist[i]+AgamimPathC_hist[i]+Descend_hist[i]) for i in range(num_classes)])
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
small_classes_num = 6
plt.bar(keys[-small_classes_num:], values[-small_classes_num:], color = 'maroon', width=0.8)
plt.xticks(range(small_classes_num), keys[-small_classes_num:], rotation=60)
plt.xlabel('Category Name', fontsize=12)
plt.ylabel('Population', fontsize=12)
# plt.show()
plt.savefig('./small_class_hist.png', bbox_inches='tight')
plt.close()

aaa=1