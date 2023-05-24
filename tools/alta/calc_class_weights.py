import numpy as np

num_classes=16
AgamimPathA_hist = [31522, 383468, 227730863, 35295341, 18005882, 1173707, 4244303, 125600525, 13837139, 11737163, 5778866, 714936537, 234801492, 58013721, 464674200, 93447, ]
AgamimPathB_hist = [112041, 117317, 164123392, 57421339, 20429650, 105651, 5048636, 328455851, 21792463, 96998734, 1634637, 781109580, 473067269, 55777244, 409130156, 60616, ]
AgamimPathC_hist = [99561, 97745, 199139015, 24103206, 21193658, 94631, 4143944, 109727685, 18196612, 308767480, 916901, 663199290, 565340325, 36926677, 203529072, 404646, ]
IrYamim_hist = [8434904, 370796, 812491000, 148552083, 91124329, 2123476, 16455715, 294282912, 224223995, 899371102, 13874347, 598347326, 548039184, 59176631, 2230630196, 1135092, ]
Descend_hist = [226192, 936721, 933172479, 93315964, 62967989, 1575349, 11840611, 494084802, 76740774, 328050796, 3451162, 1799756512, 776361480, 126191313, 1178829338, 1246038, ]  # without 0001, 0035, 0036, pre-sampled 1:5
# TODO:
#  Unfortunately, it seems that I used the hist of IrYamim instead of Descend for training the models for the CVPR paper.
#  After calculating this correctly, it seems that the difference is not negligible, yet not huge either

total_hist = np.array([(AgamimPathA_hist[i]+AgamimPathB_hist[i]+AgamimPathC_hist[i]+Descend_hist[i]) for i in range(num_classes)])

# weight_method = 'equal'
# weight_method = 'sqrt'
weight_method = 'prop'
if weight_method == 'equal':
    final_hist = np.ones(len(total_hist))
elif weight_method == 'sqrt':
    final_hist = np.sqrt(total_hist)
elif weight_method == 'prop':
    final_hist = total_hist

class_weight = 1/final_hist
class_weight[0] = 0  # Background class eliminated

class_weight /= (class_weight.sum() / (num_classes-1))
print(class_weight)
print(class_weight.sum())
