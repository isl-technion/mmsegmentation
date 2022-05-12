import os

save_dir = '/home/airsim/repos/open-mmlab/mmsegmentation/results/' + self._loss_name + '/class_{}/'.format(c)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

for f in range(0, 256, 1):
    save_path = os.path.join(save_dir, 'dir_{}.png'.format(f))
    plt.plot(hist_values_for_loss[f]);
    plt.plot(target_values[0].cpu());
    plt.title(
        'c={}, dir={}, kurt = {:.3f}, mom2 = {}'.format(c, f, kurtosis[f], var_curr_t[f]));
    plt.savefig(save_path);
    plt.close();

for f in range(0, 10000, 100):
    save_path = os.path.join(save_dir, 'dir_{}.png'.format(f))
    plt.plot(hist_values_for_loss[f]);
    plt.plot(target_values[0].cpu());
    plt.title(
        'c={}, dir={}, kurt = {:.3f}, mom2 = {}'.format(c, f, kurtosis[f], var_curr_t[f]));
    plt.savefig(save_path);
    plt.close();