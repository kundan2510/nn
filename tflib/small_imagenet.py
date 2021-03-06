import numpy as np
import scipy.misc
import time
import os

BASE_DATA_PATH = '/scratch/jvb-000-aa/faruk/smallimagenet'

def make_generator(path, n_files, batch_size):
    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        files = range(n_files)
        random_state = np.random.RandomState(42)
        random_state.shuffle(files)
        for n, i in enumerate(files):
            image = scipy.misc.imread("{}/{}.png".format(path, str(i+1).zfill(len(str(n_files)))))
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size):
    return (
        make_generator(os.path.join(BASE_DATA_PATH, 'train_64x64'), 1281149, batch_size),
        make_generator(os.path.join(BASE_DATA_PATH, 'valid_64x64'), 49999, batch_size)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
