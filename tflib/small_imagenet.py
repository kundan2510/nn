import numpy as np
import scipy.misc
import time

from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.datasets.hdf5 import H5PYDataset

data_path = ''



"This files has many bugs right now. Need to be changed!!!"

def make_generator(stream, batch_size):
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
    data_imagenet = H5PYDataset(data_path, which_sets=('train',), sources=('features',))
    train_data = DataStream(
        data_imagenet,
        iteration_scheme=ShuffledScheme(data_imagenet.num_examples, batch_size)
    )

    data_valid = H5PYDataset(data_path, which_sets=('valid',), sources=('features',))
    dev_data = DataStream(
        data_valid,
        iteration_scheme=ShuffledScheme(data_valid.num_examples, batch_size)
    )
    return (
        make_generator(train_data),
        make_generator(data_valid)
    )

if __name__ == '__main__':

    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()
