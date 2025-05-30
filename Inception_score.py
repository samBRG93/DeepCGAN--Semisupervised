import os
import sys
import tarfile
import glob
import math
import numpy as np
import tensorflow as tf
import imageio  # for reading images

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

softmax = None


def get_inception_score(images, splits=10):
    assert isinstance(images, list)
    assert isinstance(images[0], np.ndarray)
    assert len(images[0].shape) == 3
    assert np.max(images[0]) > 10
    assert np.min(images[0]) >= 0.0

    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))

    bs = 100
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))

    with tf.compat.v1.Session() as sess:
        for i in range(n_batches):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'ExpandDims:0': inp})
            preds.append(pred)
    preds = np.concatenate(preds, 0)

    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = tf.keras.utils.get_file(fname=filename, origin=DATA_URL, cache_dir=MODEL_DIR, cache_subdir='')
        print()
    # Extract the tar file
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)

    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.compat.v1.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape().as_list()
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o._shape = tf.TensorShape(new_shape)  # might trigger warnings but needed for dynamic batch
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3), w)
        softmax = tf.nn.softmax(logits)


if __name__ == '__main__':
    tf.compat.v1.reset_default_graph()
    if softmax is None:
        _init_inception()


    def get_images(filename):
        return imageio.imread(filename)


    path = r'C:\Users\samuele\Desktop\working_gan_thesis\EuroSat_GAN\saved_images\data'
    filenames = glob.glob(os.path.join(path, '*.*'))

    images = [get_images(fn) for fn in filenames]
    print(f'Number of images loaded: {len(images)}')
    mean_score, std_score = get_inception_score(images)
    print(f'Inception Score: {mean_score} ± {std_score}')
