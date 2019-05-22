import tensorflow as tf


def train_input_fn(file_list, params):
    """Train input function for the MNIST dataset.

    Args:
        file_list: (list) path to the image list and label list [record_file_path1, record_file_path2...]
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    def _parse_record(record):
        features = tf.parse_single_example(record, features={
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/filename':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/key/sha256':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/source_id':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/height':
                tf.FixedLenFeature((), tf.int64, 1),
            'image/width':
                tf.FixedLenFeature((), tf.int64, 1),
            # Object boxes and classes.
            'image/object/bbox/xmin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax':
                tf.VarLenFeature(tf.float32),
            'image/object/class/label':
                tf.VarLenFeature(tf.int64),
            'image/object/class/text':
                tf.VarLenFeature(tf.string),
            'image/object/area':
                tf.VarLenFeature(tf.float32),
            'image/object/is_crowd':
                tf.VarLenFeature(tf.int64),
            'image/object/difficult':
                tf.VarLenFeature(tf.int64),
            'image/object/group_of':
                tf.VarLenFeature(tf.int64),
        })
        # image 解析
        image = tf.cast(tf.image.decode_jpeg(features['image/encoded'], 3), tf.uint8)
        # gt_label 解析
        box_label = features['image/object/class/label']
        box_label = tf.sparse_tensor_to_dense(box_label, 0)

        # gt_box 解析
        box_size = [features['image/object/bbox/ymin'], features['image/object/bbox/xmin'],
                    features['image/object/bbox/ymax'], features['image/object/bbox/xmax']]
        sides = []
        for side in box_size:
            side = side.values
            side = tf.expand_dims(side, 0)
            sides.append(side)
        box_bounding = tf.concat(sides, 0)
        box_bounding = tf.transpose(box_bounding)

        return image, box_bounding, box_label

    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(_parse_record)
    dataset = dataset.shuffle(params.min_after_dequeue)  # whole dataset into the buffer
    dataset = dataset.repeat(params.num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(1)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    iterator = dataset.make_one_shot_iterator()
    batch_image, batch_box, batch_label = iterator.get_next()
    return batch_image, [batch_box, batch_label]
