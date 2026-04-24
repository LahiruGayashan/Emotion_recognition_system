from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()


# ---------- LOAD GRAPH ----------
def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with tf.gfile.GFile(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    return graph


# ---------- IMAGE PREPROCESS ----------
def read_tensor_from_image_file(
        file_name,
        input_height=224,
        input_width=224,
        input_mean=128,
        input_std=128):

    file_reader = tf.read_file(file_name)
    image_reader = tf.image.decode_jpeg(file_reader, channels=3)
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    with tf.Session() as sess:
        result = sess.run(normalized)

    return result


# ---------- LOAD LABELS ----------
def load_labels(label_file):
    return [line.strip() for line in tf.gfile.GFile(label_file).readlines()]


# ---------- PREDICT (USED BY API) ----------
def predict(image_path):
    model_file = "retrained_graph.pb"
    label_file = "retrained_labels.txt"

    graph = load_graph(model_file)
    image_tensor = read_tensor_from_image_file(image_path)

    input_operation = graph.get_operation_by_name("input")
    output_operation = graph.get_operation_by_name("final_result")

    with tf.Session(graph=graph) as sess:
        results = sess.run(
            output_operation.outputs[0],
            {input_operation.outputs[0]: image_tensor}
        )

    results = np.squeeze(results)
    labels = load_labels(label_file)

    return labels[np.argmax(results)]