import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
import os
import tensorflow_hub as hub
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "real"
FEATURE_KEY = "text"

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") >= 0.80 and logs.get("val_accuracy") >= 0.80:
            self.model.stop_training = True
callbacks = [MyCallback()]

# "Fungsi ini digunakan untuk mengubah nama fitur yang telah melalui proses transform."
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

# "Ia merupakan fungsi yang digunakan untuk memuat data dalam format TFRecord."
def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

# "Ia digunakan untuk memuat transformed_feature yang dihasilkan oleh komponen Transform dan membaginya ke dalam beberapa batch."
def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transformed_name(LABEL_KEY))
    return dataset

# os.environ['TFHUB_CACHE_DIR'] = '/hub_chace'
# embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")
 
# Vocabulary size and number of words in a sequence.
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100
 
vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

embedding_dim=16

# "Fungsi inilah yang bertanggung jawab dalam membuat arsitektur model. Pada latihan ini, kita menggunakan salah satu embedding layer yang tersedia dan dapat diunduh melalui TensorFlow Hub."
def model_builder(hp):
    """Build machine learning model"""
    
    # Input layer for the features
    input_features = tf.keras.Input(shape=(SEQUENCE_LENGTH,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(input_features, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim, name="embedding")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Menentukan unit untuk dense layer pertama yang akan dituning
    hp_units = hp.get('units')
    x = tf.keras.layers.Dense(units=hp_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_features, outputs = outputs)

    hp_learning_rate = hp.get('learning_rate')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss= tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

    # print(model)
    model.summary()
    return model

# "Fungsi ini digunakan untuk menjalankan tahapan preprocessing data pada raw request data."
def _get_serve_tf_examples_fn(model, tf_transform_output):

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):

        feature_spec = tf_transform_output.raw_feature_spec()

        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        # get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn

# "Ia merupakan fungsi yang bertanggung jawab untuk menjalankan proses training model sesuai dengan parameter training yang diberikan."
def run_fn(fn_args: FnArgs) -> None:

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)]])
    hyper = fn_args.hyperparameters.get('values')
    # Build the model
    model = model_builder(hyper)


    # Train the model
    model.fit(
        x=train_set,
        epochs = 10,
        validation_data=val_set,
        callbacks = callbacks
    )
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
