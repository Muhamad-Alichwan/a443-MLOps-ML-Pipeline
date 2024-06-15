
# Import yang diperlukan
from kerastuner.engine import base_tuner
import kerastuner as kt
from tensorflow import keras
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "real"
FEATURE_KEY = "text"

# callback
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") >= 0.85 and logs.get("val_accuracy") >= 0.85:
            self.model.stop_training = True
callbacks = [MyCallback()]

# Menentukan namedtuple untuk hasil tuner
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs=None,
             batch_size=32)->tf.data.Dataset:
    
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key= transformed_name(LABEL_KEY))
    return dataset

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100
 
vectorize_layer = tf.keras.layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

embedding_dim=16

def model_builder(hp):
    '''
    Membangun model dan menyiapkan hiperparameter yang akan dituning.

    Args:
        hp: Objek tuner Keras.

    Returns:
        Model dengan hiperparameter yang akan dituning.
    '''
    # Input layer for the features
    input_features = tf.keras.Input(shape=(SEQUENCE_LENGTH,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(input_features, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = tf.keras.layers.Embedding(VOCAB_SIZE, embedding_dim, name="embedding")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Menentukan unit untuk dense layer pertama yang akan dituning
    hp_units = hp.Int('units', min_value=64, max_value=215, step=16)
    x = tf.keras.layers.Dense(units=hp_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Mengatur model
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    # Menentukan learning rate untuk optimizer yang akan dituning
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

    # Compile model dengan optimizer dan loss function
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss= keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])

    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """
    Membangun tuner menggunakan API KerasTuner.

    Args:
        fn_args: Menyimpan argumen sebagai pasangan nama/nilai.

    Returns:
        Sebuah namedtuple yang berisi:
          - tuner: Tuner yang akan digunakan untuk tuning.
          - fit_kwargs: Argumen yang akan dilewatkan ke fungsi run_trial tuner
                         untuk fitting model, misalnya, dataset pelatihan dan validasi.
    """

    # Mendefinisikan strategi pencarian tuner
    tuner = kt.RandomSearch(model_builder,
                            objective="val_accuracy",
                            max_trials=5,
                            directory=fn_args.working_dir,
                            project_name='kt_random_search')

    # Load output transform
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Menggunakan input_fn() untuk mengekstrak fitur-fitur input dan label dari set pelatihan dan validasi
    train_set = input_fn(fn_args.train_files, tf_transform_output)
    val_set = input_fn(fn_args.eval_files, tf_transform_output)
    
    # Adapt the TextVectorization layer to the training data
    vectorize_layer.adapt(train_set.take(1).map(lambda x, y: x[transformed_name(FEATURE_KEY)]))

    epochs = 10

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks":[callbacks],
            'x': train_set,
            'epochs': epochs,
            'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        }
    )
