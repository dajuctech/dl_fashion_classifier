import kerastuner as kt
from src.model.cnn_model import get_callbacks
from tensorflow.keras.datasets import fashion_mnist

def build_model(hp):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv_1_kernel', values=[3,5]),
        activation='relu',
        input_shape=(28,28,1)
    ))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(
        units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
        activation='relu'
    ))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run_tuning():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train, x_train = x_train / 255.0, x_test / 255.0
    x_train, x_train = x_train[..., None], x_test[..., None]

    tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10, directory='tuner_dir', project_name='fashion_mnist')
    tuner.search(x_train, y_train, epochs=10, validation_split=0.2, callbacks=get_callbacks())

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()
    best_model.save('models/fashion_cnn_best_tuned.h5')

if __name__ == "__main__":
    run_tuning()
