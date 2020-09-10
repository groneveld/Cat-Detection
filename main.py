import os
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train_dir = 'data/train/'
    test_dir = 'data/test/'
    categories = ['cat', 'non_cat']
    train_cats_dir = os.path.join(train_dir, 'cats/')  # директория с картинками котов для обучения
    train_noncats_dir = os.path.join(train_dir, 'not_cats/')  # директория с картинками собак для обучения
    test_cats_dir = os.path.join(test_dir, 'cats/')  # директория с картинками котов для проверки
    test_noncats_dir = os.path.join(test_dir, 'not_cats/')
    num_cats_tr = len(os.listdir(train_dir))
    num_noncats_tr = len(os.listdir(train_noncats_dir))
    num_cats_test = len(os.listdir(test_cats_dir))
    num_noncats_test = len(os.listdir(test_noncats_dir))

    total_train = num_cats_tr + num_noncats_tr
    total_test = num_cats_test + num_noncats_test

    img_width, img_height = 150, 150
    input_shape = (3, img_height, img_width)
    batch_size = 32
    epochs = 15
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        subset='training',
                                                        shuffle=False)
    validation_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width),
                                                             batch_size=batch_size,
                                                             class_mode='binary',
                                                             subset='validation')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
                                                      batch_size=batch_size,
                                                      class_mode='binary',
                                                      shuffle=False)
    # sample_training_images, _ = next(train_generator)
    # plotImages(sample_training_images[:5])
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_generator,
                        steps_per_epoch=total_train // batch_size, epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples // batch_size)
    print("Evaluate on test data")
    results = model.evaluate(test_generator, batch_size=batch_size)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    # print("Generate predictions for 3 samples")
    # predictions = model.predict(x_test[:3])
    # print("predictions shape:", predictions.shape)