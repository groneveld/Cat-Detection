import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator


if __name__ == '__main__':
    train_dir = '../input/bestcatnotcatdataset/data/train'
    test_dir = '../input/user-cats/test'
    train_cats_dir = os.path.join(train_dir, 'cats/')
    train_noncats_dir = os.path.join(train_dir, 'a_not_cats/')
    test_cats_dir = os.path.join(test_dir, 'cats/')
    test_noncats_dir = os.path.join(test_dir, 'a_not_cats/')
    num_cats_tr = len(os.listdir(train_dir))
    num_noncats_tr = len(os.listdir(train_noncats_dir))
    num_cats_test = len(os.listdir(test_cats_dir))
    num_noncats_test = len(os.listdir(test_noncats_dir))

    total_train = num_cats_tr + num_noncats_tr
    total_test = num_cats_test + num_noncats_test

    img_width, img_height = 200, 200
    input_shape = (img_height, img_width, 3)
    batch_size = 32
    epochs = 15
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        shuffle=True,
                                                        subset='training')
    validation_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width),
                                                             batch_size=batch_size,
                                                             shuffle=True,
                                                             class_mode='binary',
                                                             subset='validation')
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height),
                                                      batch_size=1,
                                                      shuffle=True,
                                                      class_mode='binary')

    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=SGD(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_generator,
                        steps_per_epoch=total_train // batch_size, epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples // batch_size)
    print("Evaluate on test data")
    results = model.evaluate(test_generator, batch_size=1)
    print("test loss, test acc:", results)
    model.save('model')