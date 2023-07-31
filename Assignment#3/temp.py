import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2

# Load the pre-trained InceptionResNetV2 model
pre_model = InceptionResNetV2(weights='imagenet', include_top=False)

# Display model summary
pre_model.summary()

# Visualize the first layer filters (optional)
# You can use the code appropriate for your visualization library (e.g., matplotlib, seaborn).
# For example, to visualize the first layer filters using matplotlib:
import matplotlib.pyplot as plt

# Get the first layer filters
first_layer_filters = pre_model.layers[1].get_weights()[0]

# Plot the filters
plt.figure(figsize=(10, 10))
for i in range(32):
    plt.subplot(4, 8, i + 1)
    plt.imshow(first_layer_filters[:, :, :, i], cmap='viridis')
    plt.axis('off')
plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data directories
train_dir = 'cats_dogs_dataset/dog vs cat/dataset/training_set/'
test_dir = 'cats_dogs_dataset/dog vs cat/dataset/test_set/'


# Preprocess images
image_size = (150, 150)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Create a new sequential model
model = Sequential()

# Add the pre-trained InceptionResNetV2 model (excluding the final classification layer)
model.add(pre_model)

# Add a new classification head
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Display model summary
model.summary()


from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 4(i): Evaluate the transfer model on the test dataset
# Assuming the model is already compiled (if not, compile it with appropriate settings)
model.compile(optimizer, loss)
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")
# Compute the confusion matrix for the test data
test_predictions = model.predict(test_generator)
test_predictions = (test_predictions > 0.5).astype(int)
test_labels = test_generator.classes
confusion_matrix_result = confusion_matrix(test_labels, test_predictions)
print("Confusion matrix:")
print(confusion_matrix_result)

# Step 4(ii): Train the transfer model using binary cross-entropy loss
# Compile the model with appropriate optimizer and loss function
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
num_epochs = 10
model.fit(train_generator, epochs=num_epochs)

# Evaluate the final model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Final Test accuracy: {test_accuracy}")
# Compute the confusion matrix for the test data
test_predictions = model.predict(test_generator)
test_predictions = (test_predictions > 0.5).astype(int)
test_labels = test_generator.classes
confusion_matrix_result = confusion_matrix(test_labels, test_predictions)
print("Final Confusion matrix:")
print(confusion_matrix_result)

# Step 4(iii): Create a sub-network of the pre-trained model
# For example, you can retain only the first 100 layers of the pre-trained model
sub_network_layers = pre_model.layers[:100]

# Create a new model using the sub-network and the new classification head
sub_model = Sequential(sub_network_layers)
sub_model.add(GlobalAveragePooling2D())
sub_model.add(Dense(256, activation='relu'))
sub_model.add(Dense(1, activation='sigmoid'))

# Freeze the weights of the sub-network
for layer in sub_model.layers:
    layer.trainable = False

# Compile the sub-model
sub_model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Train the sub-model
sub_model.fit(train_generator, epochs=num_epochs)

# Evaluate the final sub-model on the test dataset
test_loss, test_accuracy = sub_model.evaluate(test_generator)
print(f"Sub-model Test accuracy: {test_accuracy}")
# Compute the confusion matrix for the test data
test_predictions = sub_model.predict(test_generator)
test_predictions = (test_predictions > 0.5).astype(int)
test_labels = test_generator.classes
confusion_matrix_result = confusion_matrix(test_labels, test_predictions)
print("Sub-model Confusion matrix:")
print(confusion_matrix_result)

