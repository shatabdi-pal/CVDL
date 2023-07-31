
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.applications import InceptionResNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Model
from keras.layers import Input



#Load the InceptionResNetv2 model
pre_model = InceptionResNetV2(weights ="imagenet", include_top=False, input_shape=(150, 150,3))

#print the model summary
#pre_model.summary()

# Access the first convolutional layer (index 1)
first_conv_layer = pre_model.layers[1]
first_layer_weights = first_conv_layer.get_weights()[0]

# Reshape and plot the filters
n_filters = first_layer_weights.shape[3]
n_channels = first_layer_weights.shape[2]
filter_size = first_layer_weights.shape[0]

plt.figure(figsize=(8, 8))
for i in range(n_filters):
    ax = plt.subplot(8, 8, i + 1)
    filter_img = first_layer_weights[:, :, :, i]
    filter_img = filter_img.reshape(filter_size, filter_size, n_channels)
    # Normalize the filter values
    filter_img = (filter_img - np.min(filter_img)) / (np.max(filter_img) - np.min(filter_img))
    plt.imshow(filter_img[:, :, 0], cmap='gray')
    plt.axis('off')
plt.show()


# Function to load and pre-process images from a specific category directory
def load_images_from_category(data_dir, category, image_size=(150, 150)):
    category_dir = os.path.join(data_dir, category)
    data = []
    labels = []
    for filename in os.listdir(category_dir):
        img_path = os.path.join(category_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img)
        labels.append(category)
    return data, labels

# training and testing data
train_dir = 'cats_dogs_dataset/dog vs cat/dataset/training_set/'
test_dir = 'cats_dogs_dataset/dog vs cat/dataset/test_set/'


# Load and preprocess training data for cats and dogs
train_cat_data, train_cat_labels = load_images_from_category(train_dir, 'cats')
train_dog_data, train_dog_labels = load_images_from_category(train_dir, 'dogs')

# Load and preprocess testing data for cats and dogs
test_cat_data, test_cat_labels = load_images_from_category(test_dir, 'cats')
test_dog_data, test_dog_labels = load_images_from_category(test_dir, 'dogs')

# Combine the cat and dog data and labels
train_data = np.array(train_cat_data + train_dog_data)
train_labels = np.array(train_cat_labels + train_dog_labels)
test_data = np.array(test_cat_data + test_dog_data)
test_labels = np.array(test_cat_labels + test_dog_labels)

# Shuffle the training data and labels
indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)
train_data = train_data[indices]
train_labels = train_labels[indices]

# Normalize pixel values to [0, 1]
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the string labels to integer labels
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.fit_transform(test_labels)

#Convert labels to categorical format (one-hot encoding)
num_classes = 2  # cat and dog
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)


# Load the pre-trained InceptionResNetV2 model with weights trained on the ImageNet dataset
pre_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the weights of the pre-trained model
pre_model.trainable = False

# Create the new model (transfer head)
model = Sequential()

# Add the pre-trained InceptionResNetV2 as the base of the model
model.add(pre_model)

# Add a Global Average Pooling layer to reduce spatial dimensions
model.add(GlobalAveragePooling2D())

# Add a new classification head (dense layer) for your specific task
# For example, let's add a dense layer with 256 units and a softmax layer for binary classification (cats vs. dogs)
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Print the summary of the new model
model.summary()




# Step 1: Evaluate the transfer model (without training the unfrozen weights) on the test dataset

# Compile the model with appropriate loss and metrics
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_data, test_labels)

# Print the overall test accuracy
print(f'Overall Test Accuracy: {accuracy}')

# Create predictions for the test data
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Create and print the confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix:')
print(confusion_mat)

# Step 2: Train the transfer model using binary cross-entropy loss on the cats/dogs training data

# Compile the model with binary cross-entropy loss and appropriate optimizer (e.g., Adam)
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model on the training data
history = model.fit(train_data, train_labels, batch_size=32, epochs=10, validation_split=0.1)

# Step 3: Report the per-epoch test loss for the trained transfer model and the confusion matrix on the test data

# Evaluate the trained model on the test dataset
loss, accuracy = model.evaluate(test_data, test_labels)

# Print the per-epoch test loss
print('Per-epoch Test Loss:')
print(loss)

# Create predictions for the test data using the trained model
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# Create and plot the confusion matrix for the final model
confusion_mat = confusion_matrix(true_labels, predicted_labels)
print('Confusion Matrix (Final Model):')
print(confusion_mat)

# Plot the training and validation loss for each epoch
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Assuming you have the pre-trained InceptionResNetV2 model "pre_model" from Step 1

# Choose the number of layers to retain in the sub-network (k layers)
k = 100

# Create a new input layer for the sub-network with the same input shape as the pre-trained model
sub_input = Input(shape=(150, 150, 3))

# Get the first k layers of the pre-trained model
sub_model = Model(inputs=pre_model.input, outputs=pre_model.layers[k].output)

# Connect the new input layer to the output of the kth layer of the pre-trained model
sub_output = sub_model(sub_input)

# Add a Global Average Pooling layer and a new classification head to the sub-network
x = GlobalAveragePooling2D()(sub_output)
x = Dense(256, activation='relu')(x)
sub_model_output = Dense(2, activation='softmax')(x)

# Create the final sub-network model with the input and output layers
sub_model = Model(inputs=sub_input, outputs=sub_model_output)

# Freeze the weights of the sub-network
sub_model.trainable = False

# Compile the sub-network model with binary cross-entropy loss and appropriate optimizer (e.g., Adam)
sub_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the summary of the sub-network model
sub_model.summary()




