import rasterio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2

#Collecting all images in a folder
folder = []
for files in os.listdir('../all'):
    if not files.endswith("checkpoints"):
        folder.append(files)
        # print(files)

mask_dir = "mask"
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

for file in folder:
    image_path =  f'../all/{file}/{file}.tif'
    possible_masks = ['BUILTUP.tif', 'BUILTUP1.tif', 'FINAL BUILTUP.tif','Builtup.tif']

    mask_path = None
    for mask_name in possible_masks:
        potential_path = f'../all/{file}/{mask_name}'
        if os.path.exists(potential_path):
            mask_path = potential_path
            break

    if mask_path:
        with rasterio.open(mask_path) as raster:
            data = raster.read(1)

            output_array = np.zeros(data.shape, dtype=np.uint8)

            data = raster.read(1)
            unique_values = list(np.unique(data))

            nb = int(raster.meta['nodata'])
            for val in unique_values:
                if val != nb:
                    b = val

            # Mark built-up areas as 0
            output_array[data == b] = 0  # built-up areas are marked with '0' in the TIFF file

            # Mark non-built-up areas as 1
            output_array[data == nb] = 1  # non-built-up areas are marked with '1' in the TIFF file

            # Mark no data areas as 255
            #output_array[data == src.nodata] = 255

        image = Image.fromarray(output_array)
        image.save(f'{mask_dir}/{file}.png')

        print(f"Mask PNG saved")


from PIL import Image

for files in folder:
    tiff_path = f'../all/{files}/{files}.tif'
    png_path = f'images/{files}.png'

    with Image.open(tiff_path) as img:
        img.save(png_path, format='PNG')

    print(f"Image saved as {png_path}")

import keras
from keras import layers
from keras import ops
import shutil
import os
import numpy as np
from glob import glob
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf
# For data preprocessing
from tensorflow import image as tf_image
from tensorflow import data as tf_data
from tensorflow import io as tf_io
from PIL import Image

# Dividing into chunks
image_chunks_dir = 'all_img_chunks_240'
mask_chunks_dir= 'all_mask_chunks_240'
os.makedirs(mask_chunks_dir,exist_ok=True)
os.makedirs(image_chunks_dir,exist_ok=True)

def cut_image_into_chunks(image_path, chunk_size=(240, 240), overlap=20, output_dir=image_chunks_dir):
  image = Image.open(image_path)
  width, height = image.size
  chunk_width, chunk_height = chunk_size

  # create_directory(output_dir)
  chunk_x = 0
  chunk_y = 0

  x = 0
  y = 0

  num_chunks_x = 0
  num_chunks_y = 0

  second_last_x = x
  second_last_y = y
  while x < width:
      second_last_x = x
      if x + chunk_width < width:
          while y < height:
              second_last_y = y
              if y + chunk_height < height:
                  chunk = image.crop((x, y, x + chunk_width, y + chunk_height))
                  chunk_name = f"{output_dir}/{image_path.split('/')[-1].split('.')[0]}_row{chunk_y}_col{chunk_x}.png"
                  chunk.save(chunk_name)
                  y += chunk_height - overlap
                  chunk_y += 1
              else:
                  y = height - chunk_height
                  chunk = image.crop((x, y, x + chunk_width, y + chunk_height))
                  chunk_name = f"{output_dir}/{image_path.split('/')[-1].split('.')[0]}_row{chunk_y}_col{chunk_x}.png"
                  chunk.save(chunk_name)
                  y = height
          x += chunk_width - overlap
          chunk_x += 1
          y = 0
          chunk_y = 0
      else:
          num_chunks_x = chunk_x
          x = width - chunk_width
          while y < height:
              if y + chunk_height < height:
                  chunk = image.crop((x, y, x + chunk_width, y + chunk_height))
                  chunk_name = f"{output_dir}/{image_path.split('/')[-1].split('.')[0]}_row{chunk_y}_col{chunk_x}.png"
                  chunk.save(chunk_name)
                  y += chunk_height - overlap
                  chunk_y += 1
              else:
                  num_chunks_y = chunk_y
                  y = height - chunk_height
                  chunk = image.crop((x, y, x + chunk_width, y + chunk_height))
                  chunk_name = f"{output_dir}/{image_path.split('/')[-1].split('.')[0]}_row{chunk_y}_col{chunk_x}.png"
                  chunk.save(chunk_name)
                  y = height
          x = width
#   return num_chunks_x + 1, num_chunks_y + 1, second_last_x, second_last_y

def process_images_in_folder(folder_path, output_dir):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png') or f.endswith('.tif')]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        cut_image_into_chunks(image_path, output_dir=output_dir)

home_dir = '/home/ec2-user/SageMaker/notebooks/harsh/builtup_pixelwise/check/model1'

#image chunks dir
folder_path = f'{home_dir}/all_image'
# output_dir = 21_au  #'img_chunks_240'
process_images_in_folder(folder_path,image_chunks_dir)

# mask_chunks_dir

folder_path = f'{home_dir}/all_mask'
output_dir = mask_chunks_dir #'mask_chunks_240'
process_images_in_folder(folder_path,output_dir)

images_list = os.listdir('val_img_chunks_240')

import random
# Shuffle the list randomly
random.shuffle(images_list)
val_images = images_list[:60]


act_all_images_chunks = []
for paths in os.listdir('all_img_chunks_240'):
    path = paths[-27:]  # Extract the last 27 characters from the current path
    found_in_val = False
    for i in val_images:
        if path == i[-27:]:  # Compare the extracted part with the corresponding part of val_images
            found_in_val = True
            break
    if not found_in_val:
        act_all_images_chunks.append(paths)


random.shuffle(act_all_images_chunks)
test_images = act_all_images_chunks[:40]



# Destination folder
destination_folder = 'test_image_chunks_240'
os.makedirs(destination_folder, exist_ok=True)

# Copy each image to the destination folder
for image_path in test_images:
    if os.path.exists(f'all_image_chunks_240/{image_path}'):
        shutil.copy(f'all_image_chunks_240/{image_path}', destination_folder)
    else:
        print(f"Image not found: {image_path}")




# Destination folder
destination_folder = 'test_mask_chunks_240'
os.makedirs(destination_folder, exist_ok=True)

# Copy each image to the destination folder
for image_path in test_images:
    if os.path.exists(f'all_mask_chunks_240/{image_path}'):
        shutil.copy(f'all_mask_chunks_240/{image_path}', destination_folder)
    else:
        print(f"Image not found: {image_path}")


import shutil
import os

# Destination folder
destination_folder = 'train_mask_chunks_240'
os.makedirs(destination_folder, exist_ok=True)

# Copy each image to the destination folder
for image_path in act_all_images_chunks[40:]:
    if os.path.exists(f'all_mask_chunks_240/{image_path}'):
        shutil.copy(f'all_mask_chunks_240/{image_path}', destination_folder)
    else:
        print(f"Image not found: {image_path}")


train_images_dir = "train_img_chunks_240"
train_masks_dir = "train_mask_chunks_240"
test_mask_dir = "test_mask_chunks_240"
test_img_dir = "test_img_chunks_240"
val_images_dir = "val_img_chunks_240"
val_masks_dir = "val_mask_chunks_240"

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return ops.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus_MobileNetV2(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    preprocessed = keras.applications.mobilenet_v2.preprocess_input(model_input)
    mobilenet_v2 = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )

    # Get feature maps from MobileNetV2
    x = mobilenet_v2.get_layer("block_13_expand_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)

    input_b = mobilenet_v2.get_layer("block_3_expand_relu").output
    input_b = convolution_block(input_b, num_filters=256, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


IMAGE_SIZE = 512
BATCH_SIZE = 16
NUM_CLASSES = 2
DATA_DIR = '/home/ec2-user/SageMaker/notebooks/siddhant' #"/home/ec2-user/SageMaker/notebooks/harsh/b_newiter1"
# NUM_TRAIN_IMAGES = 9
# NUM_VAL_IMAGES = 3

train_images_dir = os.path.join(DATA_DIR, train_images_dir) #
train_masks_dir = os.path.join(DATA_DIR,train_masks_dir)
val_images_dir = os.path.join(DATA_DIR,val_images_dir)
val_masks_dir = os.path.join(DATA_DIR,val_masks_dir)

train_images_li = [f for f in os.listdir(train_images_dir) if f.endswith('.png') or f.endswith('.jpg')]
train_masks_li = [f for f in os.listdir(train_masks_dir) if f.endswith('.png') or f.endswith('.jpg')]
val_images_li = [f for f in os.listdir(val_images_dir) if f.endswith('.png') or f.endswith('.jpg')]
val_masks_li = [f for f in os.listdir(val_masks_dir) if f.endswith('.png') or f.endswith('.jpg')]

# print(len(train_images_li))
# print(len(train_masks_li))
# print(len(val_images_li))
# print(len(val_masks_li))

train_images = [os.path.join(DATA_DIR,train_images_dir ,mg) for mg in train_images_li]
train_masks = [os.path.join(DATA_DIR,train_masks_dir,mg)for mg in train_masks_li]
val_images = [os.path.join(DATA_DIR,val_images_dir ,mg) for mg in val_images_li]
val_masks = [os.path.join(DATA_DIR, val_masks_dir,mg) for mg in val_masks_li]



def read_image(image_path, mask=False, IMAGE_SIZE=512):
    image = tf_io.read_file(image_path)
    if mask:
        image = tf_image.decode_png(image, channels=1)
        #image = tf.where(image > 0, tf.ones_like(image), tf.zeros_like(image))
        image = tf.where(image == 255, tf.zeros_like(image), tf.ones_like(image))
        image.set_shape([None, None, 1])
        image = tf_image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    else:
        image = tf_image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf_image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image

# def read_test_image(image_path, mask=False):
#     image = tf_io.read_file(image_path)
#     if mask:
# #         image[image==255]=1
#         image = tf_image.decode_png(image, channels=1)
# #         image = tf.where(image==255, tf.ones_like(image),image)
#         image.set_shape([None, None, 1])
#         image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     else:
#         image = tf_image.decode_png(image, channels=3)
#         image.set_shape([None, None, 3])
#         image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     return image

def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    #this is not doing anything as all values of 255 already converted to 1
    mask = tf.cast(mask, tf.float32)
    ignore_mask = tf.cast(tf.not_equal(mask,255),tf.float32)
    mask = mask * ignore_mask

    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf_data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

model = DeeplabV3Plus_MobileNetV2(image_size=512, num_classes=NUM_CLASSES)
#model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint
os.makedirs('checkpoints2',exist_ok = True)
checkpoint_callback = ModelCheckpoint(
    filepath='checkpoints2/checkpoint-{epoch:02d}.weights.h5',
    monitor='val_loss',  # monitor validation loss
    save_best_only=False,  # save all epochs, not just the best one
    save_weights_only=True,  # save entire model, not just weights
    mode='auto',  # auto mode for determining best model
    save_freq='epoch'  # save every epoch
)

def calculate_class_frequencies(mask_paths):
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        class_counts[0] += np.sum(mask == 0)
        class_counts[1] += np.sum(mask == 1)
        class_counts[1] += np.sum(mask == 255)
    total_pixels = np.sum(class_counts)
    class_frequencies = class_counts / total_pixels
    return class_frequencies

class_frequencies = calculate_class_frequencies(train_masks)
print("Class Frequencies:", class_frequencies)
# class_weights = np.array([1.0,0.5],dtype = np.float32)

class_weights = 1/class_frequencies
class_weights = class_weights / np.max(class_weights)
print(class_weights)

def masked_sparse_categorical_crossentropy_weights(y_true, y_pred,class_weights):
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])

    ignore_mask = tf.cast(tf.not_equal(y_true_flat, 255), tf.float32)

    y_true_flat = y_true_flat * ignore_mask
    y_pred_flat = y_pred_flat * tf.expand_dims(ignore_mask, axis=-1)

    weights = tf.gather(class_weights , tf.cast(y_true_flat , tf.int32))
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_flat, y_pred_flat, from_logits=True)
    loss = loss*weights

    return tf.reduce_mean(loss)

class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

## 214 epochs 8

# loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss = focal_loss(gamma=2.,alpha = .25)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.000001),
    loss=lambda y_true,y_pred :masked_sparse_categorical_crossentropy_weights(y_true,y_pred,class_weights_tensor),
    metrics=["accuracy"],
)

early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    min_delta = 0.001,
    verbose=1
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks = [early_stopping_callback, checkpoint_callback])

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["accuracy"])
plt.title("Training Accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_loss"])
plt.title("Validation Loss")
plt.ylabel("val_loss")
plt.xlabel("epoch")
plt.show()

plt.plot(history.history["val_accuracy"])
plt.title("Validation Accuracy")
plt.ylabel("val_accuracy")
plt.xlabel("epoch")
plt.show()

model.load_weights('checkpoints2/checkpoint-05.weights.h5')

def infer(model, image_tensor, original_size=(240, 240)):
    # Ensure the input image_tensor has the correct shape and type
    image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension if necessary

    # Make predictions
    predictions = model.predict(image_tensor)

    # Post-process predictions
    predictions = np.squeeze(predictions)  # Remove batch dimension
    predictions = np.argmax(predictions, axis=2)  # Convert softmax output to class labels

    # Resize predictions back to original size (240x240)
    resized_predictions = tf.image.resize(np.expand_dims(predictions, axis=-1), original_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    resized_predictions = np.squeeze(resized_predictions, axis=-1)

    return resized_predictions

def plot_samples_matplotlib(display_list, figsize=(5, 3)):
    _, axes = plt.subplots(nrows=1, ncols=len(display_list), figsize=figsize)
    for i in range(len(display_list)):
        if display_list[i].shape[-1] == 3:
            axes[i].imshow(keras.utils.array_to_img(display_list[i]))
        else:
            axes[i].imshow(display_list[i])
    plt.show()

Predicition_dir = 'preds_model2/preds_ckpt05'

def plot_predictions(images_list, model):
    save_dir = Predicition_dir
#     os.makedirs(save_dir, exist_ok=True)
    for image_file in images_list:
        image_tensor = read_image(image_file)

#         print(image_tensor)
        # Assuming you have a function to read image
        prediction_mask = infer(model=model, image_tensor=image_tensor)
#         print(prediction_mask)
        filename = os.path.splitext(os.path.basename(image_file))[0]
        predicted_mask_filename = os.path.join(save_dir, f'{filename}_predicted_mask.png')
        cv2.imwrite(predicted_mask_filename,prediction_mask)
        plot_samples_matplotlib([image_tensor, prediction_mask])



IMAGE_SIZE = 512
NUM_CLASSES = 2

def read_image(image_path, mask=False):
    image = tf_io.read_file(image_path)
    if mask:
#         image[image==255]=1
        image = tf_image.decode_png(image, channels=1)
        image = tf.where(image==255, tf.ones_like(image),image)
        image.set_shape([None, None, 1])
        image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    else:
        image = tf_image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf_image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


CHUNK_DATA_DIR = ""
os.makedirs(Predicition_dir,exist_ok=True)

test_images_dir = os.path.join(CHUNK_DATA_DIR,"test_img_chunks_240")
# test_masks_dir = os.path.join(CHUNK_DATA_DIR,"mask_chunks","test")

test_images_li = [f for f in os.listdir(test_images_dir) if f.endswith('.png') or f.endswith('.jpg')]
# test_masks_li = [f for f in os.listdir(test_masks_dir) if f.endswith('.png') or f.endswith('.jpg')]

test_images = [os.path.join(test_images_dir ,mg) for mg in test_images_li]
# test_masks = [os.path.join(test_masks_dir,mg)for mg in test_masks_li]
print(len(test_images))
# pred_imgs = test_images[0:5]
# print(pred_imgs)
#test_images[0]

image = [path.split('/')[-1] for path in test_images]

plot_predictions(test_images,model=model)

# test_imag = [val.split('.png')[0] for val in os.listdir('img_chunks_512/test')]  # os.listdir('img_chunks_512/test') #

test_imag = image

#test_imag = ['GRUS1D_20240618090211_L1C_PSM_N48060209.png', 'GRUS1D_20240618090211_L1C_PSM_N48070209.png']

DATA_MASK_DIR = '/home/ec2-user/SageMaker/notebooks/siddhant/'

test_paths = [os.path.join(DATA_MASK_DIR,mg)for mg in test_images]

test_imags = [path.split('.png')[0] for path in test_imag]

DATA_MASK_DIR = '/home/ec2-user/SageMaker/notebooks/siddhant/test_mask_chunks_240'

from sklearn.metrics import confusion_matrix

# mask2 = np.array(Image.open(f"preds/preds_ckpt20/{img}_predicted_mask.png"))
# mask2.shape

target_class = 0
data_dict = {}
from sklearn.metrics import confusion_matrix

fin_imag = image

for img in tqdm(fin_imag):
    mask1 = np.array(Image.open(f"{DATA_MASK_DIR}/{img}")) #{DATA_MASK_DIR}
    mask2 = np.array(Image.open(f"{Predicition_dir}/{img.split('.png')[0]}_predicted_mask.png"))
    binary_mask1 = (mask1 == target_class).astype(np.uint8) #target_class
    binary_mask2 = (mask2 == target_class).astype(np.uint8) #target_class
#     print(mask1)
#     print(mask2)
    conf_mat = confusion_matrix(binary_mask1.flatten(), binary_mask2.flatten())
    try:
        TN = conf_mat[0][0]
        FP = conf_mat[0][1]
        FN = conf_mat[1][0]
        TP = conf_mat[1][1]
        data_dict[img] = (TN,FP,FN,TP)
    except:
        pass

import pandas as pd
df = pd.DataFrame(data = data_dict, index = ['TN', 'FP', 'FN', 'TP']).T.reset_index().rename(columns = {'index': 'Image'})

df['Precision'] = round(df['TP'] * 100 / (df['TP'] + df['FP']), 2)
df['Recall'] = round(df['TP'] * 100 / (df['TP'] + df['FN']),2)

# Sum the values in the TN, FP, FN, and TP columns
TN_sum = df['TN'].sum()
FP_sum = df['FP'].sum()
FN_sum = df['FN'].sum()
TP_sum = df['TP'].sum()


# Calculate Precision and Recall
precision = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) != 0 else 0
recall = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) != 0 else 0

print(f"Total TN: {TN_sum}")
print(f"Total FP: {FP_sum}")
print(f"Total FN: {FN_sum}")
print(f"Total TP: {TP_sum}")
print(f"Overall Precision: {precision:.2f}")
print(f"Overall Recall: {recall:.2f}")
