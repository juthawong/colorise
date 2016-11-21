import os
import subprocess
import tensorflow as tf
from scipy.ndimage import imread
from scipy.misc import imresize
from skimage.color import rgb2gray

train_path = './images/'
image_px = 224
n_iters = 10
batch_size = 10

execfile('vgg16.py')
execfile('loss.py')

sess = tf.Session()

# imgs is grayscale represented in 3 dimensions but HOW?
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
imgs_uv = tf.placeholder(tf.float32, [None, 224, 224, 2])

vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

W_conv0 = weight_variable([1, 1, 512, 256])
b_conv0 = bias_variable([256])

rhs0 = tf.nn.relu(conv2d(vgg.normconv4_3, W_conv0) + b_conv0)
rhs0_ = tf.image.resize_bilinear(rhs0, [56, 56])

# layer 1
W_conv1 = weight_variable([3, 3, 512, 128])
b_conv1 = bias_variable([128])

# batch, height, width, pixel -> axis = 3
r = tf.concat(3, [rhs0_, vgg.normconv3_3])
rhs1 = tf.nn.relu(conv2d(r, W_conv1) + b_conv1)
rhs1_ = tf.image.resize_bilinear(rhs1, [112, 112])

# layer 2
W_conv2 = weight_variable([3, 3, 256, 64])
b_conv2 = bias_variable([64])

r = tf.concat(3, [rhs1_, vgg.normconv2_2])
rhs2 = tf.nn.relu(conv2d(r, W_conv2) + b_conv2)
rhs2_ = tf.image.resize_bilinear(rhs2, [224, 224])

# layer 3
W_conv3 = weight_variable([3, 3, 128, 3])
b_conv3 = bias_variable([3])

r = tf.concat(3, [rhs2_, vgg.normconv1_2])
rhs3 = tf.nn.relu(conv2d(r, W_conv3) + b_conv3)
# ^this one is 224*224*3

# layer 4
W_conv4 = weight_variable([3, 3, 6, 3])
b_conv4 = bias_variable([3])

r = tf.concat(3, [rhs3, imgs])
rhs4 = tf.nn.relu(conv2d(r, W_conv4) + b_conv4)

# finally convolving to get UV values
W_conv5 = weight_variable([3, 3, 3, 2])
b_conv5 = bias_variable([2])

output = tf.nn.relu(conv2d(rhs4, W_conv5) + b_conv5)

loss_to_min = loss(output, imgs_uv)

var = [vgg.W_conv0, vgg.b_conv0, vgg.W_conv1, vgg.b_conv1, vgg.W_conv2, vgg.b_conv2, vgg.W_conv3, vgg.b_conv3, vgg.W_conv4, vgg.b_conv4, vgg.W_conv5, vgg.b_conv5]
opt = AdamOptimizer(1e-3)
train_step = opt.minimize(loss, var)

init = tf.initilize_all_variables()
sess.run(init)


image_list = subprocess.check_output("ls -1 "+train_path, shell=True).split('\n')[:-1]
rgb_data = np.array([imresize(imread(train_path+i, mode='RGB'), (image_px, image_px)) for i in image_list], dtype='float32')
rgb_data = rgb_data/255.0
# np.save('rgb_data', rgb_data, allow_pickle=False)
# np.load('rgb_data')

y_data = np.array([rgb2gray(i) for i in rgb_data])
uv_data = np.array([get_UV(i) for i in rgb_data])
# np.save('y_data', y_data, allow_pickle=False)
# np.save('uv_data', uv_data, allow_pickle=False)
# np.load('y_data')
# np.load('uv_data')


train_accuracy = sess.run(loss, feed_dict={x: y_data, y_: uv_data, keep_prob: 0.5})
print(train_accuracy)

sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

train_accuracy = sess.run(loss, feed_dict={x: y_data, y_: uv_data, keep_prob: 0.5})
print(train_accuracy)


# n_images = len(y_data)
# for i in range(n_iters):
# 	batch_xs = y_data[batch_size*(i%) : batch_size*(i%(n_labels+1)+1)]
# 	batch_ys = uv_data[batch_size*(i%(n_labels+1)) : batch_size*(i%(n_labels+1)+1)]
# 	if i%(n_labels+1) == 0:
# 		shuffle_in_unison(train, train_labels)
# 	if i%100 == 0:
# 		train_accuracy = sess.run(accuracy, feed_dict={x: train[:n_train/100], y_: train_labels[:n_train/100], keep_prob: 0.5})
# 		valid_accuracy = sess.run(accuracy, feed_dict={x: valid, y_: valid_labels, keep_prob: 0.5})
# 		print("step %d, training accuracy %g validation accuracy %g"%(i, train_accuracy, valid_accuracy))	
# 	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
