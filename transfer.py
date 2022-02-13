import numpy as np
from PIL import Image
import time
import functools
import tensorflow as tf
from keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import base64
import io
import matplotlib.pyplot as plt
import matplotlib as mpl

class Transfer:

    def __init__(self):
        self.content_layers = ['block5_conv2'] 
        self.style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1'
                    ]

        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
    def get_model(self):  
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(name).output for name in self.content_layers]
        model_outputs = style_outputs + content_outputs
        return models.Model(vgg.input, model_outputs)
    def load_img(self, base64_string):
        max_dim = 512
        img = Image.open(io.BytesIO(base64.b64decode(base64_string)))
        long = max(img.size)
        scale = max_dim/long
        img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)        
        img = kp_image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    def get_content_loss(self, base_content, target):
        return tf.reduce_mean(tf.square(base_content - target))


    def gram_matrix(self, input_tensor):
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)
 
    def get_style_loss(self, base_style, gram_target):
        """Expects two images of dimension h, w, c"""
        # height, width, num filters of each layer
        height, width, channels = base_style.get_shape().as_list()
        gram_style = self.gram_matrix(base_style)
        
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def get_feature_representations(self, model, base64_base_image, base64_style_path):
            content_image = self.load_and_process_img(base64_base_image)
            style_image = self.load_and_process_img(base64_style_path)

            # batch compute content and style features
            style_outputs = model(style_image)
            content_outputs = model(content_image)
            
            
            # Get the style and content feature representations from our model  
            style_features = [style_layer[0] for style_layer in style_outputs[:self.num_style_layers]]
            content_features = [content_layer[0] for content_layer in content_outputs[self.num_style_layers:]]
            return style_features, content_features

    def compute_loss(self,model, loss_weights, init_image, gram_style_features, content_features):
  
        style_weight, content_weight = loss_weights
        model_outputs = model(init_image)
        
        style_output_features = model_outputs[:self.num_style_layers]
        content_output_features = model_outputs[self.num_style_layers:]
        
        style_score = 0
        content_score = 0
        weight_per_style_layer = 1.0 / float(self.num_style_layers)
        for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * self.get_style_loss(comb_style[0], target_style)
            
        weight_per_content_layer = 1.0 / float(self.num_content_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer* self.get_content_loss(comb_content[0], target_content)
        
        style_score *= style_weight
        content_score *= content_weight
        loss = style_score + content_score 
        return loss, style_score, content_score


    def compute_grads(self, cfg):
        with tf.GradientTape() as tape: 
            all_loss = self.compute_loss(**cfg)
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss

    def deprocess_img(self, processed_img):
        x = processed_img.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                    "dimension [1, height, width, channel] or [height, width, channel]")
        if len(x.shape) != 3:
            raise ValueError("Invalid input to deprocessing image")
        
        # perform the inverse of the preprocessing step
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def load_and_process_img(self, base64_image):
        img = self.load_img(base64_image)
        # img.resize(2000, 2000)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img


    def run_style_transfer(self, base64_base_image, base64_style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2): 
  
        model = self.get_model() 
        for layer in model.layers:
            layer.trainable = False
        
        style_features, content_features = self.get_feature_representations(model, base64_base_image, base64_style_path)
        gram_style_features = [self.gram_matrix(style_feature) for style_feature in style_features]
        
        init_image = self.load_and_process_img(base64_base_image)
        init_image = tf.Variable(init_image, dtype=tf.float32)
        opt = tf.optimizers.Adam(learning_rate=5, epsilon=1e-1)
        
        best_loss, best_img = float('inf'), None
        
        loss_weights = (style_weight, content_weight)
        cfg = {
            'model': model,
            'loss_weights': loss_weights,
            'init_image': init_image,
            'gram_style_features': gram_style_features,
            'content_features': content_features
        }
        
        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means   
        
        for i in range(num_iterations):
            grads, all_loss = self.compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            
            if loss < best_loss:
            # Update best loss and best image from total loss. 
                best_loss = loss
                best_img = self.deprocess_img(init_image.numpy())

        # self.show_results(best_img,base64_base_image, base64_style_path)
        try:
            best_img = Image.fromarray(best_img, 'RGB')
            img_byte_arr = io.BytesIO()
            best_img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()            
            best_img = base64.b64encode(img_byte_arr).decode("utf-8")
        except BaseException as e:
            print(e)
        return best_img, best_loss

    def show_results(self, best_img, content_path, style_path, show_large_final=True):
        plt.figure(figsize=(10, 5))
        content = self.load_img(content_path) 
        style = self.load_img(style_path)

        plt.subplot(1, 2, 1)
        self.imshow(content, 'Content Image')

        plt.subplot(1, 2, 2)
        self.imshow(style, 'Style Image')

        if show_large_final: 
            plt.figure(figsize=(10, 10))

            plt.imshow(best_img)
            plt.title('Output Image')
            plt.show()

    def imshow(self, img, title=None):
        # Remove the batch dimension
        out = np.squeeze(img, axis=0)
        # Normalize for display 
        out = out.astype('uint8')
        plt.imshow(out)
        if title is not None:
            plt.title(title)
        plt.imshow(out)