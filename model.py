import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, \
    Input, MaxPool2D,LeakyReLU,Dropout, Concatenate
from tensorflow.keras import Sequential

class UnetModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(.001)
        self.batch_size = 128
        self.leaky_relu_rate =.1
        self.leaky = LeakyReLU(self.leaky_relu_rate)
   
        self.model = Sequential()
        #s = Lambda(lambda x: x / 255)(inputs)
        
        self.block1conv1 = Conv2D(16, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block1norm1 = BatchNormalization()
        self.block1conv2 = Conv2D(16, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block1norm2 = BatchNormalization()
        self.block1pool = MaxPool2D((2, 2))

        self.block2conv1 = Conv2D(32, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block2norm1 = BatchNormalization()
        self.block2conv2 = Conv2D(32, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block2norm2 = BatchNormalization()
        self.block2pool = MaxPool2D((2, 2))

        self.block3conv1 = Conv2D(64, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block3norm1 = BatchNormalization()
        self.block3conv2 = Conv2D(64, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block3norm2 = BatchNormalization()
        self.block3pool = MaxPool2D((2, 2))

        self.block4conv1 = Conv2D(128, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block4norm1 = BatchNormalization()
        self.block4conv2 = Conv2D(128, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block4norm2 = BatchNormalization()
        self.block4pool = MaxPool2D((2, 2))



        self.block5conv1 = Conv2D(256, (3, 3), activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block5dropout = Dropout(0.3)
        self.block5conv2 = Conv2D(256, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
 

        self.block6trans = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        # u6 = concatenate([u6, c4])
        self.block6conv1 = Conv2D(128, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block6norm1 = BatchNormalization()
        self.block6conv2 = Conv2D(128, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block6norm2 = BatchNormalization()

        self.block7trans = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        # u6 = concatenate([u7, c3])
        self.block7conv1 = Conv2D(64, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block7norm1 = BatchNormalization()
        self.block7conv2 = Conv2D(64, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block7norm2 = BatchNormalization()


        self.block8trans = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')
        # u6 = concatenate([u8, c2])
        self.block8conv1 = Conv2D(32, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block8norm1 = BatchNormalization()
        self.block8conv2 = Conv2D(32, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block8norm2 = BatchNormalization()

        self.block9trans = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')
        # u6 = concatenate([u9, c1])
        self.block9conv1 = Conv2D(16, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block9norm1 = BatchNormalization()
        self.block9conv2 = Conv2D(16, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block9norm2 = BatchNormalization()


        self.outputs = Conv2D(1, (1, 1), activation='sigmoid')
    def call(self, inputs):
        # implement downsample layers
        block1output = self.block1pool(self.block1norm2(self.block1conv2(self.block1norm1(self.block1conv1(inputs)))))
        block2output = self.block2pool(self.block2norm2(self.block2conv2(self.block2norm1(self.block2conv1(block1output)))))
        block3output = self.block3pool(self.block3norm2(self.block3conv2(self.block3norm1(self.block3conv1(block2output)))))
        block4output = self.block4pool(self.block4norm2(self.block4conv2(self.block4norm1(self.block4conv1(block3output)))))
        # bottom of the u
        block5output = self.block5conv2(self.block5dropout(self.block5conv1(block4output)))
        # go back up
        block6trans= self.block6trans(block5output)
        block6cat = Concatenate([block6trans,block4output])
        block6output = self.block6norm2(self.block6conv2(self.block6norm1(self.block6conv1(block6cat))))

        block7trans= self.block6trans(block6output)
        block7cat = Concatenate([block7trans,block3output])
        block7output = self.block7norm2(self.block7conv2(self.block7norm1(self.block7conv1(block7cat))))

        block8trans= self.block6trans(block7output)
        block8cat = Concatenate([block8trans,block2output])
        block8output = self.block8norm2(self.block8conv2(self.block8norm1(self.block8conv1(block8cat))))

        block9trans= self.block9trans(block8output)
        block9cat = Concatenate([block9trans,block1output])
        block9output = self.block9norm2(self.block9conv2(self.block9norm1(self.block9conv1(block9cat))))

        output = self.outputs(block9output)

        return output
    def loss(self, preds, labels):
        loss_numerator = -2 * tf.math.reduce_sum(tf.tensordot(preds,labels))
        loss_denom = tf.reduce_sum(preds)+tf.reduce_sum(labels)+1
        return loss_numerator/loss_denom
    def accuracy(self, preds,labels):
        intersection = np.count_nonzero(preds==labels) 
        union = np.count_nonzero(preds)+np.count_nonzero(labels)
        return intersection/union