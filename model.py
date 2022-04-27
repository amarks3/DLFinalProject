import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, \
    InputLayer, MaxPool2D,LeakyReLU,Dropout, Concatenate
from tensorflow.keras import Sequential

class UnetModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(.001)
        self.batch_size = 29
        self.leaky_relu_rate =.1
        self.leaky = LeakyReLU(self.leaky_relu_rate)
        self.img_height =512
   
        #s = Lambda(lambda x: x / 255)(inputs)
        self.inputs =InputLayer(input_shape=(512,512,1),batch_size=self.batch_size)
        self.block1conv1 = Conv2D(16, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block1norm1 = BatchNormalization()
        self.block1conv2 = Conv2D(16, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block1norm2 = BatchNormalization()
        self.block1pool = MaxPool2D((2, 2))
        #256

        self.block2conv1 = Conv2D(32, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block2norm1 = BatchNormalization()
        self.block2conv2 = Conv2D(32, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block2norm2 = BatchNormalization()
        self.block2pool = MaxPool2D((2, 2))
        #128

        self.block3conv1 = Conv2D(64, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block3norm1 = BatchNormalization()
        self.block3conv2 = Conv2D(64, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block3norm2 = BatchNormalization()
        self.block3pool = MaxPool2D((2, 2))
        #64

        self.block4conv1 = Conv2D(128, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block4norm1 = BatchNormalization()
        self.block4conv2 = Conv2D(128, (3, 3), activation = self.leaky, kernel_initializer='random_normal', padding='same')
        self.block4norm2 = BatchNormalization()
        self.block4pool = MaxPool2D((2, 2))
        #32

        self.block5conv1 = Conv2D(256, (3, 3), activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block5dropout = Dropout(0.3)
        self.block5conv2 = Conv2D(256, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block5pool=MaxPool2D((2, 2))
        # 16

        self.block6trans = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        #32 

        self.block6conv1 = Conv2D(128, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block6norm1 = BatchNormalization()
        self.block6conv2 = Conv2D(128, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block6norm2 = BatchNormalization()
        self.block7trans = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        #64
        # u6 = concatenate([u7, c3])
        self.block7conv1 = Conv2D(64, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block7norm1 = BatchNormalization()
        self.block7conv2 = Conv2D(64, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block7norm2 = BatchNormalization()


        self.block8trans = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')
        #128
        # u6 = concatenate([u8, c2])
        self.block8conv1 = Conv2D(32, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block8norm1 = BatchNormalization()
        self.block8conv2 = Conv2D(32, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block8norm2 = BatchNormalization()

        self.block9trans = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')
        #256
        # u6 = concatenate([u9, c1])
        self.block9conv1 = Conv2D(16, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block9norm1 = BatchNormalization()
        self.block9conv2 = Conv2D(16, (3, 3),activation = self.leaky,kernel_initializer='random_normal', padding='same')
        self.block9norm2 = BatchNormalization()

        self.block10trans = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')

        self.outputs = Conv2D(1, (1, 1), activation='sigmoid')
    def call(self, inputs,input_shape):
        # implement downsample layers
        inputs= tf.convert_to_tensor(inputs)
        inputs = tf.reshape(inputs,(29,512,512,1))
        b = (self.inputs(inputs))

        a= self.block1conv1(b)
        block1output = self.block1pool(self.block1norm2(self.block1conv2(self.block1norm1(a))))
        block2output = self.block2pool(self.block2norm2(self.block2conv2(self.block2norm1(self.block2conv1(block1output)))))
        block3output = self.block3pool(self.block3norm2(self.block3conv2(self.block3norm1(self.block3conv1(block2output)))))
        block4output = self.block4pool(self.block4norm2(self.block4conv2(self.block4norm1(self.block4conv1(block3output)))))
        # bottom of the u
        block5output = self.block5pool(self.block5conv2(self.block5dropout(self.block5conv1(block4output))))
        print("block 5 ", tf.shape(block5output))
        # go back up
        block6trans= self.block6trans(block5output)

        print("block 6 trtans" ,tf.shape(block6trans))
        block6cat = tf.keras.layers.concatenate([block6trans,block4output],axis=-1)
        print("block6 cat ",tf.shape(block6cat))
        block6output = self.block6norm2(self.block6conv2(self.block6norm1(self.block6conv1(block6cat))))

        block7trans= self.block7trans(block6output)
        block7cat = tf.keras.layers.concatenate([block7trans,block3output],axis=-1)
        block7output = self.block7norm2(self.block7conv2(self.block7norm1(self.block7conv1(block7cat))))
        print("block7 cat ",tf.shape(block7output))

        block8trans= self.block8trans(block7output)
        block8cat = tf.keras.layers.concatenate([block8trans,block2output],axis=-1)
        block8output = self.block8norm2(self.block8conv2(self.block8norm1(self.block8conv1(block8cat))))
        print("block8 cat ",tf.shape(block8output))

        block9trans= self.block9trans(block8output)
        block9cat = tf.keras.layers.concatenate([block9trans,block1output],axis=-1)
        block9output = self.block9norm2(self.block9conv2(self.block9norm1(self.block9conv1(block9cat))))
        print("block9 cat ",tf.shape(block9output))

        block10 = self.block10trans(block9output)
        output = self.outputs(block10)

        return output
    def loss(self, preds, labels):
        print("preds, labels")
        print(tf.shape(preds))
        print(tf.shape(labels))
        loss_numerator = -2 * tf.math.reduce_sum(tf.math.reduce_sum(tf.math.multiply(tf.squeeze(preds),labels),axis=2),axis=1)
        print(tf.shape(loss_numerator))
        pred_sum = tf.cast(tf.reduce_sum(tf.reduce_sum(tf.squeeze(preds),axis=2),axis=1),tf.float32)
        label_sum = tf.cast(tf.reduce_sum(tf.reduce_sum(labels,axis=2),axis=1),tf.float32)
        print("pred sum",tf.shape(pred_sum))
        print("label sum",tf.shape(pred_sum))

        loss_denom = pred_sum+label_sum+np.ones((29))
        print(tf.shape(loss_denom))
        return tf.reduce_mean(loss_numerator/loss_denom)
    def accuracy(self, preds,labels):
        intersection = np.count_nonzero(preds==labels) 
        union = np.count_nonzero(preds)+np.count_nonzero(labels)
        return intersection/union

# def train(model, inputs, labels):
#     loss_list= list()
#     x = len(inputs)//model.batch_size
#     inputs= inputs[:x*model.batch_size]
#     labels = inputs[:x*model.batch_size]
#     combined_inputs_labels = zip(inputs,labels)
#     for i in range(0,x*model.batch_size, model.batch_size):

#         batch_images = inputs[i:i+model.batch_size]
#         batch_labels = labels[i:i+model.batch_size]
#         with tf.GradientTape() as tape:
#             predictions = model(inputs)
#             loss_num = model.loss(predictions, batch_labels)
#         loss_list.append(loss_num)
#         gradients = tape.gradient(loss_num,model.trainable_variables)
#         model.optimizer.apply_gradients(zip(gradients,model.trainable_variables))
def train_and_checkpoint(net, manager, ckpt, inputs,labels):
    loss_list= list()
    x = len(inputs)//net.batch_size
    inputs= inputs[:x*net.batch_size]
    labels = inputs[:x*net.batch_size]
    combined_inputs_labels = zip(inputs,labels)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    for i in range(0,2):

    # for i in range(0,x*net.batch_size,  net.batch_size):
        # batch_images = inputs[i:i+net.batch_size]
        # batch_labels = labels[i:i+net.batch_size]
        with tf.GradientTape() as tape:
            # predictions = net.call(batch_images)
            # loss_num = net.loss(predictions, batch_labels)
            predictions = net.call(inputs,input_shape =(512,512,1))
            loss_num = net.loss(predictions, labels)
        loss_list.append(loss_num)
        gradients = tape.gradient(loss_num,net.trainable_variables)
        net.optimizer.apply_gradients(zip(gradients,net.trainable_variables))
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(loss_num.numpy()))
        print(loss_list)