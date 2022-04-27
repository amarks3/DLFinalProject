from preprocess import *
from model import *
import tensorflow as tf

def main(): 
    i_val = 11 #change this from 1-51 
    image_slices,labels = get_data(i_val)
    print("num of slices: ", len(image_slices))


    ckpt = tf.train.Checkpoint(step=tf.Variable(1))
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    model = UnetModel()
    print("made model")
    train_and_checkpoint(model, manager, ckpt, image_slices,labels)
if __name__ == '__main__':
	main()