from preprocess import *
from model import *
import tensorflow as tf

def main():
    i_val = 3 #change this from 1-51
    data, labels = get_data(i_val)
    print("num of slices: ", len(data))


    ckpt = tf.train.Checkpoint(step=tf.Variable(1))
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
    model = UnetModel()
    print("made model")
    train_and_checkpoint(model, manager, ckpt, data,labels)
if __name__ == '__main__':
	main()
