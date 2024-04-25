import numpy as np
import keras
from keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

from viz import *
from data_utils import *
from forward_kinematics import *
import transformer_model

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.config.run_functions_eagerly(True)

import os
import argparse

parser = argparse.ArgumentParser(description='Arguments')

# Learning
parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate.')
# parser.add_argument('--learning_rate_decay_factor', type=float, default=0.95, help='Learning rate is multiplied by this much. 1 means no decay.')
# parser.add_argument('--learning_rate_step', type=int, default=10000, help='Every this many steps, do decay.')
# parser.add_argument('--max_gradient_norm', type=float, default=5, help='Clip gradients to this norm.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size to use during training.')
parser.add_argument('--iterations', type=int, default=15, help='Iterations to train for.')

# Architecture
# parser.add_argument('--architecture', type=str, default='tied', help='Seq2seq architecture to use: [basic, tied].')
parser.add_argument('--d_model', type=int, default=128, help='Size of each model layer.')
parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in the model.')
parser.add_argument('--n_heads', type=int, default=4, help='Number of heads in the model.')
parser.add_argument('--dropout', type=int, default=0.1, help='Number of layers in the model.')
parser.add_argument('--seq_length_in', type=int, default=50, help='Number of frames to feed into the encoder. 25 fps')
parser.add_argument('--seq_length_out', type=int, default=20, help='Number of frames that the decoder has to predict. 25fps')
parser.add_argument('--omit_one_hot', type=bool, default=False, help='Whether to remove one-hot encoding from the data')
# parser.add_argument('--residual_velocities', type=bool, default=False, help='Add a residual connection that effectively models velocities')

# Directories
parser.add_argument('--data_dir', type=str, default=os.path.normpath("C:/Users/khush/Documents/PSU/Academics/Spring2024/CSE586_CV/Project/data/h36m/dataset"), help='Data directory')
parser.add_argument('--checkpoint_dir', type=str, default=r"C:\Users\khush\Documents\PSU\Academics\Spring2024\CSE586_CV\Project\transformer\checkpoints", help='Checkpoint directory.')

parser.add_argument('--action', type=str, default="all", help='The action to train on. all means all the actions, all_periodic means walking, eating and smoking')
# parser.add_argument('--loss_to_use', type=str, default="sampling_based", help='The type of loss to use, supervised or sampling_based')

parser.add_argument('--test_every', type=int, default=10, help='How often to compute error on the test set.')
parser.add_argument('--save_every', type=int, default=10, help='How often to compute error on the test set.')
parser.add_argument('--train', type=bool, default=True, help='Set to True for Training.')
# parser.add_argument('--use_cpu', type=bool, default=False, help='Whether to use the CPU')
parser.add_argument('--load', type=int, default=0, help='Try to load a previous checkpoint.')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose.')


FLAGS = parser.parse_args()

def create_model(input_dim, output_dim, d_model, n_layers, n_heads, dropout):
    model = transformer_model.Transformer(input_dim, output_dim, FLAGS.d_model, FLAGS.n_layers, FLAGS.n_heads, FLAGS.dropout)
 
    if FLAGS.load <= 0:
        if FLAGS.verbose:
            print("Creating model with fresh parameters.")
        return model

    ckpt = f"{FLAGS.checkpoint_dir}\\best_model.pth"
    if FLAGS.verbose:
        print( "Checkpoint_dir", FLAGS.checkpoint_dir )
    
    if ckpt and FLAGS.load > 0:
        if FLAGS.verbose:
            print("Loading model {0}".format(ckpt))
        model.load_state_dict(torch.load(ckpt))
    else:
        print("Could not find checkpoint. Aborting.")
        
    return model

def train():
    actions = define_actions(FLAGS.action)
    number_of_actions = len( actions )
    
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot)

    dtype=tf.float32
    source_seq_len = FLAGS.seq_length_in
    target_seq_len = FLAGS.seq_length_out

    if FLAGS.action == "all":
        input_dim = 54
        output_dim = 54
    else:
        input_dim = 49
        output_dim = 49
        
    enc_in = tf.placeholder(dtype, shape=[None, source_seq_len-1, input_dim], name="enc_in")
    dec_in = tf.placeholder(dtype, shape=[None, target_seq_len, input_dim], name="dec_in")
    dec_out = tf.placeholder(dtype, shape=[None, target_seq_len, input_dim], name="dec_out")

    enc_in = tf.transpose(enc_in, [1, 0, 2])
    dec_in = tf.transpose(dec_in, [1, 0, 2])
    dec_out = tf.transpose(dec_out, [1, 0, 2])

    enc_in = tf.reshape(enc_in, [-1, input_dim])
    dec_in = tf.reshape(dec_in, [-1, input_dim])
    dec_out = tf.reshape(dec_out, [-1, input_dim])

    enc_in = tf.split(enc_in, source_seq_len-1, axis=0)
    dec_in = tf.split(dec_in, target_seq_len, axis=0)
    dec_out = tf.split(dec_out, target_seq_len, axis=0)

    train_loss = []
    
    best_checkpoint_path = f"{FLAGS.checkpoint_dir}\\best_model.pth"
    
    model = create_model(input_dim, output_dim, FLAGS.d_model, FLAGS.n_layers, FLAGS.n_heads, FLAGS.dropout)
    if FLAGS.verbose:
        print( "Model created" )
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    
    best_metric = float('inf')
  
        # Training loop
    for epoch in range(FLAGS.iterations):
        # Batch Data     
        encoder_inputs, decoder_inputs, decoder_outputs = get_batch(train_set, actions, source_seq_len, target_seq_len, input_dim, FLAGS.batch_size)

        # Forward pass
        outputs = model(encoder_inputs, decoder_inputs)

        # Calculate loss
        loss = torch.mean((decoder_outputs - outputs)**2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Print loss after each epoch
        if FLAGS.verbose:
            print(f'Epoch [{epoch+1}/{FLAGS.iterations}], Train Loss: {loss.item():.4f}')
        train_loss.append(loss.item())

        if (epoch + 1) % FLAGS.save_every == 0:
            checkpoint_path = FLAGS.checkpoint_dir + f'\\model_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), checkpoint_path)

        if loss < best_metric: 
            best_metric = loss
            torch.save(model.state_dict(), best_checkpoint_path)

        if (epoch + 1) % FLAGS.test_every == 0:
            outputs = []
            test_loss = []
            for action in actions:
                encoder_inputs, decoder_inputs, decoder_outputs = get_batch(test_set, action, source_seq_len, target_seq_len, input_dim, FLAGS.batch_size)

                outputs = model(encoder_inputs, decoder_inputs)

                loss = torch.mean((decoder_outputs - outputs)**2)
                if FLAGS.verbose:
                    print(f'Action: {action}, Test Loss: {loss.item():.4f}')
                test_loss.append(loss)
    
    if os.path.exists(r'./losses.txt'): 
        os.remove(r'./losses.txt')
        
    with open(r'./losses.txt', 'w') as fp:
        for i in range(len(train_loss)): 
            fp.write(f'Epoch [{i+1}/{FLAGS.iterations}], Train Loss: {train_loss[i]:.4f} \n')
        j = 0
        for action in actions: 
            fp.write(f'Action: {action}, Test Loss: {test_loss[j]:.4f} \n')
            j+=1
            
def test():
    if FLAGS.verbose:
        print("Testing")
    
    actions = define_actions(FLAGS.action)
    number_of_actions = len( actions )
    
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot)
    
    if FLAGS.action == "all":
        input_dim = 54
        output_dim = 54
    else:
        input_dim = 49
        output_dim = 49
        
    source_seq_len = FLAGS.seq_length_in
    target_seq_len = FLAGS.seq_length_out
    
    SAMPLES_FNAME = 'samples.h5'
    try:
        os.remove( SAMPLES_FNAME )
    except OSError:
        pass
    
    gts_expmap = get_gts(actions, model, test_set, data_mean, data_std, dim_to_ignore, FLAGS.omit_one_hot, source_seq_len, target_seq_len, input_dim, to_euler=False )
    
    FLAGS.load = 2
    
    model = create_model(input_dim, output_dim, FLAGS.d_model, FLAGS.n_layers, FLAGS.n_heads, FLAGS.dropout)

    outputs = []
    test_loss = []
    
    for action in actions:
        encoder_inputs, decoder_inputs, decoder_outputs = get_batch(test_set, action, source_seq_len, target_seq_len, input_dim, FLAGS.batch_size)

        outputs = model(encoder_inputs, decoder_inputs)

        loss = torch.mean((decoder_outputs - outputs)**2)
        if Flags.verbose:
            print(f'Action: {action}, Loss: {loss.item():.4f}')
        test_loss.append(loss)

        pred_expmap = revert_output_format(outputs, data_mean, data_std, dim_to_ignore, action, FLAGS.omit_one_hot )

          # Save the samples
        with h5py.File( SAMPLES_FNAME, 'a' ) as hf:
            for i in np.arange(10):
                # Save conditioning ground truth
                node_name = 'expmap/gt/{1}_{0}'.format(i, action)
                hf.create_dataset( node_name, data=gts_expmap[action][0][i] )
                
                # Save prediction
                node_name = 'expmap/preds/{1}_{0}'.format(i, action)
                hf.create_dataset( node_name, data=pred_expmap[i] )                             
    
    if os.path.exists(r'./test_loss.txt'): 
        os.remove(r'./test_loss.txt') 

    with open(r'./test_loss.txt', 'w') as fp:
        j = 0
        for action in actions: 
            fp.write(f'Action: {action}, Test Loss: {test_loss[j]:.4f} \n')
            j+=1 
    
def main():
    FLAGS = parser.parse_args()
    if FLAGS.train:
        train()
    else:
        test()

if __name__ == "__main__":
    main()

                