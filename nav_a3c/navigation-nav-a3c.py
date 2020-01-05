import time
import logging
import os
import glob
import sys
from sys import stdout
from collections import deque
import copy
import time
import argparse
import warnings
import threading
warnings.filterwarnings('ignore')

import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import scipy.misc
import scipy.signal
import pickle
import imageio
import numpy as np

import tensorflow as tf
import time

from mlagents.envs.environment import UnityEnvironment
from vis import vis_paths
import config
import network

# set logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Agent():
    def __init__(self, name, trainer, global_episode, model_path, position_path, gif_path, env, brain_name):
        self.name = name
        self.trainer = trainer
        self.global_episode = global_episode
        self.summary_writer = tf.summary.FileWriter('./log/' + name)
        self.network = network.Network(name, trainer) # local network
        from_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        to_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
        self.copy_network = [b.assign(a) for a, b in zip(from_var, to_var)] # op to sync from global network
          
        self.model_path = model_path
        self.position_path = position_path
        self.gif_path = gif_path
                 
        self.brain_name = brain_name
        self.env = env
        self.env_info = self.env.reset(train_mode=True)[brain_name]
            
    # static function to save frame during training
    def make_gif(images, fname, duration=2, true_image=False, salience=False, salIMGS=None):
        import moviepy.editor as mpy
        def make_frame(t):
            try:
                x = images[int(len(images)/duration*t)]
            except:
                x = images[-1]
            if true_image:
                return x.astype(np.uint8)
            else:
                return ((x+1)/2*255).astype(np.uint8)
        
        def make_mask(t):
            try:
                x = salIMGS[int(len(salIMGS)/duration*t)]
            except:
                x = salIMGS[-1]
            return x

        clip = mpy.VideoClip(make_frame, duration=duration)
        if salience == True:
            mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
            clipB = clip.set_mask(mask)
            clipB = clip.set_opacity(0)
            mask = mask.set_opacity(0.1)
            mask.write_gif(fname, fps = len(images) / duration,verbose=False)
        else:
            clip.write_gif(fname, fps = len(images) / duration,verbose=False)

    # static function
    def resize_image(image):
        image = image.astype(np.float32) / 255.0
        return image
        #return scipy.misc.imresize(image, [84, 84])
        
    # !!!!!
    # stack sliding windows into discret 1d array
    def window_stack(self, a, stepsize=10, trim=2):
        a = a[trim:a.shape[0]-trim, trim:a.shape[1]-trim]
        b = []
        for i in range(int(a.shape[0] / stepsize)):
            for j in range(int(a.shape[1] / stepsize)):
                b.append(np.mean(a[i*stepsize:i*stepsize+stepsize,j*stepsize:j*stepsize+stepsize]))

        b = np.array(b)
        # map to [0, 8) int
        b = b * 7
        b = b.astype(int)

        return b

    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]        

    def train(self, train_buffer, sess, boot_value):
        train_buffer = np.array(train_buffer)
        # unroll from train_buffer
        input_image = np.array(train_buffer[:, 0].tolist())
        aux_action = np.array(train_buffer[:, 1].tolist())
        aux_reward = np.array(train_buffer[:, 2:3].tolist())
        aux_velocity = np.array(train_buffer[:, 3].tolist())
        action = train_buffer[:, 4]
        reward = train_buffer[:, 5]
        value = train_buffer[:, 6]
        depth_pred = train_buffer[:, 7] # <- ?
        true_depth = np.array(train_buffer[:, 8].tolist())

        reward_plus = np.asarray(reward.tolist() + [boot_value])
        disc_reward = Agent.discount(reward_plus, config.GAMMA)[:-1]
        value_plus = np.asarray(value.tolist())
        #advantage = Agent.discount(reward + config.GAMMA*value_plus[1:] - value_plus[:-1], config.GAMMA)
        advantage = disc_reward - value_plus
        vl, pl, el, dl, dl2, gradn, _ , d_tmp= sess.run([self.network.value_loss,
            self.network.policy_loss,
            self.network.entropy_loss,
            self.network.depth_loss,
            self.network.depth_loss2,
            self.network.gradient_norm,
            self.network.apply_gradient, self.network.depth_loss_], feed_dict={
                self.network.input_image: input_image,
                self.network.input_action: aux_action,
                self.network.input_reward: aux_reward,
                self.network.input_velocity: aux_velocity,
                self.network.true_value: disc_reward,
                self.network.advantage: advantage,
                self.network.action: action,
                self.network.true_depth: true_depth,
                self.network.lstm1_state_c_in: self.train_lstm1_state_c,
                self.network.lstm1_state_h_in: self.train_lstm1_state_h,
                self.network.lstm2_state_c_in: self.train_lstm2_state_c,
                self.network.lstm2_state_h_in: self.train_lstm2_state_h
            })
        sys.stdout.flush()
        return vl, pl, el, dl, dl2, gradn, _

    def get_action(self, action):
        move_forward  = action // 3
        rotate = action % 3
        return [move_forward, rotate]

    def get_vel(self, prev_coord, coord, action_transformed):
        if action_transformed[0] == 0:
            distance = np.linalg.norm(coord-prev_coord)
            # cos 15 degrees, corresponding to turning angle in unity
            if action_transformed[1] == 0:
                result = [0.96*distance, (coord[1]-prev_coord[1]), -0.26*distance, 0, -15, 0]
            if action_transformed[1] == 1:
                # cos 15 degrees, corresponding to turning angle in unity
                result = [distance, (coord[1]-prev_coord[1]), 0, 0, 0, 0]
            if action_transformed[1] == 2:
                result = [0.96*distance, (coord[1]-prev_coord[1]), 0.26*distance, 0, 15, 0]

        elif action_transformed[0] == 1:
            if action_transformed[1] == 0:
                result = [0, 0, 0, 0, -15, 0]
            if action_transformed[1] == 1:
                # cos 15 degrees, corresponding to turning angle in unity
                result = [0, 0, 0, 0, 0, 0]
            if action_transformed[1] == 2:
                result = [0, 0, 0, 0, 15, 0]
        else:
            assert 0, "action_transformed's first element is unknown"

        result = np.array(result)
        return result

    def run(self, sess, trainer, saver, coordinator):
        print('starting agent:', self.name)
        sys.stdout.flush()
        with sess.as_default(), sess.graph.as_default():
            while not coordinator.should_stop():
                sess.run(self.global_episode.assign_add(1))
                print('episode:', sess.run(self.global_episode))
                sys.stdout.flush()
                
                ep = sess.run(self.global_episode)
                ep_reward = 0
                ep_step = 0
                ep_start_time = time.time()

                sess.run(self.copy_network)
                self.train_buffer = []
                frame_buffer = []
                running = True

                # !!!!!!
                # self.game.reset()
                # rgb, prev_d = self.game.frame()
                rgb = np.asarray(self.env_info.visual_observations[1][0])
                prev_d = self.window_stack(np.asarray(self.env_info.visual_observations[0][0]))

                frame_buffer.append(rgb * 255)
                # rgb = Agent.resize_image(rgb)
                prev_act_idx = 0
                prev_reward = 0
                prev_vel = np.array([0.0]*6)
                prev_coord = self.env_info.vector_observations[0][-3:]

                self.lstm1_state_c, self.lstm1_state_h, self.lstm2_state_c, self.lstm2_state_h = self.network.lstm1_init_state_c, self.network.lstm1_init_state_h,self.network.lstm2_init_state_c,self.network.lstm2_init_state_h
                
                self.env_info = self.env.reset(train_mode=True)[self.brain_name]
                print('initial position:({:.2f} {:.2f} {:.2f})'.format(self.env_info.vector_observations[0][-3:][0], 
                                                                       self.env_info.vector_observations[0][-3:][1],
                                                                       self.env_info.vector_observations[0][-3:][2]))
                
                # !!!!!!
                # while self.game.running():
                ep_step_max = 1000
                positions = []
                while True:
                    if len(self.train_buffer)==0:
                        self.train_lstm1_state_h = self.lstm1_state_h
                        self.train_lstm1_state_c = self.lstm1_state_c
                        self.train_lstm2_state_h = self.lstm2_state_h
                        self.train_lstm2_state_c = self.lstm2_state_c
                    act_prob, pred_value, depth_pred, self.lstm1_state_c, self.lstm1_state_h, self.lstm2_state_c, self.lstm2_state_h = sess.run([self.network.policy,
                        self.network.value, self.network.depth_pred,
                        self.network.lstm1_state_c_out, 
                        self.network.lstm1_state_h_out, 
                        self.network.lstm2_state_c_out, 
                        self.network.lstm2_state_h_out]
                        , 
                        feed_dict={self.network.input_image: [rgb], 
                        self.network.input_action: [prev_act_idx], 
                        self.network.input_reward: [[prev_reward]], 
                        self.network.input_velocity: [prev_vel],
                        self.network.lstm1_state_c_in:self.lstm1_state_c,
                        self.network.lstm1_state_h_in:self.lstm1_state_h,
                        self.network.lstm2_state_c_in:self.lstm2_state_c,
                        self.network.lstm2_state_h_in:self.lstm2_state_h
                    })
                    
                    self.action_visualize = act_prob
                    action = np.random.choice(act_prob[0], p=act_prob[0])
                    action_idx = np.argmax(act_prob==action)
                    
                    # !!!!!!
                    # rgb_next, d, vel, reward, running = self.game.step(action_idx)
                    action_transformed = self.get_action(action_idx)
                    self.env_info = self.env.step(action_transformed)[self.brain_name] # send the action to the environment
                    rgb_next = np.asarray(self.env_info.visual_observations[1][0])  # get the next state
                    d = self.window_stack(np.asarray(self.env_info.visual_observations[0][0]))

                    reward = self.env_info.rewards[0]                   # get the reward
                    done = self.env_info.local_done[0]                  # see if episode has finished
                    coord = self.env_info.vector_observations[0][-3:]

                    # !!!!!!
                    # only an estimation, get velocity from previous and current coordination, and action vector
                    vel = self.get_vel(prev_coord, coord, action_transformed)
                    
                    sys.stdout.write('\r episode:{}, step: {}, position:({:.2f} {:.2f} {:.2f}), score:{:.2f}, action:{}'.format(self.name, ep_step,
                        self.env_info.vector_observations[0][-3:][0], self.env_info.vector_observations[0][-3:][1],
                        self.env_info.vector_observations[0][-3:][2], ep_reward, str(action_idx)))
                    sys.stdout.flush()
                    
                    positions.append([self.env_info.vector_observations[0][-3:][0], 
                                     self.env_info.vector_observations[0][-3:][1], 
                                     self.env_info.vector_observations[0][-3:][2]])
                    
                    self.train_buffer.append([rgb, prev_act_idx, prev_reward, prev_vel, action_idx, 
                                         reward, pred_value[0][0], depth_pred, prev_d])

                    ep_reward += reward
                    ep_step += 1
                    
                    running = not ((ep_step >= ep_step_max) or done)

                    if running:
                        if ep%config.SAVE_PERIOD==0:
                            frame_buffer.append(rgb_next * 255)
                        # rgb_next = Agent.resize_image(rgb_next)
                        rgb = rgb_next
                    
                    prev_act_idx = action_idx
                    prev_reward = reward
                    prev_vel = vel
                    prev_d = d

                    if len(self.train_buffer)==config.GRADIENT_CHUNK and running:
                        boot_value = sess.run(self.network.value, feed_dict={
                            self.network.input_image: [rgb], 
                            self.network.input_action: [prev_act_idx], 
                            self.network.input_reward: [[prev_reward]], 
                            self.network.input_velocity: [prev_vel],
                            self.network.lstm1_state_c_in:self.lstm1_state_c,
                            self.network.lstm1_state_h_in:self.lstm1_state_h,
                            self.network.lstm2_state_c_in:self.lstm2_state_c,
                            self.network.lstm2_state_h_in:self.lstm2_state_h
                        })
                        vl, pl, el, dl, dl2, gradn, _ = self.train(self.train_buffer, sess, boot_value)
                        self.train_buffer = []
                        sess.run(self.copy_network)
                    if not running:
                        break
                if len(self.train_buffer)>0:
                    vl, pl, el, dl, dl2, gradn, _ = self.train(self.train_buffer, sess, 0.0)
                    self.test1 = [vl, pl, el, dl, dl2, gradn]

                ep_finish_time = time.time()
                print(self.name, 'elapse', str(int(ep_finish_time-ep_start_time)), 'seconds, reward:',ep_reward)
                sys.stdout.flush()

                
                if ep%config.SAVE_PERIOD==0:
                    imgs = np.array(frame_buffer)
                    Agent.make_gif(imgs, self.gif_path + str(ep)+'_'+str(ep_reward)+'.gif', duration=len(imgs)*0.066, true_image=True, salience=False)
                    print('frame saved')
                    sys.stdout.flush()
                

                if ep%config.SAVE_PERIOD==0:
                    saver.save(sess, self.model_path+'/model'+str(ep)+'.cptk')
                    print('model saved')
                    sys.stdout.flush()

                    summary = tf.Summary()
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(pl))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(vl))
                    summary.value.add(tag='Losses/Entropy Loss', simple_value=float(el))
                    summary.value.add(tag='Losses/Depth Loss', simple_value=float(dl))
                    summary.value.add(tag='Losses/Depth Loss2', simple_value=float(dl2))
                    summary.value.add(tag='Losses/Gradient Norm', simple_value=float(gradn))
                    summary.value.add(tag='Performance/Reward', simple_value=float(ep_reward))
                    self.summary_writer.add_summary(summary, ep)
                    self.summary_writer.flush()
                    
                if ep%config.SAVE_PERIOD==0:
                    print('save positions')
                    sys.stdout.flush()
                    
                    with open(self.position_path + '/' + str(ep), 'wb') as fp:
                        pickle.dump(positions, fp)

    def evaluate(self, sess, saver, coordinator):
        print('evaluation:', self.name)
        sys.stdout.flush()
        with sess.as_default(), sess.graph.as_default():
            while not coordinator.should_stop():
                sess.run(self.global_episode.assign_add(1))
                print('episode:', sess.run(self.global_episode))
                sys.stdout.flush()
                
                ep = sess.run(self.global_episode)
                ep_reward = 0
                ep_step = 0

                sess.run(self.copy_network)
                frame_buffer = []
                running = True

                rgb = np.asarray(self.env_info.visual_observations[1][0])

                frame_buffer.append(rgb * 255)
                
                self.lstm1_state_c, self.lstm1_state_h, self.lstm2_state_c, self.lstm2_state_h = self.network.lstm1_init_state_c, self.network.lstm1_init_state_h,self.network.lstm2_init_state_c,self.network.lstm2_init_state_h
                
                self.env_info = self.env.reset(train_mode=True)[self.brain_name]
                print('initial position:({:.2f} {:.2f} {:.2f})'.format(self.env_info.vector_observations[0][-3:][0], 
                                                                       self.env_info.vector_observations[0][-3:][1],
                                                                       self.env_info.vector_observations[0][-3:][2]))

                ep_step_max = 1000
                positions = []
                while True:
                    
                    self.train_lstm1_state_h = self.lstm1_state_h
                    self.train_lstm1_state_c = self.lstm1_state_c
                    self.train_lstm2_state_h = self.lstm2_state_h
                    self.train_lstm2_state_c = self.lstm2_state_c
                    act_prob, pred_value, depth_pred, self.lstm1_state_c, self.lstm1_state_h, self.lstm2_state_c, self.lstm2_state_h = sess.run([self.network.policy,
                        self.network.value, self.network.depth_pred,
                        self.network.lstm1_state_c_out, 
                        self.network.lstm1_state_h_out, 
                        self.network.lstm2_state_c_out, 
                        self.network.lstm2_state_h_out], 
                        feed_dict={self.network.input_image: [rgb], 
                        self.network.input_action: [prev_act_idx], 
                        self.network.input_reward: [[prev_reward]], 
                        self.network.input_velocity: [prev_vel],
                        self.network.lstm1_state_c_in:self.lstm1_state_c,
                        self.network.lstm1_state_h_in:self.lstm1_state_h,
                        self.network.lstm2_state_c_in:self.lstm2_state_c,
                        self.network.lstm2_state_h_in:self.lstm2_state_h})
                    
                    self.action_visualize = act_prob
                    action = np.random.choice(act_prob[0], p=act_prob[0])
                    action_idx = np.argmax(act_prob==action)
                    
                    action_transformed = self.get_action(action_idx)
                    self.env_info = self.env.step(action_transformed)[self.brain_name] # send the action to the environment
                    rgb_next = np.asarray(self.env_info.visual_observations[1][0])  # get the next state
                    d = self.window_stack(np.asarray(self.env_info.visual_observations[0][0]))

                    reward = self.env_info.rewards[0]                   # get the reward
                    done = self.env_info.local_done[0]                  # see if episode has finished
                    coord = self.env_info.vector_observations[0][-3:]

                    vel = self.get_vel(prev_coord, coord, action_transformed)
                    
                    sys.stdout.write('\r episode:{}, step: {}, position:({:.2f} {:.2f} {:.2f}), score:{:.2f}, action:{}'.format(self.name, ep_step,
                        self.env_info.vector_observations[0][-3:][0], self.env_info.vector_observations[0][-3:][1],
                        self.env_info.vector_observations[0][-3:][2], ep_reward, str(action_idx)))
                    sys.stdout.flush()
                    
                    positions.append([self.env_info.vector_observations[0][-3:][0], 
                                     self.env_info.vector_observations[0][-3:][1], 
                                     self.env_info.vector_observations[0][-3:][2]])
                    
                    ep_reward += reward
                    ep_step += 1
                    
                    running = not ((ep_step >= ep_step_max) or done)

                    if running:
                        if ep%config.SAVE_PERIOD==0:
                            frame_buffer.append(rgb_next * 255)
                        # rgb_next = Agent.resize_image(rgb_next)
                        rgb = rgb_next
                    
                    if not running:
                        break
                        
                if ep%config.SAVE_PERIOD==0:
                    imgs = np.array(frame_buffer)
                    Agent.make_gif(imgs, './frame/image'+str(ep)+'_'+str(ep_reward)+'.gif', duration=len(imgs)*0.066, true_image=True, salience=False)
                    print('frame saved')
                    sys.stdout.flush()
                                    
                if ep%config.SAVE_PERIOD==0:
                    print('save positions')
                    sys.stdout.flush()
                    
                    with open('./positions/'+str(ep), 'wb') as fp:
                        pickle.dump(positions, fp)
                        
       
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action="store_true", default=False, required=False)
    parser.add_argument('--run_id', default='Null', required=False)
    parser.add_argument('--env_name', default='Null', required=False)
    
    args = parser.parse_args()
    evaluate = args.evaluate
    env_name = args.env_name
    run_id   = args.run_id
    print(evaluate, env_name, run_id)
    # model_path = "./model/1226_final_without_visual_goals"
    # model_time = "1577341381.6629117"
    # env_name = '1226_final_without_visual_goals'
                 
    env_path = "C:/data/ml-agents-old/scripts/envs/{}/HBF-navigation-experiment.exe".format(env_name)

    # env = UnityEnvironment(file_name="./envs/%s/%s.x86_64" % (env_name, env_name), worker_id=1, seed=1, no_graphics=False)
    env = UnityEnvironment(file_name=env_path, worker_id=0, seed=1, no_graphics=False)
    env.step()
    default_brain = env.external_brain_names[0]
    brain = env.brains[default_brain]
    env_info = env.reset(train_mode=True)[default_brain]

    print('brain_name', brain.brain_name)
    print('camera_resolutions', brain.camera_resolutions)
    print('number_visual_observations', brain.number_visual_observations)
    print('vector_action_descriptions', brain.vector_action_descriptions)
    print('vector_action_space_size', brain.vector_action_space_size)
    print('vector_action_space_type', brain.vector_action_space_type)
    print('vector_observation_space_size', brain.vector_observation_space_size)

    # examine the visual space
    img_rgb = np.asarray(env_info.visual_observations[1][0])
    imgplt = plt.imshow(img_rgb)
    height, width, channel = img_rgb.shape
    logger.info('Shape of image: %s' % str(img_rgb.shape))

    env_info = env.reset(train_mode=True)[default_brain]

    model_path = './model/%s' % run_id
    gif_path = './frame/%s' % run_id
    position_path = './position/%s' % run_id
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
                 
    with tf.device('cpu:0'):
        global_episode = tf.Variable(0, trainable=False, dtype=tf.int32)
        trainer = tf.train.RMSPropOptimizer(config.LEARNING_RATE, decay=config.DECAY, 
                                            momentum=config.MOMENTUM, epsilon=config.EPSILON)
        master_network = network.Network('global', trainer)
        print('master network created')
        sys.stdout.flush()
        agent_arr = []
        if evaluate == False:
            for i in range(config.THREAD):
                agent_arr.append(Agent(run_id, trainer, global_episode, 
                                       model_path, position_path, gif_path, env, default_brain))
        else:
            for i in range(config.THREAD):
                agent_arr.append(Agent(run_id, trainer, global_episode, 
                                       model_path, position_path, gif_path, env, default_brain))
        saver = tf.train.Saver()
                 
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())
        if evaluate == False:
            thread_arr = []
            for a in agent_arr:
                _ = lambda: a.run(sess, trainer, saver, coord)
                t = threading.Thread(target=(_))
                t.start()
                print('thread started')
                sys.stdout.flush()
                time.sleep(1)
                thread_arr.append(t)
            coord.join(thread_arr)
        elif evaluate:
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            thread_arr = []
            for a in agent_arr:
    #             _ = lambda: a.run(sess, trainer, saver, coord)
    #             t = threading.Thread(target=(_))
    #             t.start()
    #             print('thread started')
    #             sys.stdout.flush()
    #             time.sleep(1)
                thread_arr.append(t)
            coord.join(thread_arr)
                 
if __name__ == "__main__":
    main()
                 
                 