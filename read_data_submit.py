from scipy.io import loadmat
from scipy.spatial import distance
import os, math
import numpy as np


def get_splits(splits_path='/home/wpc/master-thesis-master/srnn-copy/data/JHMDB/splits', ind_split = 1, JHMDB=True):
    train = np.array([],dtype=str)
    test = np.array([],dtype=str)

    ind_split = str(ind_split) + '.txt'
    splits_dic = {1 : train, 2: test}
    all_splits = os.listdir(splits_path)
    for splits in all_splits:
        if splits.strip().split('_')[-1] == 'split'+str(ind_split):
            file = open(os.path.join(splits_path,splits))
            for line in file:
                type = line.strip().split(' ')
                name = type[0].split('.')[0]
                if JHMDB and len(name)>30:
                    name = name[0:22]+name[-8:]
                split = int(type[1])
                splits_dic[split] = np.append(splits_dic[split], name)
            file.close()

    return splits_dic

def get_pos_imgsJHMDB(joint_positions_path='/home/wpc/master-thesis-master/srnn-copy/data/JHMDB/joint_positions',
                      sub_activities=False, normalized=False, ind_split=1, subset_joints=True):
    if sub_activities:
        actions = ['catch', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick',
               'pullup', 'push', 'run', 'shoot_ball',
               'swing_baseball','walk']
    else:
        actions = ['brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump', 'kick_ball', 'pick',
               'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'stand',
               'swing_baseball', 'throw', 'walk', 'wave']

    splits = get_splits(ind_split=ind_split, JHMDB=True)
    train_data = {}
    train_data_size=0
    valid_data = {}
    valid_data_size=0
    for i in range(len(actions)):
        all_videos = os.listdir(os.path.join(joint_positions_path,actions[i]))
        train_pos_imgs = {}
        valid_pos_imgs = {}
        for video in all_videos:
            mat_path = os.path.join(joint_positions_path,actions[i],video,'joint_positions.mat')
            if os.path.isfile(mat_path):
                joint = loadmat(os.path.join(joint_positions_path, actions[i], video,'joint_positions.mat'))
                if video in splits[1]:
                    train_data_size += 1
                    if normalized:
                        if not subset_joints:
                            train_pos_imgs[video] = joint['pos_world']
                        else:
                            train_pos_imgs[video] = joint['pos_world'][:, [2, 0, 1, 3, 4, 7, 8, 11, 12, 5, 6, 9, 10, 13, 14], :]
                            #train_pos_imgs[video] = joint['pos_world'][:, [2, 1, 3, 4, 7, 8, 11, 12, 9, 10, 13, 14], :]
                            #train_pos_imgs[video] = joint['pos_world'][:,[2,1,11,12,13,14],:]
                    else:
                        if not subset_joints:
                            train_pos_imgs[video] = joint['pos_img']
                        else:
                            #train_pos_imgs[video] = joint['pos_img'][:,[2,1,11,12,13,14],:]
                            # train_pos_imgs[video] = joint['pos_img'][:, [2, 1, 3, 4, 7, 8, 11, 12, 9, 10, 13, 14], :]
                            train_pos_imgs[video] = joint['pos_img'][:, [2, 0, 1, 3, 4, 7, 8, 11, 12, 5, 6, 9, 10, 13, 14], :]
                            # min_max_scaler = MinMaxScaler()
                            #train_pos_imgs[video] = min_max_scaler.fit_transform(train_pos_imgs[video])
                elif video in splits[2]:
                    valid_data_size += 1
                    if normalized:
                        if not subset_joints:
                            valid_pos_imgs[video] = joint['pos_world']
                        else:
                            #valid_pos_imgs[video] = joint['pos_world'][:,[2,1,11,12,13,14],:]
                            # valid_pos_imgs[video] = joint['pos_world'][:, [2, 1, 3, 4, 7, 8, 11, 12, 9, 10, 13, 14], :]
                            valid_pos_imgs[video] = joint['pos_world'][:, [2, 0, 1, 3, 4, 7, 8, 11, 12, 5, 6, 9, 10, 13, 14], :]
                    else:
                        if not subset_joints:
                            valid_pos_imgs[video] = joint['pos_img']
                        else:
                            #valid_pos_imgs[video] = joint['pos_img'][:, [2, 1, 11, 12, 13, 14], :]
                            #valid_pos_imgs[video] = joint['pos_img'][:, [2, 1, 3, 4, 7, 8, 11, 12, 9, 10, 13, 14], :]
                            valid_pos_imgs[video] = joint['pos_img'][:, [2, 0, 1, 3, 4, 7, 8, 11, 12, 5, 6, 9, 10, 13, 14], :]

        train_data[i] = train_pos_imgs
        valid_data[i] = valid_pos_imgs


    print("Num videos:")
    print(train_data_size)

    return train_data, train_data_size, valid_data, valid_data_size



def get_pos_imgsMRS_new(joint_positions_path='/home/wpc/master-thesis-master/srnn-copy/MSRAction3DSkeleton(20joints)',subset_joints=True):
    train_split = ['s01','s03','s05','s07','s09']

    train_data = {}
    train_data_size=0
    valid_data = {}
    valid_data_size=0
    actions = {}
    for i in range(1,21):
        if i<10:
            action_name = 'a0' + str(i)
        else:
            action_name = 'a'  + str(i)
        actions[action_name]=i-1
        train_data[i-1] = {}
        valid_data[i-1] = {}

    max_num_frames = -float('inf')
    min_num_frames = float('inf')

    all_videos = os.listdir(joint_positions_path)
    for video in all_videos:
        file_path = os.path.join(joint_positions_path,video)
        if os.path.isfile(file_path):
            split = video.split('_')
            action = actions[split[0]]
            pos_img, num_frames = read_pos_imgMRS(file_path)
            if split[1] in train_split:
                if subset_joints:
                    train_data[action][video] = pos_img[:, [19, 2, 6, 1, 0, 8, 7, 10, 9, 5, 4, 14, 13, 16, 15], :]
                else:
                    train_data[action][video] = pos_img
                train_data_size += 1
            else:
                if subset_joints:
                    valid_data[action][video] = pos_img[:, [19, 2, 6, 1, 0, 8, 7, 10, 9, 5, 4, 14, 13, 16, 15], :]
                else:
                    valid_data[action][video] = pos_img
                valid_data_size += 1
            if max_num_frames < num_frames:
                    max_num_frames = num_frames
            if min_num_frames > num_frames:
                    min_num_frames = num_frames

    return train_data, train_data_size, valid_data, valid_data_size,  max_num_frames, min_num_frames


def read_pos_imgMRS(file_path):
    """
    read the joint locations from a file and return a matrix similar to
    pos_img for the JHMDB data set
    :param file_path: path to the file to read
    :return: a 3D matrix containing the x and y location for each joint in each frame
    """
    file = open(file_path)
    num_frames = sum(1 for line in file)
    num_frames /= 20
    file.close()
    file = open(file_path)
    pos_img = np.zeros((3,20,num_frames))

    for frame in range(num_frames):
        for joint in range(20):
            coors = file.readline().strip().split()
            if len(coors) == 4:
                pos_img[0][joint][frame] = float(coors[0])
                pos_img[1][joint][frame] = float(coors[1])
                pos_img[2][joint][frame] = float(coors[2])
    file.close()
    return pos_img, num_frames


def extract_features(all_data, num_video, num_activities, num_considered_frames,JHMDB=False):
    print("Num videos:")
    print(num_video)
    temp_features_names = ['face-face', 'neck-neck', 'belly-belly', 'rightShoulder-rightShoulder',
                                'leftShoulder-leftShoulder',
                                'rightElbow-rightElbow', 'leftElbow-leftElbow', 'rightArm-rightArm', 'leftArm-leftArm',
                                'rightHip-rightHip', 'leftHip-leftHip',
                                'rightKnee-rightKnee', 'leftKnee-leftKnee', 'rightLeg-rightLeg', 'leftLeg-leftLeg']

    st_features_names = ['face-neck', 'face-belly', 'face-rightShoulder', 'face-leftShoulder', 'face-rightElbow',
                              'face-leftElbow', 'face-rightArm', 'face-leftArm', 'face-rightHip', 'face-leftHip',
                              'face-rightKnee', 'face-leftKnee', 'face-rightLeg', 'face-leftLeg',

                              'neck-belly', 'neck-rightShoulder', 'neck-leftShoulder', 'neck-rightElbow',
                              'neck-leftElbow',
                              'neck-rightArm', 'neck-leftArm', 'neck-rightHip', 'neck-leftHip', 'neck-rightKnee',
                              'neck-leftKnee',
                              'neck-rightLeg', 'neck-leftLeg',

                              'belly-rightShoulder', 'belly-leftShoulder', 'belly-rightElbow',
                              'belly-leftElbow', 'belly-rightArm', 'belly-leftArm',
                              'belly-rightHip', 'belly-leftHip',
                              'belly-rightKnee', 'belly-leftKnee',
                              'belly-rightLeg', 'belly-leftLeg',

                              'rightShoulder-leftShoulder', 'rightShoulder-rightElbow', 'rightShoulder-leftElbow',
                              'rightShoulder-rightArm', 'rightShoulder-leftArm', 'rightShoulder-rightHip',
                              'rightShoulder-leftHip',
                              'rightShoulder-rightKnee', 'rightShoulder-leftKnee', 'rightShoulder-rightLeg',
                              'rightShoulder-leftLeg',

                              'leftShoulder-rightElbow', 'leftShoulder-leftElbow',
                              'leftShoulder-rightArm', 'leftShoulder-leftArm', 'leftShoulder-rightHip',
                              'leftShoulder-leftHip',
                              'leftShoulder-rightKnee', 'leftShoulder-leftKnee', 'leftShoulder-rightLeg',
                              'leftShoulder-leftLeg',

                              'rightElbow-leftElbow', 'rightElbow-rightArm', 'rightElbow-leftArm',
                              'rightElbow-rightHip', 'rightElbow-leftHip',
                              'rightElbow-rightKnee', 'rightElbow-leftKnee', 'rightElbow-rightLeg',
                              'rightElbow-leftLeg',

                              'leftElbow-rightArm', 'leftElbow-leftArm', 'leftElbow-rightHip', 'leftElbow-leftHip',
                              'leftElbow-rightKnee',
                              'leftElbow-leftKnee', 'leftElbow-rightLeg', 'leftElbow-leftLeg',

                              'rightArm-leftArm', 'rightArm-rightHip', 'rightArm-leftHip', 'rightArm-rightKnee',
                              'rightArm-leftKnee',
                              'rightArm-rightLeg', 'rightArm-leftLeg',

                              'leftArm-rightHip', 'leftArm-leftHip', 'leftArm-rightKnee', 'leftArm-leftKnee',
                              'leftArm-rightLeg', 'leftArm-leftLeg',

                              'rightHip-leftHip', 'rightHip-rightKnee', 'rightHip-leftKnee', 'rightHip-rightLeg',
                              'rightHip-leftLeg',

                              'leftHip-rightKnee', 'leftHip-leftKnee', 'leftHip-rightLeg', 'leftHip-leftLeg',

                              'rightKnee-leftKnee', 'rightKnee-rightLeg', 'rightKnee-leftLeg',

                              'leftKnee-rightLeg', 'leftKnee-leftLeg',

                              'rightLeg-leftLeg']

    joints = {'face-face': 0, 'neck-neck': 1, 'belly-belly': 2, 'rightShoulder-rightShoulder': 3,
              'leftShoulder-leftShoulder': 4, 'rightElbow-rightElbow': 5, 'leftElbow-leftElbow': 6,
              'rightArm-rightArm': 7, 'leftArm-leftArm': 8, 'rightHip-rightHip': 9, 'leftHip-leftHip': 10,
              'rightKnee-rightKnee': 11, 'leftKnee-leftKnee': 12, 'rightLeg-rightLeg': 13, 'leftLeg-leftLeg': 14}

    joints_pairs = {'face-neck': [0, 1],'face-belly': [0, 2],'face-rightShoulder': [0, 3], 'face-leftShoulder': [0, 4], 'face-rightElbow':[0,5],
                    'face-leftElbow':[0, 6],'face-rightArm': [0, 7],'face-leftArm': [0, 8], 'face-rightHip':[0, 9], 'face-leftHip':[0, 10],
                    'face-rightKnee': [0, 11],'face-leftKnee': [0, 12], 'face-rightLeg':[0, 13], 'face-leftLeg':[0,14],

                    'neck-belly':[1, 2], 'neck-rightShoulder':[1, 3], 'neck-leftShoulder':[1, 4], 'neck-rightElbow':[1,5], 'neck-leftElbow':[1,6],
                    'neck-rightArm':[1,7], 'neck-leftArm':[1,8], 'neck-rightHip':[1,9], 'neck-leftHip':[1,10], 'neck-rightKnee':[1,11],'neck-leftKnee':[1,12],
                    'neck-rightLeg':[1,13], 'neck-leftLeg':[1,14],

                    'belly-rightShoulder': [2, 3], 'belly-leftShoulder': [2, 4], 'belly-rightElbow': [2, 5],
                    'belly-leftElbow': [2, 6], 'belly-rightArm': [2, 7], 'belly-leftArm': [2, 8],
                    'belly-rightHip': [2, 9], 'belly-leftHip': [2, 10],
                    'belly-rightKnee': [2, 11], 'belly-leftKnee': [2, 12],
                    'belly-rightLeg': [2, 13], 'belly-leftLeg': [2, 14],

                    'rightShoulder-leftShoulder': [3, 4], 'rightShoulder-rightElbow': [3, 5], 'rightShoulder-leftElbow': [3, 6],
                    'rightShoulder-rightArm': [3, 7], 'rightShoulder-leftArm': [3, 8], 'rightShoulder-rightHip':[3, 9], 'rightShoulder-leftHip':[3,10],
                    'rightShoulder-rightKnee':[3,11], 'rightShoulder-leftKnee':[3,12], 'rightShoulder-rightLeg':[3,13], 'rightShoulder-leftLeg':[3,14],

                    'leftShoulder-rightElbow': [4, 5], 'leftShoulder-leftElbow': [4, 6],
                    'leftShoulder-rightArm': [4, 7], 'leftShoulder-leftArm': [4, 8], 'leftShoulder-rightHip':[4,9], 'leftShoulder-leftHip':[4,10],
                    'leftShoulder-rightKnee':[4,11], 'leftShoulder-leftKnee':[4,12], 'leftShoulder-rightLeg':[4,13], 'leftShoulder-leftLeg':[4,14],

                    'rightElbow-leftElbow':[5,6], 'rightElbow-rightArm':[5,7],'rightElbow-leftArm':[5,8],'rightElbow-rightHip':[5,9],'rightElbow-leftHip':[5,10],
                    'rightElbow-rightKnee':[5,11], 'rightElbow-leftKnee':[5,12], 'rightElbow-rightLeg':[5,13],'rightElbow-leftLeg':[5,14],

                    'leftElbow-rightArm':[6,7], 'leftElbow-leftArm':[6,8], 'leftElbow-rightHip':[6,9], 'leftElbow-leftHip':[6,10], 'leftElbow-rightKnee':[6,11],
                    'leftElbow-leftKnee':[6,12],'leftElbow-rightLeg':[6,13], 'leftElbow-leftLeg':[6,14],

                    'rightArm-leftArm':[7,8],'rightArm-rightHip':[7,9], 'rightArm-leftHip':[7,10], 'rightArm-rightKnee':[7,11], 'rightArm-leftKnee':[7,12],
                    'rightArm-rightLeg':[7,13], 'rightArm-leftLeg':[7,14],

                    'leftArm-rightHip':[8,9], 'leftArm-leftHip':[8,10],'leftArm-rightKnee':[8,11],'leftArm-leftKnee':[8,12], 'leftArm-rightLeg':[8,13],'leftArm-leftLeg':[8,14],

                    'rightHip-leftHip': [9, 10],'rightHip-rightKnee': [9, 11], 'rightHip-leftKnee': [9, 12], 'rightHip-rightLeg': [9, 13],
                    'rightHip-leftLeg': [9, 14],

                    'leftHip-rightKnee': [10, 11],'leftHip-leftKnee': [10, 12], 'leftHip-rightLeg': [10, 13], 'leftHip-leftLeg': [10, 14],

                    'rightKnee-leftKnee': [11, 12],'rightKnee-rightLeg': [11, 13], 'rightKnee-leftLeg': [11, 14],

                    'leftKnee-rightLeg': [12, 13], 'leftKnee-leftLeg': [12, 14],

                    'rightLeg-leftLeg': [13, 14]}

    temp_features = {}
    infos = {}
    NUM_ST_FEATURES=1
    if JHMDB:
        NUM_TEMP_FEATURES=3
    else:
        NUM_TEMP_FEATURES=4
    for name in temp_features_names:
        temp_features[name] = np.zeros((num_video,num_considered_frames,NUM_TEMP_FEATURES))
        current_video = 0
        for action_id in all_data:
            for video in all_data[action_id]:
                infos[current_video] = [action_id, video]
                temp_features[name][current_video,:,:] = extract_ntraj(all_data[action_id][video], joints[name], num_considered_frames,JHMDB)
                current_video += 1
    st_features = {}
    for name in st_features_names:
        st_features[name] = np.zeros((num_video,num_considered_frames,NUM_ST_FEATURES))
        current_video = 0
        for action_id in all_data:
            for video in all_data[action_id]:
                st_features[name][current_video,:,:]  = extract_st_features(all_data[action_id][video], joints_pairs[name], num_considered_frames,NUM_ST_FEATURES)
                current_video += 1

    action_classes = np.zeros((num_video,num_activities))
    print('Num activities')
    print(num_activities)
    current_video = 0
    for action_id in all_data:
        for _ in all_data[action_id]:
            action_classes[current_video,action_id] = 1
            current_video += 1

    return [temp_features, st_features, action_classes, infos]


def extract_st_features(pos_img, joints_id, num_frames,NUM_ST_FEATURES):
    """
    Extract the spatio temporal features for a video for a pair of joints
    the features is the distance between the two joint at each frame
    :param pos_img: the joint positions info for a video
    :param joints_id: the id of the pair of joints to consider
    :return: the spatio temporal features
    """

    _ , _ , tot_num_frames = pos_img.shape

    frames_chosen = np.linspace(0, tot_num_frames - 1, num_frames).astype(int)
    st_features = np.zeros((num_frames, NUM_ST_FEATURES))

    for i in range(num_frames):

        dist = distance.euclidean(pos_img[:,joints_id[0],frames_chosen[i]], pos_img[:,joints_id[1],frames_chosen[i]])
        #d = pos_img[:,joints_id[0],frames_chosen[i]] - pos_img[:,joints_id[1],frames_chosen[i]]
        #ort = math.atan2(d[1],d[0])*180.0/math.pi
        st_features[i,0] = dist

    return st_features

def extract_ntraj(pos_img, joint_id, num_frames, JHMDB):

    _ , _ , tot_num_frames = pos_img.shape
    if JHMDB:
        NUM_TEMP_FEATURES = 3
    else:
        NUM_TEMP_FEATURES = 4
    #relative_pos = normalize_positions(pos_img, NUM_TEMP_FEATURES,6)
    relative_pos = normalize_positions(pos_img, NUM_TEMP_FEATURES, 15)
    frames_chosen = np.linspace(0, tot_num_frames - 1, num_frames).astype(int)

    temp_features = np.zeros((num_frames, NUM_TEMP_FEATURES))

    for i in range(num_frames-1):
        #d = pos_img[:,joint_id,frames_chosen[i+1]] - pos_img[:,joint_id,frames_chosen[i]]
        dist = distance.euclidean(pos_img[:,joint_id,frames_chosen[i+1]], pos_img[:,joint_id,frames_chosen[i]])
        #ort = math.atan2(d[1],d[0])*180.0/math.pi
        if not JHMDB:
            temp_features[i,0] = relative_pos[0,joint_id,frames_chosen[i]] #  relative x
            temp_features[i,1] = relative_pos[1,joint_id,frames_chosen[i]] #  relative y
            temp_features[i,2] = relative_pos[2,joint_id,frames_chosen[i]] #  relative z
            temp_features[i,3] = dist
        else:
            temp_features[i,0] = relative_pos[0,joint_id,frames_chosen[i]] #  relative x
            temp_features[i,1] = relative_pos[1,joint_id,frames_chosen[i]] #  relative y
            temp_features[i,2] = dist
    if not JHMDB:
        temp_features[num_frames - 1,0] = relative_pos[0,joint_id,frames_chosen[num_frames - 1]] #  realative x
        temp_features[num_frames - 1,1] = relative_pos[1,joint_id,frames_chosen[num_frames - 1]] #  reative y
        temp_features[num_frames - 1,2] = relative_pos[2,joint_id,frames_chosen[num_frames - 1]] #  reative z
        temp_features[num_frames - 1,3] = 0.0
    else:
        temp_features[num_frames - 1,0] = relative_pos[0,joint_id,frames_chosen[num_frames - 1]] #  realative x
        temp_features[num_frames - 1,1] = relative_pos[1,joint_id,frames_chosen[num_frames - 1]] #  reative y
        temp_features[num_frames - 1,2] = 0.0

    return temp_features


def normalize_positions(pos_img,num_temp_feats,num_joints):
    """
    Returns the relative positions of normalized joint positions w.r.t to the puppet center
    :param pos_img:
    :return:
    """
    # head_id = 30
    # belly_id = 1#1
    head_id = 0
    belly_id = 1  # 1
    torso_positions = (pos_img[:,head_id,:] + pos_img[:,belly_id,:])/2
    torso_positions=np.reshape(torso_positions,[num_temp_feats-1,1,pos_img.shape[2]])
    relative_pos = pos_img - np.tile(torso_positions, [1 ,num_joints, 1])
    return relative_pos