# --------------------------------------------------------
# PyTorch Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Kevin Cao, based on code from Jianwei Yang
# --------------------------------------------------------

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import pynvml


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc_face', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res18.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res18', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=7, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=4551, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--video', dest='video_file_name',
                        help='Video File Name',
                        default='video.mp4', type=str)

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
    im_orig = im.astype(np.float32, copy=True)
    # changed to use pytorch models
    im_orig /= 255.  # Convert range to [0,1]
    pixel_means = [0.485, 0.456, 0.406]
    im_orig -= pixel_means  # Minus mean
    pixel_stdens = [0.229, 0.224, 0.225]
    im_orig /= pixel_stdens  # divide by stddev

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    # for target_size in cfg.TEST.SCALES:
    #     im_scale = float(target_size) / float(im_size_min)
    #     # Prevent the biggest axis from being more than MAX_SIZE
    #     if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
    #         im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    #     im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
    #                     interpolation=cv2.INTER_LINEAR)
    #     im_scale_factors.append(im_scale)
    #     processed_ims.append(im)

    # no need to change size
    im_scale_factors.append(1.0)
    processed_ims.append(im_orig)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


if __name__ == '__main__':

    # 这里的0是GPU id
    GPU_id = 0
    pynvml.nvmlInit()

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        pascal_classes = np.asarray(['__background__',
                                     'aeroplane', 'bicycle', 'bird', 'boat',
                                     'bottle', 'bus', 'car', 'cat', 'chair',
                                     'cow', 'diningtable', 'dog', 'horse',
                                     'motorbike', 'person', 'pottedplant',
                                     'sheep', 'sofa', 'train', 'tvmonitor'])
    elif args.dataset == "pascal_voc_face":
        pascal_classes = None
    elif args.dataset == "pascal_voc_0712":
        pascal_classes = None
    elif args.dataset == "coco":
        pascal_classes = None
    elif args.dataset == "imagenet":
        pascal_classes = None
    elif args.dataset == "vg":
        pascal_classes = None

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res18':
        fasterRCNN = resnet(pascal_classes, 18, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res34':
        fasterRCNN = resnet(pascal_classes, 34, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % load_name)
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # 显示显存
    handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('GPU memery used: %.10f G' % (meminfo.used / (1024 * 1024 * 1024)), 'after load the weight')
    # pdb.set_trace()

    print('load video')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = False

    webcam_num = args.webcam_num
    video_file_name = args.video_file_name
    # Set up webcam or get image from video
    if webcam_num >= 0:
        cap = cv2.VideoCapture(webcam_num)
        num_images = 0
    else:
        # use opencv open the video
        cap = cv2.VideoCapture(video_file_name)
        # cap = cv2.VideoCapture('videocapture.avi')
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('num_frame: ', num_frame)
        num_images = num_frame

    # get the inf of vedio,fps and size
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # point out how to encode videos
    # I420-avi=>cv2.cv.CV_FOURCC('X','2','6','4');
    # MP4=>cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    videowriter = cv2.VideoWriter('video_det.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)

    print('Loaded Photo: {} images.'.format(num_images))

    # 显示显存
    handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('GPU memery used: %.10f G' % (meminfo.used / (1024 * 1024 * 1024)), 'before img itr')
    pynvml.nvmlShutdown()

    while num_images > 0:
        total_tic = time.time()
        if webcam_num == -1:
            num_images -= 1
        # print('num_images: %d ' % num_images)

        # Get image from the webcam
        if webcam_num >= 0:
            if not cap.isOpened():
                raise RuntimeError("Webcam could not open. Please check connection.")
            ret, frame_bgr = cap.read()
            im_bgr = np.array(frame_bgr)
            # bgr -> rgb
            im_rgb = im_bgr[:, :, ::-1]
        # Load the demo image
        else:
            if not cap.isOpened():
                raise RuntimeError("Video file could not open. Please check file path is ture.")
            # read one frame from the video
            ret, frame_bgr = cap.read()
            im_bgr = np.array(frame_bgr)
            # bgr -> rgb
            im_rgb = im_bgr[:, :, ::-1]
        if len(im_rgb.shape) == 2:
            im_rgb = im_rgb[:, :, np.newaxis]
            im_rgb = np.concatenate((im_rgb, im_rgb, im_rgb), axis=2)
        # in image is rgb
        im_in = im_rgb

        blobs, im_scales = _get_image_blob(im_in)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()

        # 显示显存
        # handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_id)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print('GPU memery used: %.10f G' % (meminfo.used / (1024 * 1024 * 1024)), 'befor go in net', num_images+1)

        # pdb.set_trace()
        det_tic = time.time()

        # with torch.no_grad():
        #     rois, cls_prob, bbox_pred, \
        #     rpn_loss_cls, rpn_loss_box, \
        #     RCNN_loss_cls, RCNN_loss_bbox, \
        #     rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        # 显示显存
        # handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_id)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print('GPU memery used: %.10f G' % (meminfo.used / (1024 * 1024 * 1024)), 'after go in net', num_images+1)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        im2show = np.copy(frame_bgr)
        for j in range(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                # add boxes to img
                im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        if vis: 
            cv2.imshow('frame', cv2.resize(im2show, None, fx=0.35, fy=0.35))
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        # result_path = os.path.join(args.image_dir, str(num_images) + "_det.jpg")
        # cv2.imwrite(result_path, im2show)
        videowriter.write(im2show)  # write one frame into the output video
        # fps caulate
        total_toc = time.time()
        total_time = total_toc - total_tic
        frame_rate = 1 / total_time
        # need time caulate
        need_time = num_images / frame_rate
        # print sys
        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s  fps: {:.3f} Hz need_time: {:.3f}s \r' \
                         .format(num_images + 1, num_frame, detect_time, nms_time, total_time, frame_rate, need_time))
        sys.stdout.flush()

        # 清除缓存
        # torch.cuda.empty_cache()
        # 显示显存
        # handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_id)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print('GPU memery used: %.10f G' % (meminfo.used / (1024 * 1024 * 1024)), 'after empty cache')

    if webcam_num >= 0:
        cap.release()
        videowriter.release()
        cv2.destroyAllWindows()
