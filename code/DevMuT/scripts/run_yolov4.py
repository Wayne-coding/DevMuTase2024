# coding=utf-8
import os
import sys
from collections import defaultdict
from datetime import datetime
import numpy as np
import psutil
import torch
import mindspore
import mindspore as ms
from mindspore import Tensor
from common.dataset_utils import get_dataset
from common.loss_utils import get_loss
from common.opt_utils import get_optimizer
from network.cv.yolov4.model_utils.config import config as yolov4config
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
from common.analyzelog_util import train_result_analyze

if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
    final_device = 'cuda:0'
else:
    final_device = 'cpu'


def com_mAP(detection, name="torch"):
    detection.do_nms_for_results()
    result_file_path = detection.write_result()
    eval_result = detection.get_eval_result()



def seteval(network):
    network.detect_1.conf_training = False
    network.detect_2.conf_training = False
    network.detect_3.conf_training = False
    return network


def settrain(network):
    network.detect_1.conf_training = True
    network.detect_2.conf_training = True
    network.detect_3.conf_training = True
    return network


class Redirct:
    def __init__(self):
        self.content = ""

    def write(self, content):
        self.content += content

    def flush(self):
        self.content = ""


class DetectionEngine:
    """Detection engine."""

    def __init__(self, args_detection):
        self.eval_ignore_threshold = args_detection.eval_ignore_threshold
        self.labels = yolov4config.labels
        self.num_classes = len(self.labels)
        self.results = {}
        self.file_path = ''
        self.save_prefix = args_detection.outputs_dir
        self.ann_file = args_detection.val_ann_file
        self._coco = COCO(self.ann_file)
        self._img_ids = list(sorted(self._coco.imgs.keys()))
        self.det_boxes = []
        self.nms_thresh = args_detection.nms_thresh
        self.coco_catids = self._coco.getCatIds()
        self.multi_label = yolov4config.multi_label
        self.multi_label_thresh = yolov4config.multi_label_thresh

    def do_nms_for_results(self):
        """Get result boxes."""
        for img_id in self.results:
            for clsi in self.results[img_id]:
                dets = self.results[img_id][clsi]
                dets = np.array(dets)
                keep_index = self._diou_nms(dets, thresh=0.6)

                keep_box = [{'image_id': int(img_id),
                             'category_id': int(clsi),
                             'bbox': list(dets[i][:4].astype(float)),
                             'score': dets[i][4].astype(float)}
                            for i in keep_index]
                self.det_boxes.extend(keep_box)
        return self.det_boxes

    def _nms(self, predicts, threshold):
        """Calculate NMS."""
        # convert xywh -> xmin ymin xmax ymax
        x1 = predicts[:, 0]
        y1 = predicts[:, 1]
        x2 = x1 + predicts[:, 2]
        y2 = y1 + predicts[:, 3]
        scores = predicts[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        reserved_boxes = []
        while order.size > 0:
            i = order[0]
            reserved_boxes.append(i)
            max_x1 = np.maximum(x1[i], x1[order[1:]])
            max_y1 = np.maximum(y1[i], y1[order[1:]])
            min_x2 = np.minimum(x2[i], x2[order[1:]])
            min_y2 = np.minimum(y2[i], y2[order[1:]])

            intersect_w = np.maximum(0.0, min_x2 - max_x1 + 1)
            intersect_h = np.maximum(0.0, min_y2 - max_y1 + 1)
            intersect_area = intersect_w * intersect_h
            ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area)

            indexes = np.where(ovr <= threshold)[0]
            order = order[indexes + 1]
        return reserved_boxes

    def _diou_nms(self, dets, thresh=0.5):
        """
        convert xywh -> xmin ymin xmax ymax
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            center_x1 = (x1[i] + x2[i]) / 2
            center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
            center_y1 = (y1[i] + y2[i]) / 2
            center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
            inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
            out_max_x = np.maximum(x2[i], x2[order[1:]])
            out_max_y = np.maximum(y2[i], y2[order[1:]])
            out_min_x = np.minimum(x1[i], x1[order[1:]])
            out_min_y = np.minimum(y1[i], y1[order[1:]])
            outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
            diou = ovr - inter_diag / outer_diag
            diou = np.clip(diou, -1, 1)
            inds = np.where(diou <= thresh)[0]
            order = order[inds + 1]
        return keep

    def write_result(self, result):
        """Save result to file."""
        import json
        t = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        try:
            self.file_path = self.save_prefix + '/predict' + t + '.json'
            f = open(self.file_path, 'w')
            json.dump(result, f)
        except IOError as e:
            raise RuntimeError("Unable to open json file to dump. What(): {}".format(str(e)))
        else:
            f.close()
            return self.file_path

    def get_eval_result(self):
        """Get eval result."""
        if not self.results:
            return 0.0, 0.0
        coco_gt = COCO(self.ann_file)
        coco_dt = coco_gt.loadRes(self.file_path)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        rdct = Redirct()
        stdout = sys.stdout
        sys.stdout = rdct
        coco_eval.summarize()
        res_map = coco_eval.stats[0]
        sys.stdout = stdout
        return rdct.content, float(res_map)

    def detect(self, outputs, batch, image_shape, image_id):
        """Detect boxes."""
        outputs_num = len(outputs)
        # output [|32, 52, 52, 3, 85| ]
        for batch_id in range(batch):
            for out_id in range(outputs_num):
                # 32, 52, 52, 3, 85
                out_item = outputs[out_id]
                # 52, 52, 3, 85
                out_item_single = out_item[batch_id, :]
                # get number of items in one head, [B, gx, gy, anchors, 5+80]
                dimensions = out_item_single.shape[:-1]
                out_num = 1
                for d in dimensions:
                    out_num *= d
                ori_w, ori_h = image_shape[batch_id]
                img_id = int(image_id[batch_id])
                x = out_item_single[..., 0] * ori_w
                y = out_item_single[..., 1] * ori_h
                w = out_item_single[..., 2] * ori_w
                h = out_item_single[..., 3] * ori_h

                conf = out_item_single[..., 4:5]
                cls_emb = out_item_single[..., 5:]

                cls_argmax = np.expand_dims(np.argmax(cls_emb, axis=-1), axis=-1)
                x = x.reshape(-1)
                y = y.reshape(-1)
                w = w.reshape(-1)
                h = h.reshape(-1)
                x_top_left = x - w / 2.
                y_top_left = y - h / 2.
                cls_emb = cls_emb.reshape(-1, self.num_classes)
                if not self.multi_label:
                    conf = conf.reshape(-1)
                    cls_argmax = cls_argmax.reshape(-1)

                    # create all False
                    flag = np.random.random(cls_emb.shape) > sys.maxsize
                    for i in range(flag.shape[0]):
                        c = cls_argmax[i]
                        flag[i, c] = True
                    confidence = cls_emb[flag] * conf
                    for x_lefti, y_lefti, wi, hi, confi, clsi in zip(x_top_left, y_top_left, w, h, confidence,
                                                                     cls_argmax):
                        if confi < self.eval_ignore_threshold:
                            continue
                        if img_id not in self.results:
                            self.results[img_id] = defaultdict(list)
                        x_lefti = max(0, x_lefti)
                        y_lefti = max(0, y_lefti)
                        wi = min(wi, ori_w)
                        hi = min(hi, ori_h)
                        # transform catId to match coco
                        coco_clsi = self.coco_catids[clsi]
                        self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])
                else:
                    conf = conf.reshape(-1, 1)
                    # create all False
                    confidence = cls_emb * conf
                    flag = cls_emb > self.multi_label_thresh
                    flag = flag.nonzero()
                    for index in range(len(flag[0])):
                        i = flag[0][index]
                        j = flag[1][index]
                        confi = confidence[i][j]
                        if confi < self.eval_ignore_threshold:
                            continue
                        if img_id not in self.results:
                            self.results[img_id] = defaultdict(list)
                        x_lefti = max(0, x_top_left[i])
                        y_lefti = max(0, y_top_left[i])
                        wi = min(w[i], ori_w)
                        hi = min(h[i], ori_h)
                        clsi = j
                        # transform catId to match coco
                        coco_clsi = self.coco_catids[clsi]
                        self.results[img_id][coco_clsi].append([x_lefti, y_lefti, wi, hi, confi])


def apply_eval_torch(eval_param_dict):
    network = eval_param_dict["net"]
    ds = eval_param_dict["dataset"]
    data_size = eval_param_dict["data_size"]
    args = eval_param_dict["args"]
    detection = DetectionEngine(args)
    for index, data in enumerate(ds.create_dict_iterator(num_epochs=1, output_numpy=True)):
        image = data["image"]
        image_shape_ = data["image_shape"]
        image_id_ = data["img_id"]
        image = torch.tensor(image, dtype=torch.float32).to(final_device)
        image_shape_ = torch.tensor(image_shape_, dtype=torch.float32).to(final_device)
        image_id_ = torch.tensor(image_id_, dtype=torch.float32).to(final_device)
        prediction = network(image)
        output_big, output_me, output_small = prediction
        output_big = output_big.detach().cpu().numpy()
        output_me = output_me.detach().cpu().numpy()
        output_small = output_small.detach().cpu().numpy()
        image_id_ = image_id_.cpu().numpy()
        image_shape_ = image_shape_.cpu().numpy()

        detection.detect([output_small, output_me, output_big], args.per_batch_size, image_shape_, image_id_)

    result = detection.do_nms_for_results()
    result_file_path = detection.write_result(result)
    eval_result, map = detection.get_eval_result()
    # view_result(args, result, score_threshold=None, recommend_threshold=config.recommend_threshold)
    return map


def apply_eval_ms(eval_param_dict):
    network = eval_param_dict["net"]
    ds = eval_param_dict["dataset"]
    data_size = eval_param_dict["data_size"]
    args = eval_param_dict["args"]
    detection = DetectionEngine(args)
    for index, data in enumerate(ds.create_dict_iterator(num_epochs=1)):
        image = data["image"]
        image_shape_ = data["image_shape"]
        image_id_ = data["img_id"]
        prediction = network(image)
        output_big, output_me, output_small = prediction
        output_big = output_big.asnumpy()
        output_me = output_me.asnumpy()
        output_small = output_small.asnumpy()
        image_id_ = image_id_.asnumpy()
        image_shape_ = image_shape_.asnumpy()

        detection.detect([output_small, output_me, output_big], args.per_batch_size, image_shape_, image_id_)

    result = detection.do_nms_for_results()
    result_file_path = detection.write_result(result)
    eval_result, map = detection.get_eval_result()
    # view_result(args, result, score_threshold=None, recommend_threshold=config.recommend_threshold)
    return map


def start_yolov4_train(model_ms, model_torch, yml_train_configs, train_logger):
    loss_name = yml_train_configs['loss_name']
    learning_rate = yml_train_configs['learning_rate']
    batch_size = yml_train_configs['batch_size']
    per_batch = batch_size * 100
    dataset_name = yml_train_configs['dataset_name']
    optimizer = yml_train_configs['optimizer']
    max_epoch = yml_train_configs['epoch']
    model_name = yml_train_configs['model_name']
    loss_truth, acc_truth, memory_truth = yml_train_configs['loss_ground_truth'], yml_train_configs['eval_ground_truth'], \
                                          yml_train_configs['memory_threshold']

    # 获取当前进程的内存使用情况
    process = psutil.Process(os.getpid())

    yolov4config.max_epoch = max_epoch
    cfg = yolov4config
    lr = learning_rate
    yolov4config.per_batch_size = batch_size

    loss_ms, loss_torch = get_loss(loss_name)
    loss_ms, loss_torch = loss_ms(), loss_torch()
    loss_torch = loss_torch.to(final_device)

    yolov4config.data_dir = yml_train_configs['dataset_path']
    yolov4config.train_img_dir = yolov4config.data_dir + r"/train2017"
    yolov4config.val_img_dir = yolov4config.data_dir + r"/val2017"
    yolov4config.ann_file = yolov4config.data_dir + "/annotations/instances_train2017.json"
    yolov4config.train_ann_file = yolov4config.data_dir + "/annotations/instances_train2017.json"
    yolov4config.val_ann_file = yolov4config.data_dir + "/annotations/instances_val2017.json"

    dataset = get_dataset(dataset_name)
    ds = dataset(yolov4config.data_dir, yolov4config.per_batch_size, True)
    ts = dataset(yolov4config.data_dir, yolov4config.per_batch_size, False)

    data_loader = ds.create_dict_iterator(output_numpy=True)
    testdata = ts.create_dict_iterator(output_numpy=True)

    data_size = len(ds)
    ts_size = len(ts)
    yolov4config.steps_per_epoch = int(data_size / yolov4config.per_batch_size / yolov4config.group_size)

    old_progress = -1
    optimizer_ms, optimizer_torch = get_optimizer(optimizer)
    torch_optimizer = optimizer_torch(model_torch.parameters(),
                                      momentum=cfg.momentum,
                                      weight_decay=cfg.weight_decay, lr=lr)

    modelms_trainable_params = model_ms.trainable_params()
    new_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in modelms_trainable_params:
        modelms_trainable_param.name =  model_name + str(
            layer_nums) + "_" + modelms_trainable_param.name
        new_trainable_params.append(modelms_trainable_param)
        layer_nums += 1

    ms_optimizer = optimizer_ms(params=new_trainable_params,
                                learning_rate=lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay, )

    def forward_fn(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,
                   batch_gt_box2, input_shape):
        yolo_output = model_ms(images)
        loss = loss_ms(yolo_output, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1,batch_gt_box2, input_shape)
        return loss

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, ms_optimizer.parameters, has_aux=False)

    def train_step(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0, batch_gt_box1, batch_gt_box2,
                   input_shape):
        (loss), grads = grad_fn(images, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0,
                                batch_gt_box1,
                                batch_gt_box2, input_shape)
        loss = mindspore.ops.depend(loss, ms_optimizer(grads))
        return loss

    losses_ms_avg, losses_torch_avg = [], []
    ms_memorys_avg, torch_memorys_avg = [], []
    ms_times_avg, torch_times_avg = [], []
    eval_ms, eval_torch = [], []

    for epoch_idx in range(yolov4config.max_epoch):
        train_logger.info('----------------------------')
        train_logger.info(f"epoch: {epoch_idx}/{max_epoch}")

        model_torch.train()
        model_ms.set_train(True)
        model_ms = settrain(model_ms)
        model_torch = settrain(model_torch)
        nums = 0
        losses_torch, losses_ms = [], []
        ms_memorys, torch_memorys = [], []
        ms_times, torch_times = [], []

        for i, data in enumerate(data_loader):
            nums += data['image'].shape[0]

            images = data["image"]
            input_shape = images.shape[2:4]
            images_ms = Tensor.from_numpy(images)
            batch_y_true_0 = Tensor.from_numpy(data['bbox1'])
            batch_y_true_1 = Tensor.from_numpy(data['bbox2'])
            batch_y_true_2 = Tensor.from_numpy(data['bbox3'])
            batch_gt_box0 = Tensor.from_numpy(data['gt_box1'])
            batch_gt_box1 = Tensor.from_numpy(data['gt_box2'])
            batch_gt_box2 = Tensor.from_numpy(data['gt_box3'])
            input_shape_ms = Tensor(tuple(input_shape[::-1]), ms.float32)
            images_t = torch.tensor(images).to(final_device)

            y_true_0 = torch.tensor(data['bbox1'], dtype=torch.float32).to(final_device)
            y_true_1 = torch.tensor(data['bbox2'], dtype=torch.float32).to(final_device)
            y_true_2 = torch.tensor(data['bbox3'], dtype=torch.float32).to(final_device)
            gt_0 = torch.tensor(data['gt_box1'], dtype=torch.float32).to(final_device)
            gt_1 = torch.tensor(data['gt_box2'], dtype=torch.float32).to(final_device)
            gt_2 = torch.tensor(data['gt_box3'], dtype=torch.float32).to(final_device)
            input_shape_t = torch.tensor(tuple(input_shape[::-1]), dtype=torch.float32).to(final_device)

            memory_info = process.memory_info()
            torch_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            torch_time_start = time.time()
            yolo_out = model_torch(images_t)
            loss_torch_result = loss_torch(yolo_out, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2, input_shape_t)

            loss_torch_result.backward()
            torch_optimizer.step()
            torch_time_end = time.time()
            torch_time_train = torch_time_end - torch_time_start
            memory_info = process.memory_info()
            torch_memory_train_end = memory_info.rss / 1024 / 1024
            torch_memory_train = torch_memory_train_end - torch_memory_train_start
            torch_optimizer.zero_grad()

            memory_info = process.memory_info()
            ms_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            ms_time_start = time.time()
            loss_ms_result = train_step(images_ms, batch_y_true_0, batch_y_true_1, batch_y_true_2, batch_gt_box0,
                                        batch_gt_box1, batch_gt_box2, input_shape_ms)
            ms_times_end = time.time()
            ms_time_train = ms_times_end - ms_time_start
            memory_info = process.memory_info()
            ms_memory_train = memory_info.rss / 1024 / 1024 - ms_memory_train_start

            if nums % per_batch == 0:
                train_logger.info(f"batch: {nums}, torch_loss: {loss_torch_result.item()}, ms_loss: {loss_ms_result.asnumpy()}, torch_memory: {torch_memory_train}MB, ms_memory:  {ms_memory_train}MB, torch_time: {torch_time_train}, ms_time:  {ms_time_train}")
                if nums == 2000:
                    break

            losses_torch.append(loss_torch_result.item())
            losses_ms.append(loss_ms_result.asnumpy())
            ms_memorys.append(ms_memory_train)
            torch_memorys.append(torch_memory_train)
            ms_times.append(ms_time_train)
            torch_times.append(torch_time_train)

        losses_ms_avg.append(np.mean(losses_ms))
        losses_torch_avg.append(np.mean(losses_torch))
        ms_memorys_avg.append(np.mean(ms_memorys))
        torch_memorys_avg.append(np.mean(torch_memorys))
        ms_times_avg.append(np.mean(ms_times))
        torch_times_avg.append(np.mean(torch_times))

        train_logger.info("epoch {}: ".format(epoch_idx) + "torch_loss: " + str(np.mean(losses_torch)) + " ms_loss: " + str(np.mean(losses_ms)))

        #start eval
        model_ms = seteval(model_ms)
        model_torch = seteval(model_torch)
        model_ms.set_train(False)
        model_torch.eval()

        yolov4config.outputs_dir = "./log/"

        detection1 = DetectionEngine(yolov4config)
        detection2 = DetectionEngine(yolov4config)

        for i, data in enumerate(testdata):
            image = data["image"]
            image_shape = data["image_shape"]
            image_id = data["img_id"]

            image_ms = mindspore.Tensor(image)
            image_id_ms = mindspore.Tensor(image_id)
            image_shape_ms = mindspore.Tensor(image_shape)

            image_t = torch.Tensor(image).to(final_device)
            image_id_t = torch.Tensor(image_id).to(final_device)
            image_shape_t = torch.Tensor(image_shape).to(final_device)

            output_big, output_me, output_small = model_ms(image_ms)
            output_big_t, output_me_t, output_small_t = model_torch(image_t)

            output_big = output_big.asnumpy()
            output_me = output_me.asnumpy()
            output_small = output_small.asnumpy()
            output_big_t = output_big_t.detach().cpu().numpy()
            output_me_t = output_me_t.detach().cpu().numpy()
            output_small_t = output_small_t.detach().cpu().numpy()

            image_id_ms = image_id_ms.asnumpy()
            image_shape_ms = image_shape_ms.asnumpy()
            image_id_t = image_id_t.cpu().numpy()
            image_shape_t = image_shape_t.cpu().numpy()

            detection1.detect([output_small, output_me, output_big], yolov4config.per_batch_size, image_shape_ms,
                              image_id_ms)
            detection2.detect([output_big_t, output_me_t, output_small_t], yolov4config.per_batch_size, image_shape_t,image_id_t)



        result1 = detection1.do_nms_for_results()
        result2 = detection2.do_nms_for_results()
        result_file_path1 = detection1.write_result(result1)
        result_file_path2 = detection2.write_result(result2)

        yolov4config.recommend_threshold = False
        eval_param_dict1 = {"net": model_torch, "dataset": ts, "data_size": ts_size,
                            "anno_json": yolov4config.ann_val_file,
                            "args": yolov4config}
        eval_param_dict2 = {"net": model_ms, "dataset": ts, "data_size": ts_size,
                            "anno_json": yolov4config.ann_val_file,
                            "args": yolov4config}


        eval_result1 = apply_eval_torch(eval_param_dict1)
        eval_result2 = apply_eval_ms(eval_param_dict2)

        train_logger.info(f"MindSpore_map: {eval_result2}" + " PyTorch_map: {}".format(eval_result1))
        eval_torch.append(eval_result1)
        eval_ms.append(eval_result2)

    train_logger.generation = yml_train_configs['generation']
    analyze_util = train_result_analyze(model_name=model_name, epochs=max_epoch, loss_ms=losses_ms_avg,
                                        loss_torch=losses_torch_avg, eval_ms=eval_ms, eval_torch=eval_torch,
                                        memories_ms=ms_memorys, memories_torch=torch_memorys, loss_truth=loss_truth,
                                        acc_truth=acc_truth, memory_truth=memory_truth, train_logger=train_logger)
    analyze_util.analyze_main()