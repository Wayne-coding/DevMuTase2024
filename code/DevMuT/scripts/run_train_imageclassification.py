import sys

sys.path.append(".")
# sys.path.append("../")
import argparse
import os
from copy import deepcopy
import numpy as np
import psutil
import torch
import mindspore
from mindspore.common import dtype as mstype
from common.dataset_utils import get_dataset
from common.loss_utils import get_loss
from common.opt_utils import get_optimizer
from common.analyzelog_util import train_result_analyze
import time
from common.log_recoder import Logger
from common.model_utils import get_model
from scripts.run_textcnn import *


def start_imageclassification_train(model_ms, model_torch, train_configs, train_logger):
    loss_name = train_configs['loss_name']
    learning_rate = train_configs['learning_rate']
    batch_size = train_configs['batch_size']
    per_batch = batch_size * 100
    dataset_name = train_configs['dataset_name']
    optimizer = train_configs['optimizer']
    epochs = train_configs['epoch']
    model_name = train_configs['model_name']
    loss_truth, acc_truth, memory_truth = train_configs['loss_ground_truth'], train_configs['eval_ground_truth'], \
        train_configs['memory_threshold']

    print(train_configs)

    process = psutil.Process()

    if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
        final_device = 'cuda:0'
    else:
        final_device = 'cpu'

    # 损失函数
    loss_fun_ms, loss_fun_torch = get_loss(loss_name)
    loss_fun_ms, loss_fun_torch = loss_fun_ms(), loss_fun_torch()
    loss_fun_torch = loss_fun_torch.to(final_device)

    # 优化器
    optimizer_ms, optimizer_torch = get_optimizer(optimizer)
    optimizer_torch = optimizer_torch(model_torch.parameters(), lr=learning_rate)

    modelms_trainable_params = model_ms.trainable_params()
    new_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in modelms_trainable_params:
        modelms_trainable_param.name = model_ms.__class__.__name__ + str(
            layer_nums) + "_" + modelms_trainable_param.name
        new_trainable_params.append(modelms_trainable_param)
        layer_nums += 1
    optimizer_ms = optimizer_ms(params=new_trainable_params, learning_rate=learning_rate, momentum=0.9,
                                weight_decay=0.0001)
    # 对参数名称重命名，创建一个新的优化器

    dataset = get_dataset(dataset_name)
    train_set = dataset(data_dir=train_configs['dataset_path'], batch_size=batch_size, is_train=True)
    test_set = dataset(data_dir=train_configs['dataset_path'], batch_size=batch_size, is_train=False)

    train_iter = train_set.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    test_iter = test_set.create_dict_iterator(output_numpy=True, num_epochs=epochs)

    def forward_fn(data, label):
        logits = model_ms(data)
        loss = loss_fun_ms(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, optimizer_ms.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = mindspore.ops.depend(loss, optimizer_ms(grads))
        return loss

    losses_ms_avg, losses_torch_avg = [], []
    ms_memorys_avg, torch_memorys_avg = [], []
    ms_times_avg, torch_times_avg = [], []
    eval_ms, eval_torch = [], []
    for epoch in range(epochs):
        train_logger.info('----------------------------')
        train_logger.info(f"epoch: {epoch}/{epochs}")

        # 训练步骤开始
        model_torch.train()
        model_ms.set_train(True)

        losses_torch, losses_ms = [], []
        ms_memorys, torch_memorys = [], []
        ms_times, torch_times = [], []

        batch = 0
        nums = 0
        for item in train_iter:
            nums += item['image'].shape[0]
            imgs_array, targets_array = deepcopy(item['image']), deepcopy(item['label'])
            imgs_torch, targets_torch = torch.tensor(imgs_array, dtype=torch.float32).to(final_device), torch.tensor(
                targets_array, dtype=torch.long).to(final_device)
            imgs_ms, targets_ms = mindspore.Tensor(imgs_array, mstype.float32), mindspore.Tensor(targets_array,
                                                                                                 mstype.int32)

            memory_info = process.memory_info()
            torch_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            torch_time_start = time.time()

            outputs_torch_tensor = model_torch(imgs_torch)
            loss_torch = loss_fun_torch(outputs_torch_tensor, targets_torch)
            loss_torch.backward()
            optimizer_torch.step()

            torch_time_end = time.time()
            torch_time_train = torch_time_end - torch_time_start

            memory_info = process.memory_info()
            torch_memory_train_end = memory_info.rss / 1024 / 1024
            torch_memory_train = torch_memory_train_end - torch_memory_train_start
            optimizer_torch.zero_grad()

            memory_info = process.memory_info()
            ms_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            ms_time_start = time.time()

            loss_ms = train_step(imgs_ms, targets_ms)




            # add jax train
            # jax_out_put  = outputs_torch_tensor.detach().cpu().numpy()
            # jax loss cal
            # jax gredi


            ms_time_end = time.time()

            ms_time_train = ms_time_end - ms_time_start
            memory_info = process.memory_info()
            ms_memory_train = memory_info.rss / 1024 / 1024 - ms_memory_train_start

            if batch % per_batch == 0:
                # folder_path = '/data1/ypr/net-sv/output_model/resnet50'
                # os.makedirs(folder_path, exist_ok=True)
                # os.makedirs(folder_path+'/pytorch_model', exist_ok=True)
                # os.makedirs(folder_path+'/mindspore_model', exist_ok=True)
                # torch.save(model_torch.state_dict(),
                #            folder_path + '/pytorch_model/pytorch_model_' + str(epoch) + '_' + str(
                #                batch // per_batch) + '.pth')
                # mindspore.save_checkpoint(model_ms,
                #                           folder_path + '/mindspore_model/mindspore_model' + str(epoch) + '_' + str(
                #                               batch // per_batch) + '.ckpt')
                train_logger.info(
                    f"batch: {batch}, torch_loss: {loss_torch.item()}, ms_loss: {loss_ms.asnumpy()}, torch_memory: {torch_memory_train}MB, ms_memory:  {ms_memory_train}MB, torch_time: {torch_time_train}, ms_time:  {ms_time_train}")

                if batch == 5000:
                    break

            losses_torch.append(loss_torch.item())
            losses_ms.append(loss_ms.asnumpy())
            ms_memorys.append(ms_memory_train)
            torch_memorys.append(torch_memory_train)
            ms_times.append(ms_time_train)
            torch_times.append(torch_time_train)
            batch += batch_size

        losses_ms_avg.append(np.mean(losses_ms))
        losses_torch_avg.append(np.mean(losses_torch))
        ms_memorys_avg.append(np.mean(ms_memorys))
        torch_memorys_avg.append(np.mean(torch_memorys))
        ms_times_avg.append(np.mean(ms_times))
        torch_times_avg.append(np.mean(torch_times))

        train_logger.info(
            f"epoch: {epoch}, torch_loss_avg: {np.mean(losses_torch)}, ms_loss_avg: {np.mean(losses_ms)}, torch_memory_avg: {np.mean(torch_memory_train)}MB, ms_memory_avg:  {np.mean(ms_memory_train)}MB, torch_time_avg: {np.mean(torch_time_train)}, ms_time_avg:  {np.mean(ms_time_train)}")

        # 测试步骤开始
        model_torch.eval()
        model_ms.set_train(False)
        test_data_size = 0
        total_accuracy = 0
        correct_ms = 0

        with torch.no_grad():
            for item in test_iter:
                nums += item['image'].shape[0]

                imgs_array, targets_array = deepcopy(item['image']), deepcopy(item['label'])
                imgs_torch, targets_torch = torch.tensor(imgs_array), torch.tensor(targets_array, dtype=torch.long)
                imgs_ms, targets_ms = mindspore.Tensor(imgs_array, mstype.float32), mindspore.Tensor(targets_array,
                                                                                                     mstype.int32)

                test_data_size += len(imgs_ms)

                imgs_torch = imgs_torch.to(final_device)
                targets_torch = targets_torch.to(final_device)
                outputs_torch_tensor = model_torch(imgs_torch)
                outputs_torch_array = outputs_torch_tensor.cpu().numpy()
                targets_torch_array = targets_torch.cpu().numpy()

                accuracy = (outputs_torch_array.argmax(1) == targets_torch_array).sum()
                total_accuracy = total_accuracy + accuracy

                pred_ms = model_ms(imgs_ms)
                correct_ms += (pred_ms.argmax(1) == targets_ms).asnumpy().sum()

        correct_ms /= test_data_size

        train_logger.info(f"Mindspore Test Accuracy: {(100 * correct_ms)}%" + " Pytorch Test Accuracy: {}%".format(
            100 * total_accuracy / test_data_size))

        eval_torch.append(total_accuracy / test_data_size)
        eval_ms.append(correct_ms)

    train_logger.generation = train_configs['generation']
    analyze_util = train_result_analyze(model_name=model_name, epochs=epochs, loss_ms=losses_ms_avg,
                                        loss_torch=losses_torch_avg, eval_ms=eval_ms, eval_torch=eval_torch,
                                        memories_ms=ms_memorys, memories_torch=torch_memorys, loss_truth=loss_truth,
                                        acc_truth=acc_truth, memory_truth=memory_truth, train_logger=train_logger)
    analyze_util.analyze_main()


if __name__ == '__main__':
    """
    export CONTEXT_DEVICE_TARGET=GPU
    export CUDA_VISIBLE_DEVICES=2,3
    """

    """
        resnet50 /data1/pzy/raw/cifar10
        textcnn r"/data1/pzy/mindb/rt-polarity"
        yolov4 /data1/pzy/raw/coco2017
        unet r"/data1/pzy/raw/ischanllge"
        ssimae /data1/pzy/raw/MVTecAD/
    """

    train_configs = {
        "model_name": "mobilenetv2",
        'dataset_name': "cifar10",
        'batch_size': 5,
        'input_size': (2, 3, 224, 224),
        'test_size': 2,
        'dtypes': ['float'],
        'epoch': 30,
        'loss_name': "CrossEntropy",
        'optimizer': "SGD",
        'learning_rate': 0.02,
        'loss_ground_truth': 2.950969386100769,
        'eval_ground_truth': 0.998740881321355,
        'memory_threshold': 0.01,
        'device_target': 'GPU',
        'device_id': 0,
        "dataset_path": "/data1/pzy/raw/cifar10"

    }

    data = [np.ones(train_configs["input_size"])]
    ms_dtypes = [mindspore.float32]
    torch_dtypes = [torch.float32]
    model_ms_origin, model_torch_origin = get_model(train_configs["model_name"], train_configs["input_size"],
                                                    only_ms=False, scaned=True)

    log_path = '/data1/myz/empirical_exp/common/log/E3/rq2result/patchcore-2023.9.20.8.43.32/mutation.txt'

    model_name = train_configs['model_name']
    logger = Logger(log_file='./log/debug.log')
    args_opt = argparse.Namespace(
        model=model_name,
        dataset_path=r'/data1/pzy/mindb/ssd/datamind/ssd.mindrecord0',
        batch_size=5,
        epoch=200,
        mutation_iterations=5,
        selected_model_num=1,
        mutation_type=["WS", "NS", "NAI", "NEB", "GF"],
        # "LA","LD","WS","NS","NAI","NEB","GF","LC","LS","RA","WS", "NS", "NAI", "NEB", "GF"
        mutation_log='/data1/myz/netsv/common/log',
        selected_generation=None,
        mutation_strategy="random"
    )
    model_ms, model_torch, train_configs = model_ms_origin, model_torch_origin, train_configs

    start_imageclassification_train(model_ms, model_torch, train_configs, logger.logger)