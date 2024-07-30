import sys

sys.path.append(".")
import os
import numpy as np
import psutil
import torch
import mindspore
import argparse
from mindspore.common import dtype as mstype
from common.dataset_utils import get_dataset
from common.loss_utils import get_loss
from common.analyzelog_util import train_result_analyze
from common.opt_utils import get_optimizer
import time
from common.log_recoder import Logger
from common.model_utils import get_model


def start_textcnn_train(model_ms, model_torch, train_configs, train_logger):
    loss_name = train_configs['loss_name']
    learning_rate = train_configs['learning_rate']
    batch_size = train_configs['batch_size']
    per_batch = batch_size * 100
    dataset_name = train_configs['dataset_name']
    optimizer = train_configs['optimizer']
    epoch_num = train_configs['epoch']
    model_name = train_configs['model_name']
    loss_truth, acc_truth, memory_truth = train_configs['loss_ground_truth'], train_configs['eval_ground_truth'], \
        train_configs['memory_threshold']
    num_classes = 2

    process = psutil.Process()

    if "CONTEXT_DEVICE_TARGET" in os.environ and os.environ['CONTEXT_DEVICE_TARGET'] == 'GPU':
        final_device = 'cuda:0'
    else:
        final_device = 'cpu'

    loss_ms, loss_t = get_loss(loss_name)
    loss_ms, loss_t = loss_ms(), loss_t()
    loss_t = loss_t.to(final_device)

    dataset = get_dataset(dataset_name)
    train_dataset = dataset(data_dir=train_configs['dataset_path'], batch_size=batch_size,
                            is_train=True)
    test_dataset = dataset(data_dir=train_configs['dataset_path'], batch_size=batch_size,
                           is_train=False)
    train_iter = train_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)
    test_iter = test_dataset.create_dict_iterator(output_numpy=False, num_epochs=epoch_num)

    opt_ms, opt_t = get_optimizer(optimizer)
    opt_t = opt_t(filter(lambda x: x.requires_grad, model_torch.parameters()), lr=learning_rate,
                  weight_decay=float(3e-5))

    modelms_trainable_params = model_ms.trainable_params()
    new_trainable_params = []
    layer_nums = 0
    for modelms_trainable_param in modelms_trainable_params:
        modelms_trainable_param.name = train_configs['model_name'] + str(
            layer_nums) + "_" + modelms_trainable_param.name
        new_trainable_params.append(modelms_trainable_param)
        layer_nums += 1

    opt_ms = opt_ms(filter(lambda x: x.requires_grad, model_ms.get_parameters()), learning_rate=learning_rate,
                    weight_decay=float(3e-5))

    def forward_fn(data, label, num_classes):
        outputs = model_ms(data)
        loss = loss_ms(outputs, label, num_classes)
        return loss

    grad_fn = mindspore.ops.value_and_grad(forward_fn, None, opt_ms.parameters, has_aux=False)

    def train_step(data, label, num_classes):
        (loss), grads = grad_fn(data, label, num_classes)
        loss = mindspore.ops.depend(loss, opt_ms(grads))
        return loss

    losses_ms_avg, losses_torch_avg = [], []
    ms_memorys_avg, torch_memorys_avg = [], []
    ms_times_avg, torch_times_avg = [], []
    eval_ms, eval_torch = [], []
    for epoch in range(epoch_num):
        train_logger.info('----------------------------')
        train_logger.info(f"epoch: {epoch}/{epoch_num}")
        model_torch.train()
        model_ms.set_train(True)

        losses_torch, losses_ms = [], []
        ms_memorys, torch_memorys = [], []
        ms_times, torch_times = [], []

        batch = 0
        nums = 0

        for item in train_iter:
            nums += item['data'].shape[0]
            text_array, targets_array = item['data'].asnumpy(), item['label'].asnumpy()
            text_tensor, targets_tensor = mindspore.Tensor(text_array, dtype=mstype.int32), mindspore.Tensor(
                targets_array, dtype=mstype.int32)

            memory_info = process.memory_info()
            torch_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            torch_time_start = time.time()
            output_torch = model_torch(torch.LongTensor(text_array).to(final_device))
            loss_t_result = loss_t(output_torch, torch.LongTensor(targets_array).to(final_device), num_classes)
            loss_t_result.backward()
            opt_t.step()
            torch_time_end = time.time()
            torch_time_train = torch_time_end - torch_time_start

            memory_info = process.memory_info()
            torch_memory_train_end = memory_info.rss / 1024 / 1024
            torch_memory_train = torch_memory_train_end - torch_memory_train_start

            opt_t.zero_grad()

            memory_info = process.memory_info()
            ms_memory_train_start = memory_info.rss / 1024 / 1024 / 1024
            ms_time_start = time.time()
            loss_ms_result = train_step(text_tensor, targets_tensor, num_classes)
            ms_time_end = time.time()

            ms_time_train = ms_time_end - ms_time_start
            memory_info = process.memory_info()
            ms_memory_train = memory_info.rss / 1024 / 1024 - ms_memory_train_start

            if batch % per_batch == 0:
                folder_path = '/data1/ypr/net-sv/output_model/textcnn'
                os.makedirs(folder_path, exist_ok=True)
                os.makedirs(folder_path + '/pytorch_model', exist_ok=True)
                os.makedirs(folder_path + '/mindspore_model', exist_ok=True)
                torch.save(model_torch.state_dict(),
                           folder_path + '/pytorch_model/pytorch_model_' + str(epoch) + '_' + str(
                               batch // per_batch) + '.pth')
                mindspore.save_checkpoint(model_ms,
                                          folder_path + '/mindspore_model/mindspore_model' + str(epoch) + '_' + str(
                                              batch // per_batch) + '.ckpt')
                train_logger.info(
                    f"batch: {batch}, torch_loss: {loss_t_result.item()}, ms_loss: {loss_ms_result.asnumpy()}, torch_memory: {torch_memory_train}MB, ms_memory:  {ms_memory_train}MB, torch_time: {torch_time_train}, ms_time:  {ms_time_train}")
                if batch == 5000:
                    break

            losses_torch.append(loss_t_result.item())
            losses_ms.append(loss_ms_result.asnumpy())
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
        correct_torch = 0
        correct_ms = 0

        with torch.no_grad():
            for item in test_iter:
                text, targets = item['data'], item['label']
                test_data_size += text.shape[0]
                text_array, targets_array = text.asnumpy(), targets.asnumpy()

                text_tensor, targets_tensor = torch.LongTensor(text_array).to(final_device), torch.LongTensor(
                    targets_array).to(final_device)

                output_torch = model_torch(text_tensor)
                output_ms = model_ms(text)
                indices_ms = np.argmax(output_ms.asnumpy(), axis=1)
                result_ms = (np.equal(indices_ms, targets.asnumpy()) * 1).reshape(-1)
                accuracy_ms = result_ms.sum()
                correct_ms = correct_ms + accuracy_ms

                indices = torch.argmax(output_torch.to(final_device), dim=1)
                result = (np.equal(indices.detach().cpu().numpy(), targets_tensor.detach().cpu().numpy()) * 1).reshape(
                    -1)
                accuracy = result.sum()
                correct_torch = correct_torch + accuracy
        train_logger.info(
            f"Mindspore Test Accuracy: {(correct_ms / test_data_size)}%" + " Pytorch Test Accuracy: {}%".format(
                correct_torch / test_data_size))

        eval_torch.append(correct_torch / test_data_size)
        eval_ms.append(correct_ms / test_data_size)

    train_logger.generation = train_configs['generation']
    analyze_util = train_result_analyze(model_name=model_name, epochs=epoch_num, loss_ms=losses_ms_avg,
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
        "model_name": "textcnn",
        'dataset_name': "rtpolarity",
        'batch_size': 32,
        'input_size': (2, 51),
        'test_size': 2,
        'dtypes': ['int'],
        'epoch': 30,
        'loss_name': "textcnnloss",
        'optimizer': "SGD",
        'learning_rate': 0.02,
        'loss_ground_truth': 2.950969386100769,
        'eval_ground_truth': 0.998740881321355,
        'memory_threshold': 0.01,
        'device_target': 'GPU',
        'device_id': 0,
        "dataset_path": "/data1/pzy/mindb/rt-polarity"

    }

    data = [np.ones(train_configs["input_size"])]
    ms_dtypes = [mindspore.int32]
    torch_dtypes = [torch.int32]
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
        mutation_type=["LA","LD","WS","NS","NAI"],
        # "LA","LD","WS","NS","NAI","NEB","GF","LC","LS","RA",
        mutation_log='/data1/myz/netsv/common/log',
        selected_generation=None,
        mutation_strategy="random"
    )
    model_ms, model_torch, train_configs = model_ms_origin, model_torch_origin, train_configs

    start_textcnn_train(model_ms, model_torch, train_configs, logger.logger)
