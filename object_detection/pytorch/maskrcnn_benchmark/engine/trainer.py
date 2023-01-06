# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, is_main_process
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

# Instead of zeroing, set parameter grads to None
# Prevents extraneous copy as we're not accumulating
def set_grads_to_none(model):
    for param in model.parameters():
        param.grad = None


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    args,
    per_iter_start_callback_fn=None,
    per_iter_end_callback_fn=None,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = 0
    model.train()
    start_training_time = time.time()
    end = time.time()

    if args.channels_last and args.device != "xpu":
        model = model.to(memory_format=torch.channels_last)
        print("---- use NHWC format")

    total_time = 0.0
    total_count = 0
    profile_len = min(max_iter, args.num_iter) // 2
    datatype = torch.float16 if args.precision == "float16" else torch.bfloat16 if args.precision == "bfloat16" else torch.float
    if args.profile and args.device == "xpu":
        for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
            scheduler.step()

            start_time = time.time()
            images = images.to(device)
            targets = [target.to(device) for target in targets]

            with torch.xpu.amp.autocast(enabled=True if args.precision != "float32" else False, dtype=datatype):
                loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            losses.backward()

            optimizer.step()
            optimizer.zero_grad()

            torch.xpu.synchronize()

            duration = time.time() - start_time
            print("iteration:{}, training time: {} sec.".format(iteration, duration))
            if iteration >= args.num_warmup:
                total_time += duration
                total_count += 1
            if iteration == profile_len:
                import pathlib
                timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                if not os.path.exists(timeline_dir):
                    try:
                        os.makedirs(timeline_dir)
                    except:
                        pass
                torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                    timeline_dir+'profile.pt')
                torch.save(prof.key_averages(group_by_input_shape=True).table(),
                    timeline_dir+'profile_detail.pt')
                torch.save(prof.table(sort_by="id", row_limit=100000),
                    timeline_dir+'profile_detail_withId.pt')
                prof.export_chrome_trace(timeline_dir+"trace.json")
    elif args.profile and args.device == "cuda":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=profile_len,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
                scheduler.step()

                if args.channels_last and args.device != "xpu":
                    images.tensors = images.tensors.to(memory_format=torch.channels_last)
                start_time = time.time()
                images = images.to(device)
                targets = [target.to(device) for target in targets]

                with torch.cuda.amp.autocast(enabled=True if args.precision != "float32" else False, dtype=datatype):
                    loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss=losses_reduced, **loss_dict_reduced)

                losses.backward()

                optimizer.step()
                optimizer.zero_grad()

                torch.cuda.synchronize()
                duration = time.time() - start_time
                p.step()
                print("iteration:{}, training time: {} sec.".format(iteration, duration))
                if iteration >= args.num_warmup:
                    total_time += duration
                    total_count += 1
    elif args.profile and args.device == "cpu":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=profile_len,
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
                scheduler.step()

                if args.channels_last and args.device != "xpu":
                    images.tensors = images.tensors.to(memory_format=torch.channels_last)
                start_time = time.time()
                images = images.to(device)
                targets = [target.to(device) for target in targets]

                with torch.cpu.amp.autocast(enabled=True if args.precision != "float32" else False, dtype=datatype):
                    loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss=losses_reduced, **loss_dict_reduced)

                losses.backward()

                optimizer.step()
                optimizer.zero_grad()

                duration = time.time() - start_time
                p.step()
                print("iteration:{}, training time: {} sec.".format(iteration, duration))
                if iteration >= args.num_warmup:
                    total_time += duration
                    total_count += 1
    else:
        for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
            scheduler.step()

            if args.channels_last and args.device != "xpu":
                images.tensors = images.tensors.to(memory_format=torch.channels_last)
            start_time = time.time()
            images = images.to(device)
            targets = [target.to(device) for target in targets]

            if args.device == "xpu":
                with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
                    loss_dict = model(images, targets)
            elif args.device == "cuda":
                with torch.cuda.amp.autocast(enabled=True, dtype=datatype):
                    loss_dict = model(images, targets)
            else:
                with torch.cpu.amp.autocast(enabled=True, dtype=datatype):
                    loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            losses.backward()

            optimizer.step()
            optimizer.zero_grad()

            if args.device == "xpu":
                torch.xpu.synchronize()
            elif args.device == "cuda":
                torch.cuda.synchronize()
            duration = time.time() - start_time
            print("iteration:{}, training time: {} sec.".format(iteration, duration))
            if iteration >= args.num_warmup:
                total_time += duration
                total_count += 1

    batch_size = args.batch_size
    avg_time = total_time / total_count
    latency = avg_time / batch_size * 1000
    perf = batch_size / avg_time
    print("total time:{}, total count:{}".format(total_time, total_count))
    print('%d epoch training latency: %6.2f ms'%(0, latency))
    print('%d epoch training Throughput: %6.2f fps'%(0, perf))

    return None
