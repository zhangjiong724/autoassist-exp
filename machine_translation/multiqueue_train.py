#!/usr/bin/env python3 -u

import os, time, math
import random
import signal
import torch
import collections
from fairseq.data.assistant import AssistantSamplerParallel

#from train import main as single_process_main
import numpy as np
from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from train import load_dataset_splits, get_training_stats
import queue
from queue import Empty,Full
import multiprocessing
from train import get_valid_stats, get_perplexity



def BQ_feeder(rank, args, boss_queue, assistant_queue, iter_stats):
    """Assistant process
    """
    if args.max_tokens is None:
        args.max_tokens = 6000
    # Setup task 
    task = tasks.setup_task(args)
    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])
    
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
    )
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)
    # Initialize dataloader
    dataset= task.dataset(args.train_subset)
    
    # compute IDF
    idf_src, idf_tgt = None, None
    if args.use_tfidf:
        # construct the tfidf_dataset
        num_doc = len(dataset)
        tfidf_dataset = utils.build_tf_idf(dataset, task.source_dictionary, task.target_dictionary)
    else:
        tfidf_dataset = None


    # initialize Assistant
    assistant = AssistantSamplerParallel( task.source_dictionary, task.target_dictionary, base_prob = 0.5, num_proc = args.distributed_world_size, proc_id = rank, num_bins_per_proc = 24, tfidf_feature = tfidf_dataset)
    
    epoch_itr = task.get_batch_iterator(
        dataset= dataset,
        assistant = assistant,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=1,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=rank,
        batch_method=args.batch_method,
    )

    update_freq = args.update_freq[-1]
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus, shuffle=True)
    itr = iterators.GroupedIterator(itr, update_freq)

    assistant_log_interval = 25

    print("| Assistant%d: Iterator length = %d, verbose inverval %d"%(rank, len(itr), assistant_log_interval))
    
    assistant_train_counter = 0
    iter_stats.put((len(itr), epoch_itr.epoch, epoch_itr.iterations_in_epoch))


    while True:
        for i, samples in enumerate(itr):
            boss_queue.put((i, samples))
            if assistant_queue.qsize() > 1 and boss_queue.qsize() > 10:
                try:
                    idcs, losses = assistant_queue.get(block=False)
                    assistant.train_tfidf_step(idcs, losses, n_steps = min(math.ceil(len(idcs)/4), 4))
                    # assistant log
                    assistant_train_counter += 1
                    if assistant_train_counter%assistant_log_interval==0 and rank==0:
                        print("Assistant batch step %d, accuracy %f, confident %f, loss_mean %f, real_pos %f, pred_pos %f"%
	(assistant_train_counter, assistant.sec_loss, assistant.confident, assistant.mean_loss(), assistant.real_pos, assistant.pred_pos), flush=True)
                except queue.Empty:
                    print('WARNING Empty-Queue: BQ%d failed at %i-th attempt to train Assistant'%(rank, i))
                    pass
        # Initialize data iterator
        itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus, shuffle=True)
        itr = iterators.GroupedIterator(itr, update_freq)

def single_process_main(args, boss_queue, assistant_queue, iter_stats):
    print(args)



    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    load_dataset_splits(task, ['train', 'valid'])

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    if args.spl or args.assistant:
        assert "noreduce" in args.criterion, "SPL and AutoAssist need noreduce criterion"
        #args.criterion += "_noreduce"
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))


    # Build trainer
    trainer = Trainer(args, task, model, criterion, dummy_batch=None)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')

    
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    epoch = 0
    # Wait for the Assistants to init
    while boss_queue.qsize() < 20:
        time.sleep(2)
    
    iter_len, epoch_itr_epoch, epoch_itr_iterations_in_epoch = iter_stats.get(block=True)
    



    while epoch < max_epoch and trainer.get_num_updates() < max_update:
        epoch += 1 
        print("============ Starting epoch %d ==========="%(epoch))
        progress = progress_bar.build_progress_bar(
            args, range(iter_len), epoch, no_progress_bar='simple', assistant = None,
        )

        num_batches_seen = 0
        for ii, objs in enumerate(progress, start=epoch_itr_iterations_in_epoch):
            while boss_queue.qsize() < 1:
                print("Empty BQ%d, wait 0.01 s"%(args.distributed_rank), force=True) 
                time.sleep(0.01)

            i, samples = boss_queue.get(block=True)            
            num_batches_seen += 1
            log_output = trainer.train_step(samples, assistant_queue = assistant_queue)
            if log_output is None:
                continue

            # log mid-epoch stats
            stats = get_training_stats(trainer)
            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue  # these are already logged above
                if 'loss' in k:
                    extra_meters[k].update(v, log_output['sample_size'])
                else:
                    extra_meters[k].update(v)
                stats[k] = extra_meters[k].avg
            progress.log(stats)

            # ignore the first mini-batch in words-per-second calculation
            trainer.get_meter('wps').reset()

            num_updates = trainer.get_num_updates()
            if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0 :
                valid_losses = validate(args, trainer, task, epoch, [first_valid])
                save_checkpoint(args, trainer, epoch, (epoch==max_epoch-1), valid_losses[0])

            if num_updates >= max_update:
                break

        # log end-of-epoch stats
        stats = get_training_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats)

        # reset training meters
        for k in [
            'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
        ]:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()

        if epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch, valid_subsets)

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch, valid_losses[0])

        # save checkpoint
        if epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch, (epoch==max_epoch-1), valid_losses[0])
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))



def validate(args, trainer, task, cur_epoch, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, cur_epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)
        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats)

        valid_losses.append(stats['valid_loss'])
    return valid_losses


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best'] = min(save_checkpoint.best, stats['valid_loss'])
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, cur_epoch, epoch_itr_end_of_epoch,  val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = cur_epoch
    end_of_epoch = epoch_itr_end_of_epoch
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and True
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'train_iterator': None,
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def main(args):
    print("Startint model with distribution of %d"%(args.distributed_world_size))
    # Set distributed training parameters for a single node.
    if not args.distributed_world_size:
        args.distributed_world_size = torch.cuda.device_count()
    port = random.randint(10000, 20000)
    args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
    args.distributed_init_host = 'localhost'
    args.distributed_port = port + 1

    mp = torch.multiprocessing.get_context('spawn')

    boss_queue = []
    assistant_queue = []
    iter_stats = []
    for i in range(args.distributed_world_size):
        boss_queue.append(mp.Queue(maxsize=50))
        iter_stats.append(mp.Queue(maxsize=8))
        assistant_queue.append(mp.Queue(maxsize=50))
    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)


    # Train with multiprocessing.
    procs = []
    samplers = []


    for i in range(args.distributed_world_size):
        args.distributed_rank = i
        args.device_id = i
        samplers.append(multiprocessing.Process(target=BQ_feeder, \
                   args=( i,args, boss_queue[i], assistant_queue[i], iter_stats[i], )))
        samplers[i].start()
        procs.append(mp.Process(target=run, args=(args, error_queue, boss_queue[i], assistant_queue[i], iter_stats[i]), daemon=True))
        procs[i].start()
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()
    for q in samplers:
        q.terminate()

def run(args, error_queue, boss_queue, assistant_queue, iter_stats):
    try:
        args.distributed_rank = distributed_utils.distributed_init(args)
        single_process_main(args, boss_queue, assistant_queue, iter_stats)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.distributed_rank, traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        self.children_pids.append(pid)

    def error_listener(self):
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = "\n\n-- Tracebacks above this line can probably be ignored --\n\n"
        msg += original_trace
        raise Exception(msg)


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
