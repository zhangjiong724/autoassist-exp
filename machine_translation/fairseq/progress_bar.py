"""
Wrapper around various loggers and progress bars (e.g., tqdm).
"""

from collections import OrderedDict
import json
from numbers import Number
import sys

from tqdm import tqdm

from fairseq.meters import AverageMeter
from fairseq.distributed_utils import get_rank, get_world_size

def build_progress_bar(args, iterator, epoch=None, prefix=None, default='tqdm', no_progress_bar='none', assistant=None):
    main_process = True
    try:
        main_process = (args.device_id == 0)
    except:
        pass

    if args.log_format is None:
        args.log_format = no_progress_bar if args.no_progress_bar else default

    if args.log_format == 'tqdm' and not sys.stderr.isatty():
        args.log_format = 'simple'

    if args.log_format == 'json':
        bar = json_progress_bar(iterator, epoch, prefix, args.log_interval, main_process)
    elif args.log_format == 'none':
        bar = noop_progress_bar(iterator, epoch, prefix, main_process)
    elif args.log_format == 'simple':
        if hasattr(args, 'distributed_world_size') and  args.distributed_world_size > 1:
            bar = simple_parallel_progress_bar(iterator, epoch, prefix, args.log_interval, main_process, assistant=assistant)
        else:
            bar = simple_progress_bar(iterator, epoch, prefix, args.log_interval, main_process, assistant=assistant)
    elif args.log_format == 'tqdm':
        bar = tqdm_progress_bar(iterator, epoch, prefix, main_process)
    else:
        raise ValueError('Unknown log format: {}'.format(args.log_format))
    return bar


class progress_bar(object):
    """Abstract class for progress bars."""
    def __init__(self, iterable, epoch=None, prefix=None, main_process = False):
        self.iterable = iterable
        self.main_process = main_process
        self.epoch = epoch
        self.prefix = ''
        if epoch is not None:
            self.prefix += '| epoch {:03d}'.format(epoch)
        if prefix is not None:
            self.prefix += ' | {}'.format(prefix)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        raise NotImplementedError

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        raise NotImplementedError

    def print(self, stats):
        """Print end-of-epoch stats."""
        raise NotImplementedError

    def _str_commas(self, stats):
        return ', '.join(key + '=' + stats[key].strip()
                         for key in stats.keys())

    def _str_pipes(self, stats):
        return ' | '.join(key + ' ' + stats[key].strip()
                          for key in stats.keys())

    def _format_stats(self, stats):
        postfix = OrderedDict(stats)
        # Preprocess stats according to datatype
        for key in postfix.keys():
            # Number: limit the length of the string
            if isinstance(postfix[key], Number):
                postfix[key] = '{:g}'.format(postfix[key])
            # Meter: display both current and average value
            elif isinstance(postfix[key], AverageMeter):
                postfix[key] = '{:.2f} ({:.2f})'.format(
                    postfix[key].val, postfix[key].avg)
            # Else for any other type, try to get the string conversion
            elif not isinstance(postfix[key], str):
                postfix[key] = str(postfix[key])
            # Else if it's a string, don't need to preprocess anything
        return postfix


class json_progress_bar(progress_bar):
    """Log output in JSON format."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000, main_process=True):
        super().__init__(iterable, epoch, prefix, main_process = main_process)
        self.log_interval = log_interval
        self.stats = None

    def __iter__(self):
        size = float(len(self.iterable))
        for i, obj in enumerate(self.iterable):
            yield obj
            if self.stats is not None and i > 0 and \
                    self.log_interval is not None and i % self.log_interval == 0:
                update = self.epoch - 1 + float(i / size) if self.epoch is not None else None
                stats = self._format_stats(self.stats, epoch=self.epoch, update=update)
                if self.main_process:
                    print(json.dumps(stats), flush=True)

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        self.stats = stats

    def print(self, stats):
        """Print end-of-epoch stats."""
        self.stats = stats
        stats = self._format_stats(self.stats, epoch=self.epoch)
        if self.main_process:
            print(json.dumps(stats), flush=True)

    def _format_stats(self, stats, epoch=None, update=None):
        postfix = OrderedDict()
        if epoch is not None:
            postfix['epoch'] = epoch
        if update is not None:
            postfix['update'] = update
        # Preprocess stats according to datatype
        for key in stats.keys():
            # Meter: display both current and average value
            if isinstance(stats[key], AverageMeter):
                postfix[key] = stats[key].val
                postfix[key + '_avg'] = stats[key].avg
            else:
                postfix[key] = stats[key]
        return postfix


class noop_progress_bar(progress_bar):
    """No logging."""

    def __init__(self, iterable, epoch=None, prefix=None, main_process = True):
        super().__init__(iterable, epoch, prefix, main_process = main_process)

    def __iter__(self):
        for obj in self.iterable:
            yield obj

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        pass

    def print(self, stats):
        """Print end-of-epoch stats."""
        pass

class simple_parallel_progress_bar(progress_bar):
    """A minimal logger for non-TTY environments."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000, main_process = False, assistant=None, truncate=None):
        super().__init__(iterable, epoch, prefix, main_process = main_process)
        self.log_interval = log_interval
        self.stats = None
        self.assistant = assistant
        self.world_size = get_world_size()
        self.successes = [ 0 for i in range(self.world_size)]
        self.samples = [ 0 for i in range(self.world_size)]
        self.confidence = [ 0 for i in range(self.world_size)]
        self.my_id = get_rank()
        self.truncate = truncate

    def __iter__(self):
        size = len(self.iterable)
        for i, obj in enumerate(self.iterable):
            yield obj
            if self.stats is not None and i > 0 and \
                    self.log_interval is not None and i % self.log_interval == 0:
                postfix = self._str_commas(self.stats)
                if self.assistant is not None:
                    self.successes[self.my_id] = self.assistant.total_success
                    self.samples[self.my_id] = self.assistant.total_samples
                    self.confidence[self.my_id] = self.assistant.confident
                if self.main_process:
                    if self.assistant is not None:
                        print('{}:  {:5d} / {:d} {}, accept_rate={:d}/{:d}, confidence={}'.format(self.prefix, i, size, postfix,
                            sum(self.successes), sum(self.samples), self.confidence),
                            flush=True)
                    else:
                        print('{}:  {:5d} / {:d} {}'.format(self.prefix, i, size, postfix),
                            flush=True)

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        self.stats = self._format_stats(stats)

    def print(self, stats):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        if self.main_process:
            print('{} | {}'.format(self.prefix, postfix), flush=True)

class simple_progress_bar(progress_bar):
    """A minimal logger for non-TTY environments."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000, main_process = False, assistant=None):
        super().__init__(iterable, epoch, prefix, main_process = main_process)
        self.log_interval = log_interval
        self.stats = None
        self.assistant = assistant
        try: 
            self.world_size = get_world_size()
            self.my_id = get_rank()
        except:
            self.world_size = 1
            self.my_id = 0
        self.successes = [ 0 for i in range(self.world_size)]
        self.samples = [ 0 for i in range(self.world_size)]
        self.confidence = [ 0 for i in range(self.world_size)]

    def __iter__(self):
        size = len(self.iterable)
        for i, obj in enumerate(self.iterable):
            yield obj
            if self.stats is not None and i > 0 and \
                    self.log_interval is not None and i % self.log_interval == 0:
                postfix = self._str_commas(self.stats)
                if self.assistant is not None:
                    self.successes[self.my_id] = self.assistant.total_success
                    self.samples[self.my_id] = self.assistant.total_samples
                    self.confidence[self.my_id] = self.assistant.confident
                if self.main_process:
                    print('{}:  {:5d} / {:d} {}, no sampelr'.format(self.prefix, i, size, postfix),
                        flush=True)

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        self.stats = self._format_stats(stats)

    def print(self, stats):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        if self.main_process:
            print('{} | {}'.format(self.prefix, postfix), flush=True)


class tqdm_progress_bar(progress_bar):
    """Log to tqdm."""

    def __init__(self, iterable, epoch=None, prefix=None, main_process=True):
        super().__init__(iterable, epoch, prefix, main_process = main_process)
        self.tqdm = tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.tqdm)

    def log(self, stats):
        """Log intermediate stats according to log_interval."""
        self.tqdm.set_postfix(self._format_stats(stats), refresh=False)

    def print(self, stats):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        if self.main_process:
            self.tqdm.write('{} | {}'.format(self.tqdm.desc, postfix))
