from fairseq.data import data_utils, FairseqDataset, iterators
import torch
import numpy as np

class FairseqTask(object):
    """
    Tasks store dictionaries and provide helpers for loading/iterating over
    Datasets, initializing the Model/Criterion and calculating the loss.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        pass

    def __init__(self, args):
        self.args = args
        self.datasets = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        raise NotImplementedError

    def dataset(self, split):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset
        if split not in self.datasets:
            raise KeyError('Dataset not loaded: ' + split)
        if not isinstance(self.datasets[split], FairseqDataset):
            raise TypeError('Datasets are expected to be of type FairseqDataset')
        return self.datasets[split]


    def get_batch_iterator(
        self, dataset, assistant = None, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, batch_method = 'sentences'
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch.
                Default: ``None``
            max_sentences (int, optional): max number of sentences in each
                batch. Default: ``None``
            max_positions (optional): max sentence length supported by the
                model. Default: ``None``
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long. Default: ``False``
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N. Default: ``1``
            seed (int, optional): seed for random number generator for
                reproducibility. Default: ``1``
            num_shards (int, optional): shard the data iterator into N
                shards. Default: ``1``
            shard_id (int, optional): which shard of the data iterator to
                return. Default: ``0``

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        indices = data_utils.filter_by_size(
            indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
        )

        # create mini-batches with given size constraints
        if assistant is not None:
            assistant.associate_data(dataset, indices)
        else:
            batch_sampler = data_utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )

        if assistant is not None:
            # return a reusable, sharded iterator
            return iterators.AssistantEpochBatchIterator(
                dataset=dataset,
                collate_fn=dataset.collater,
                assistant = assistant,
                max_tokens = max_tokens,
                max_sentences = max_sentences,
                required_batch_size_multiple = required_batch_size_multiple,
                shard_num = num_shards,
                shard_id = shard_id,
                batch_method = batch_method,
                seed=seed,
            )

        else:
            # return a reusable, sharded iterator
            return iterators.EpochBatchIterator(
                dataset=dataset,
                collate_fn=dataset.collater,
                batch_sampler=batch_sampler,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
            )

    def build_model(self, args):
        """
        Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
        task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.models.BaseFairseqModel` instance
        """
        from fairseq import models
        return models.build_model(args, self)

    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions
        return criterions.build_criterion(args, self)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False, lambda_t = None):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        

        use_spl = (self.args.spl and (lambda_t is not None))
        if self.args.assistant:
            loss, sample_size, logging_output, precisions = criterion(model, sample)
        elif use_spl:
            loss, sample_size, logging_output, precisions = criterion(model, sample)
        else:
            loss, sample_size, logging_output = criterion(model, sample)

        if loss.numel() > 1 and not use_spl:
            loss_sum = loss.sum()
        elif loss.numel() > 1 and use_spl: 
            q_t = 4.0
            base_weight = 0.01
            batch_size = len(sample['id'])
            loss_copy = loss.clone().detach()
            
            def self_paced_weight( losses):
                act = torch.nn.ReLU()
                sp_binary = (losses < lambda_t).float().cuda()
                sp_weights = act(1.0 - losses / lambda_t)**(1.0/ (q_t -1.0))
                weight_sum = sp_weights.sum() + base_weight * batch_size
                return ( batch_size / weight_sum) *  (sp_weights + base_weight )

            #weights = torch.tensor(self_paced_weight(losses).numpy()).cuda()
            weights = self_paced_weight(loss_copy)

            loss_sum = (loss * weights).sum()
        else:
            loss_sum = loss
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss_sum)
        if self.args.assistant:
            return loss, sample_size, logging_output, precisions
        elif use_spl:
            return loss, sample_size, logging_output, precisions
        else:
            return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            if self.args.assistant or self.args.spl:
                if sample is None:
                    logging_output = {
                        'loss': 0,
                        'nll_loss': 0,
                        'ntokens': 0,
                        'nsentences': 0,
                        'sample_size': 0,
                    }
                    return 0, 0, logging_output
                loss, sample_size, logging_output, precisions = criterion(model, sample)
            else:
                loss, sample_size, logging_output = criterion(model, sample)
            if loss.numel() > 1:
                loss_sum = loss.sum()
            else:
                loss_sum = loss
        return loss_sum, sample_size, logging_output

    def init_logging_output(self, sample):
        return {
            'ntokens': sample['ntokens'] if sample is not None else 0,
            'nsentences': sample['target'].size(0) if sample is not None else 0,
        }

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        return criterion._aggregate_logging_outputs(logging_outputs)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        raise NotImplementedError
