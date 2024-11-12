#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Starter code for Conditional Random Fields.

from __future__ import annotations
import logging
from math import inf, log, exp
from pathlib import Path
from typing import Callable, Optional
from typing_extensions import override
from typeguard import typechecked

import torch
from torch import Tensor, cuda
from jaxtyping import Float
import sys
import itertools, more_itertools
from tqdm import tqdm # type: ignore

from corpus import (BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag,
                    TaggedCorpus, Word)
from integerize import Integerizer
from hmm import HiddenMarkovModel

TorchScalar = Float[Tensor, ""] # a Tensor with no dimensions, i.e., a scalar

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
    # Note: We use the name "logger" this time rather than "log" since we
    # are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

class ConditionalRandomField(HiddenMarkovModel):
    """An implementation of a CRF that has only transition and 
    emission features, just like an HMM."""
    
    # CRF inherits forward-backward and Viterbi methods from the HMM parent class,
    # along with some utility methods.  It overrides and adds other methods.
    # 
    # Really CRF and HMM should inherit from a common parent class, TaggingModel.  
    # We eliminated that to make the assignment easier to navigate.
    
    @override
    def __init__(self, 
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 unigram: bool = False):
        """Construct an CRF with initially random parameters, with the
        given tagset, vocabulary, and lexical features.  See the super()
        method for discussion."""

        super().__init__(tagset, vocab, unigram)

    @override
    def init_params(self) -> None:
        """Initialize params self.WA and self.WB to small random values, and
        then compute the potential matrices A, B from them.
        As in the parent method, we respect structural zeroes ("Don't guess when you know")."""

        # See the "Training CRFs" section of the reading handout.
        # 
        # For a unigram model, self.WA should just have a single row:
        # that model has fewer parameters.

        rows = 1 if self.unigram else self.k
        self.WB = 0.01 * torch.randn(self.k, self.V)      
        self.WA = 0.01 * torch.randn(rows, self.k)

        self.WB[self.eos_t, :] = -inf             # EOS_TAG can't emit any column's word
        self.WB[self.bos_t, :] = -inf             # BOS_TAG can't emit any column's word
        self.WA[:, self.bos_t] = -inf             # Nothing can transit into BOS
        if not self.unigram:
            self.WA[self.eos_t, :] = -inf         # EOS can transit into nothing
        
        self.updateAB()  

    def updateAB(self) -> None:
        """Set the transition and emission matrices self.A and self.B, 
        based on the current parameters self.WA and self.WB.
        See the "Parametrization" section of the reading handout."""
       
        # Even when self.WA is just one row (for a unigram model), 
        # you should make a full k × k matrix A of transition potentials,
        # so that the forward-backward code will still work.
        # See init_params() in the parent class for discussion of this point.
        self.A = torch.exp(self.WA)
        self.B = torch.exp(self.WB)

        if self.unigram:
            self.A = self.A.repeat(self.k, 1) 





    @override
    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[ConditionalRandomField], float],
              tolerance: float =0.001,
              minibatch_size: int = 16,
              eval_interval: int = 500,
              lr: float = 1.0,
              reg: float = 0.0,
              max_steps: int = 50000,
              save_path: Optional[Path] = Path("my_hmm.pkl")) -> None:
        """Train the CRF on the given training corpus, starting at the current parameters.

        The minibatch_size controls how often we do an update.
        (Recommended to be larger than 1 for speed; can be inf for the whole training corpus,
        which yields batch gradient ascent instead of stochastic gradient ascent.)
        
        The eval_interval controls how often we evaluate the loss function (which typically
        evaluates on a development corpus).
        
        lr is the learning rate, and reg is an L2 batch regularization coefficient.

        We always do at least one full epoch so that we train on all of the sentences.
        After that, we'll stop after reaching max_steps, or when the relative improvement 
        of the evaluation loss, since the last evalbatch, is less than the
        tolerance.  In particular, we will stop when the improvement is
        negative, i.e., the evaluation loss is getting worse (overfitting)."""
        
        def _loss() -> float:
            # Evaluate the loss on the current parameters.
            # This will print its own log messages.
            # 
            # In the next homework we will extend the codebase to use backprop, 
            # which finds gradient with respect to the parameters.
            # However, during evaluation on held-out data, we don't need this
            # gradient and we can save time by turning off the extra bookkeeping
            # needed to compute it.
            with torch.no_grad():  # type: ignore 
                return loss(self)      

        # This is relatively generic training code.  Notice that the
        # updateAB() step before each minibatch produces A, B matrices
        # that are then shared by all sentences in the minibatch.
        #
        # All of the sentences in a minibatch could be treated in
        # parallel, since they use the same parameters.  The code
        # below treats them in series -- but if you were using a GPU,
        # you could get speedups by writing the forward algorithm
        # using higher-dimensional tensor operations that update
        # alpha[j-1] to alpha[j] for all the sentences in the
        # minibatch at once.  PyTorch could then take better advantage
        # of hardware parallelism on the GPU.

        if reg < 0: raise ValueError(f"{reg=} but should be >= 0")
        if minibatch_size <= 0: raise ValueError(f"{minibatch_size=} but should be > 0")
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)  # no point in having a minibatch larger than the corpus
        min_steps = len(corpus)   # always do at least one epoch

        self._zero_grad()     # get ready to accumulate their gradient
        steps = 0
        old_loss = _loss()    # evaluate initial loss
        for evalbatch in more_itertools.batched(
                           itertools.islice(corpus.draw_sentences_forever(), 
                                            max_steps),  # limit infinite iterator
                           eval_interval): # group into "evaluation batches"
            for sentence in tqdm(evalbatch, total=eval_interval):
                # Accumulate the gradient of log p(tags | words) on this sentence 
                # into A_counts and B_counts.
                self.accumulate_logprob_gradient(sentence, corpus)
                steps += 1
                
                if steps % minibatch_size == 0:              
                    # Time to update params based on the accumulated 
                    # minibatch gradient and regularizer.
                    self.logprob_gradient_step(lr)
                    self.reg_gradient_step(lr, reg, minibatch_size / len(corpus))
                    self.updateAB()      # update A and B potential matrices from new params
                    self._zero_grad()    # get ready to accumulate a new gradient for next minibatch
            
            # Evaluate our progress.
            curr_loss = _loss()
            if steps >= min_steps and curr_loss >= old_loss * (1-tolerance):
                break   # we haven't gotten much better since last evalbatch, so stop
            old_loss = curr_loss   # remember for next evalbatch

        # For convenience when working in a Python notebook, 
        # we automatically save our training work by default.
        if save_path: self.save(save_path)
 
    @override
    @typechecked
    def logprob(self, sentence: Sentence, corpus: TaggedCorpus) -> TorchScalar:
        """Return the conditional log-probability log p(tags | words) under the current
        model parameters. If the sentence is not fully tagged, the probability
        will marginalize over all possible tags."""

        # Integerize the sentence for efficient tensor operations
        isent = self._integerize_sentence(sentence, corpus)
        
        # Calculate the joint score log p̃(t, w)
        log_p_t_w = 0.0
        n = len(isent)
        for i in range(1, n-1):
            tag_prev, tag = isent[i - 1][1], isent[i][1]  # Previous and current tags
            word = isent[i][0]  # Current word
            if tag_prev is not None and tag is not None:
                log_p_t_w += self.WA[tag_prev, tag]  # Transition weight
            if tag is not None:
                log_p_t_w += self.WB[tag, word]  # Emission weight


        # Calculate log Z(w) by running the forward pass (normalizing constant)
        log_Z_w = self.forward_pass(isent)

        # Return the conditional log-probability log p(t | w)
        return log_p_t_w - log_Z_w

    def accumulate_logprob_gradient(self, sentence: Sentence, corpus: TaggedCorpus) -> None:
        """Add the gradient of self.logprob(sentence, corpus) into a total minibatch
        gradient that will eventually be used to take a gradient step."""

        # Integerize the supervised and desupervised sentences
        isent_sup = self._integerize_sentence(sentence, corpus)
        isent_desup = self._integerize_sentence(sentence.desupervise(), corpus)


        # emission counts don't need to count for BOS and EOS, thus skip 0 and len - 1
        for i in range(1, len(isent_sup) - 2):
            word = isent_sup[i][0]  # Current word
            tag = isent_sup[i][1]  # Current tag
            self.B_counts[tag, word] += 1
        
        # transition counts iterate from 0 1 transition to n-2 n-1 transition
        for i in range(0, len(isent_sup) - 2):
            tag, tag_next = isent_sup[i][1], isent_sup[i + 1][1] 
            self.A_counts[tag, tag_next] += 1

        # -1.0 for minus the expected counts
        self.E_step(isent_desup, mult=-1.0)



        
    def _zero_grad(self):
        """Reset the gradient accumulator to zero."""
        # You'll have to override this method in the next homework; 
        # see comments in accumulate_logprob_gradient().
        self._zero_counts()

    def logprob_gradient_step(self, lr: float) -> None:
        """Update the parameters using the accumulated logprob gradient.
        lr is the learning rate (stepsize)."""
        
        # Warning: Careful about how to handle the unigram case, where self.WA
        # is only a vector of tag unigram potentials (even though self.A_counts
        # is a still a matrix of tag bigram potentials).
        if self.unigram:
            # for unigram, sum self.A_counts across rows
            self.WA += lr * torch.sum(self.A_counts, dim=0, keepdim=True)
        else:
            self.WA += lr * self.A_counts

        self.WB += lr * self.B_counts

        self._zero_grad()

        
        
    def reg_gradient_step(self, lr: float, reg: float, frac: float) -> None:
        """Update the parameters using the gradient of our regularizer.
        More precisely, this is the gradient of the portion of the regularizer 
        that is associated with a specific minibatch, and frac is the fraction
        of the corpus that fell into this minibatch."""
        
        if reg == 0:
            return  # No regularization needed if reg is zero

        # the coefficient is -2C/M, so technically, we want to do [A, B] -= lr * 2C/M[A, B] or [A, B] = [A, B] - lr * 2C/M[A, B]
        # But since inf - inf = nan, we would instead do [A, B] = (1 - 2*lr*C/M)[A, B]
        # About the frac. 1/M represent the gradient for one sentence. naturally, frac 
        # represents k/M for k number of sentences (the minibatch) from the M sized corpus

        # this is the 1 - 2*lr*C*k/M
        reg_factor = 1 - (2 * lr * reg * frac)

        self.WA *= reg_factor
        self.WB *= reg_factor
    
