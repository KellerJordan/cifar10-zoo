# Nonlinear dynamics in data valuation
The following are a series of **existence proofs/demonstrations** for nonlinear dynamics in data valuation.
The main prior work is [Wallace 2019](https://distill.pub/2019/advex-bugs-discussion/response-6/).

Let CIFAR-2 refer to the 10,000-example subset of cats and dogs in CIFAR-10, viewed as a binary-classification dataset.
That is, cats are labeled as negative and dogs as positive. (This was used in [TRAK](https://arxiv.org/abs/2303.14186))

Baseline.

- Training a standard ResNet on CIFAR-2 yields 90.6% accuracy.
---------------

Result 1 ("bad + bad = good").

- (Baseline): Training on a random balanced half of CIFAR-2 yields 90.1% accuracy.
- Proof: We construct an *adversarial* split of CIFAR-2, into balanced halves A and B, such that A yields 66.0% and B yields 69.4% accuracy (much much worse than the random split).

Result 2 ("good + worse-than-nothing = better").

- Proof: We construct a split of CIFAR-2 into balanced subsets A and B, such that A yields 84.6% accuracy and B yields 46.8% accuracy (worse than random chance).

Corollary 1. ("worse-than-nothing + better-than-nothing = even-worse")

- Proof: Just invert the labels of everything in Result 2.

Result 3 ("worse-than-nothing x 10 = good")

- Proof: We construct a balanced subset which (a) yields 73.7% accuracy, and (b) has the property that if we randomly split it into 10 pieces, then each piece yields ~37.6% accuracy.

Corollary 2. ("better-than-nothing x 10 = much-worse-than-nothing")

- Proof: Invert the labels in Result 3.

