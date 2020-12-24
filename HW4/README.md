# Language Model + Speech-to-Text

Part 1: Language Modeling using RNNs:

* Dataset: A pre-processed WikiText-2 Language Modeling Dataset

* Overwrite Pytorch's DataLoader

* Training: uses regularization techniques in the [paper](https://arxiv.org/pdf/1708.02182.pdf)

  * Locked dropout, Embedding dropout, Weight decay, Weight tying, Activity regularization, Temporal activity regularization
 
* Problems:

  * Prediction of a Single Word
  
  * Generation of a Sequence

* Evaluation: Negative Log Likelihood (NLL) < 5.0

Part 2: Attention-based End-to-End Speech-to-Text Deep Neural Network

* Approach: [Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211.pdf) (LAS)

* Character-based

* Encoder - hidden size 256, uses pBLSTM

* Decoder - hidden size 512, uses attention

* Cross-Entropy Loss with Padding

* Teacher Forcing
