11785 HW1P2
Name: Xinkai Chen
AndrewID: xinkaic
Code: 11785_hw1p2.ipynb
Steps:
1. Download data and unzip them
2. Import libraries and run on GPU
3. Define dataloader and load training data and dev data
- Read in the inputs, pad the utterances, record the utterance indices and indices within utterances (in a dictionary)
- Only get context when __getitem__ is called: find the utterance index and within utterance index, and then give context (slice)
- __getitem__ returns the label as well: it also find the indices
- After tuning, used a context size of 20
4. Define network
- My structure: Linear((1+2*20)*13,1024)->ReLU()->BatchNorm(1024)->Linear(1024,2048)->ReLU()->BatchNorm(2048)->Linear(2048,2048)->ReLU()
             ->BatchNorm(2048)->Linear(2048,2048)->ReLU()->BatchNorm(2048)->Linear(2048,1024)->ReLU()->BatchNorm(1024)->Linear(1024,346)
5. Define training parameters
- Criterion: CrossEntropyLoss()
- Learning rate: starting with 0.001, decay through training steps
- Optimizer: AdamW, it has Decoupled Weight Decay Regularization
6. Training
- Batch size: 5000 (so there are around 5000+ mini-batches, works fine for me)
- Epochs: 10
- After every two epochs, decay the learning rate
- Validate after each epoch
7. Evaluate results and tune hyperparameters
- At first my context size was 10, and then I increased it to 20 to boost model performance
- At first I used a smaller network (3 layers, width = 1024) and that wasn't good enough. So I increased layer width and depth
- At first I decay the learning rate manually and too fast, and then I adjusted it
8. When the val accuracy is okay, load the test set and predict on test set
9. Generate submission.csv and submit to Kaggle
10. Sorry if something looks weird, I got 72% accuracy and then I didn't have much time to refine it due to some personal matters. 
Will try to do better next time. Thank you!