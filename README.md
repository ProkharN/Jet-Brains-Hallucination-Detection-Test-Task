# Jet-Brains-Hallucination-Detection-Test-Task

## 1. Short Description

This repository contains my proposed solution for the test assignment
for the **JetBrains Internship in the Hallucination Detection project**.

The original test task was formulated as follows:

> Implement the core training loop of word2vec in pure NumPy (no PyTorch
> / TensorFlow or other ML frameworks). The applicant is free to choose
> any suitable text dataset. The task is to implement the optimization
> procedure (forward pass, loss, gradients, and parameter updates) for a
> standard word2vec variant (e.g. skip-gram with negative sampling or
> CBOW).

> The submitted solution should be fully understood by the applicant:
> during follow-up we will ask questions about the ideas behind
> word2vec, the code, gradient derivations, and possible alternative
> implementations or optimizations.

In this repository I implemented a **skip-gram Word2Vec model with
negative sampling**, including the full training loop written in **pure
NumPy**.

------------------------------------------------------------------------

# 2. Datasets

During the development of the project two datasets were used:

-   The full text of **Alice's Adventures in Wonderland**, available
    from **Project Gutenberg**
-   A **5MB subset of the text8 dataset**, which contains the first 100
    million characters from Wikipedia

The **Alice dataset** was used during development in order to test the
training loop while it was being implemented. Since it is a relatively
small dataset with predictable contextual patterns, it allows monitoring
whether the intermediate training results appear reasonable at each
stage of the implementation.

Once the implementation was stable, the model was trained on a **subset
of text8**, which represents a more realistic general-purpose corpus.

------------------------------------------------------------------------

# 3. Project Structure

Besides the datasets used in the training loop, the project contains all
modules required to run the training pipeline.

### Modules

main.py data.py model.py train.py utils.py

### main.py

This file orchestrates the entire pipeline.

It calls functions from the other modules and contains easily adjustable
**hyperparameters**, which are separated for convenience and
reproducibility of experiments.

------------------------------------------------------------------------

### data.py

This module handles **data preprocessing and tokenization**.

Functions included:

tokenize\
- converts all text to lowercase\
- removes additional characters except alphabetic symbols

build_vocab\
- constructs a tuple of three dictionaries on a specified vocabulary
size

encode_tokens\
- returns a list of token IDs corresponding to the tokenized text

generate_skipgram_pairs\
- generates pairs consisting of a target word and its context word

build_negative_sampling_distribution\
- computes the probability distribution used for selecting negative
samples

------------------------------------------------------------------------

### model.py

This module implements the **Word2Vec Skip-Gram model with negative
sampling**.

The module contains:

-   initialization of **input and output embedding matrices**
-   implementation of the **forward pass**
-   calculation of the **negative sampling loss**
-   manual computation of **gradients**
-   parameter updates using **stochastic gradient descent**

The model is implemented entirely in **NumPy without external machine
learning frameworks**.

------------------------------------------------------------------------

### train.py

This module implements the **training loop**.

The main responsibilities include:

-   iterating over skip-gram pairs
-   sampling negative examples from the negative sampling distribution
-   calling the model training step
-   computing running loss statistics
-   logging intermediate training results

The training loop is intentionally kept simple and transparent in order
to clearly demonstrate how the optimization process works.

------------------------------------------------------------------------

### utils.py

This module contains **helper functions** used during evaluation.

The main function included here is:

find_nearest_neighbors\
- computes cosine similarity between embeddings\
- returns the nearest contextual neighbors for a given query word

This allows qualitative inspection of the learned word embeddings.

------------------------------------------------------------------------

# 4. Running the Project

To run the training pipeline execute the following command:

python -m src.main

During execution the training process will display intermediate
statistics in the console, including:

-   dataset statistics
-   loss values during training
-   nearest neighbors for selected query words

------------------------------------------------------------------------

# 5. Results

Since the training loop was first tested on the **Alice dataset**, the
results of that experiment are shown below.

### Vocabulary sample

First 10 vocabulary items:

the -\> 0 and -\> 1 to -\> 2 a -\> 3 it -\> 4 she -\> 5 i -\> 6 of -\> 7
said -\> 8 you -\> 9

### Skip-gram pairs

First 10 skip-gram pairs:

(alice, s) (alice, adventures) (s, alice) (s, adventures) (s, in)
(adventures, alice) (adventures, s) (adventures, in) (adventures,
wonderland) (in, s)

### Negative sampling distribution

Sum = 1.000000\
Shape = (2574,)

### Training

Epoch 1 finished.\
Average epoch loss: 3.4539

Epoch 2 finished.\
Average epoch loss: 2.7266

### Nearest neighbors example

alice:
  said            0.8588
  nonsense        0.8552
  herself         0.8323
  thought         0.8131
  indeed          0.8084

rabbit:
  white           0.9945
  by              0.9908
  side            0.9903
  box             0.9902
  turtle          0.9869

queen:
  gryphon         0.9934
  hare            0.9932
  dormouse        0.9907
  march           0.9903
  witness         0.9802

king:
  turning         0.9729
  cat             0.9714
  gryphon         0.9707
  here            0.9658
  queen           0.9639

cat:
  gryphon         0.9755
  king            0.9714
  here            0.9656
  mock            0.9645
  hatter          0.9599


------------------------------------------------------------------------

# 6. Training on text8

The model was also trained on a **subset of text8**, containing
approximately 5MB of text extracted from Wikipedia.

Because the dataset is significantly larger, the model was trained for
**one epoch** to keep training time manageable.

Several hyperparameters were adjusted for this training run:

-   increased vocabulary size
-   larger embedding dimension
-   updated query words more suitable for a general-domain corpus

### Dataset statistics

Number of tokens: 846987\
Vocabulary size: 10000\
Encoded token count: 778152\
Number of skip-gram pairs: 3112602

### Training result

Average epoch loss: 2.3443

### Nearest neighbors example

king:
  philip          0.8485
  lord            0.8338
  pope            0.8282
  president       0.8129
  prince          0.8083

queen:
  mayor           0.9208
  chancellor      0.8801
  president       0.8755
  prime           0.8499
  chicago         0.8394

man:
  woman           0.6404
  scheme          0.6328
  became          0.6195
  person          0.6129
  statue          0.6079

woman:
  person          0.7626
  elected         0.7187
  receiver        0.7173
  custom          0.7108
  entitled        0.7035

city:
  nation          0.8433
  largest         0.8210
  constitution    0.8127
  party           0.7768
  valley          0.7759


------------------------------------------------------------------------

# 7. Interpretation

Both the **Alice dataset** and the **text8 subset** demonstrate that
words that appear as nearest contextual neighbors are often semantically
related or frequently co‑occur within similar contexts.

In the **Alice dataset**, the neighbors are highly **context-specific**,
which is expected because the text represents a single narrative domain.

In contrast, the **text8 experiment** produces neighbors that belong to
a broader semantic field, since the dataset contains a mixture of topics
extracted from Wikipedia.

Therefore, the resulting embeddings are less domain-specific and can
potentially be used for experiments related to **language model
development and debugging**.
