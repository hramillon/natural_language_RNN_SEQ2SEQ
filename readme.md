# Image recognition with MLP and CNN

The goal is to understand how we can compute natural language with deep learning.
To do that we have to use Recurrent Neural Network.

The final goal is to try to do a simple system of english -> french translation.

### Our Approach

We start from the fundamentals of the recurrent neural network and progressively increase complexity :

1. **Word extension with RNN**
  i. Recurrent Neural Network (RNN)
  ii. How to turn words in vectors (Embeddings)
  iii bigram and trigram
2. **How to remember things in our sentences**
  i. Our first sentences with RNN
  ii. LSTM and GRU to have more context
3. **Translation with seq2seq**
  i.Principle of seq2seq
  ii Our strategy
  iii our models

## Word Extension with RNN

### What is a recurrent neural network ?

**Recurrent Neurons**

![Recurrent neural network unfolded](https://upload.wikimedia.org/wikipedia/commons/b/b5/Recurrent_neural_network_unfold.svg)

A recurrent neuron (shown on the left of the image) has a self-loop that feeds its output back as input. In other words, the neuron receives both an external input and its own previous output.

Mathematically, this is expressed as:

```
S_0 = o

S_{t+1} = ρ((e_{t+1} · S_t) W_r + b_r)

o = S_{t+1} W_o + b_o
```

Where:
- `S_t` is the output at state at time `t`
- `e_{t+1}` is the input at time `t+1`
- `W_r` and `b_r` are the recurrent layer's weights and bias
- `ρ` is the activation function
- `W_o` and `b_o` are the output layer's weights and bias

Memory Through Time :
Basically, we can see in the equation $(et+1⋅St)(e_{t+1} \cdot S_t)$ and especially $(et+1​⋅St​)$ that the output is mixed with the new input. This means the neuron is supposed to remember what happened in the past.

For instance, if we analyze a sentence word by word, at the second word we mix the output (which analyzed the first word) with the input of the second word. This is clearly better to understand a sentence, as we carry forward the context from previous words.
In this way, we can visualize the neuron through time by connecting it on a timeline, like in the right side of the image. We clearly see the equation has two inputs: the new one from the corpus and the output of the last sequence. This recurrent connection allows the network to maintain information about previous inputs and use it to process current ones.

### How to turn words in vectors ? (Embeddings)

Now that we have an idea of how to use RNNs to understand sentences, we need a way to convert words into vectors. It's much easier for computers to compute with floating-point numbers than with raw text.
The idea is to assign each word a vector in an n-dimensional space. In this space, words are positioned based on their semantic meaning and usage patterns. If the word "cat" is often followed by "eat," then the vectors for "cat" and "eat" should be positioned close to each other in this semantic space.
Example:

"cat" might be represented as [0.2, 0.8, 0.1, ...]
"dog" might be represented as [0.25, 0.75, 0.15, ...] (similar to "cat")
"pizza" might be represented as [0.1, 0.2, 0.9, ...] (different from "cat")

If we have n dimesions it's for having different features (n actually), for instance "cat", "dog" and "pizza" if we arein a 2-dimensional space the first dimension can be about being alive and the second being eatable, it's oversimplified but it's the idea.

#### How Embeddings Connect with RNNs

**Step 1: Tokenization to Embedding Lookup**

Words are first converted to numerical tokens, then these tokens are used to retrieve vectors from an embeddings matrix:

"dog" → token 238 → [0.2, 0.8, 0.1, ...]
"cat" → token 543 → [0.4, 0.1, 0.9, ...]

Each token acts as an index to fetch a corresponding vector representation. These vectors capture semantic relationships learned from the entire corpus during training.

**Step 2: Contextual Processing by the RNN**
The embedding vectors are then passed sequentially to the RNN, which processes them one word at a time while maintaining hidden state. The RNN's role is fundamentally different from embeddings:

Embeddings provide static vector representations based on global patterns in the data
RNN provides dynamic contextual understanding based on the sequence of words in the current sentence

The Key Distinction
Embeddings alone cannot understand that "bank" means something different in "river bank" versus "savings bank", they assign the same vector regardless of context. The RNN solves this by analyzing the surrounding words and previous words in the sequence.

#### Lets make our Embdedding