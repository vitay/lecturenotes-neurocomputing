# Attentional neural networks

Slides: [pdf](https://www.tu-chemnitz.de/informatik/KI/edu/neurocomputing/lectures/pdf/3.10-Attention.pdf)

```{note}
Most figures and all videos in this chapter are taken from a series of great blog posts by Jay Alammar:

<https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/>

<http://jalammar.github.io/illustrated-transformer/>

<https://jalammar.github.io/illustrated-bert/>

<https://jalammar.github.io/illustrated-gpt2/>
```


## RNNs with Attention

<div class='embed-container'><iframe src='https://www.youtube.com/embed/Gi8aHnnyDaM' frameborder='0' allowfullscreen></iframe></div>

<br>

The problem with the seq2seq architecture is that it **compresses** the complete input sentence into a single state vector.

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="https://jalammar.github.io/images/seq2seq_6.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

Source: <https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/>

For long sequences, the beginning of the sentence may not be present in the final state vector:

* Truncated BPTT, vanishing gradients.
* When predicting the last word, the beginning of the paragraph might not be necessary.

Consequence: there is not enough information in the state vector to start translating. A solution would be to concatenate the **state vectors** of all steps of the encoder and pass them to the decoder.

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="https://jalammar.github.io/images/seq2seq_7.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

Source: <https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/>

* **Problem 1:** it would make a lot of elements in the state vector of the decoder (which should be constant).
* **Problem 2:** the state vector of the decoder would depend on the length of the input sequence.

Attentional mechanisms {cite}`Bahdanau2016` let the decoder decide (by learning) which state vectors it needs to generate each word at each step.

The **attentional context vector** of the decoder $A^\text{decoder}_t$ at time $t$ is a weighted average of all state vectors $C^\text{encoder}_i$ of the encoder. 

$$A^\text{decoder}_t = \sum_{i=0}^T a_i \, C^\text{encoder}_i$$

<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="https://jalammar.github.io/images/seq2seq_9.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

Source: <https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/>

The coefficients $a_i$ are called the **attention scores** : how much attention is the decoder paying to each of the encoder's state vectors? The attention scores $a_i$ are computed as a **softmax** over the scores $e_i$ (in order to sum to 1):

$$a_i = \frac{\exp e_i}{\sum_j \exp e_j} \Rightarrow A^\text{decoder}_t = \sum_{i=0}^T \frac{\exp e_i}{\sum_j \exp e_j} \, C^\text{encoder}_i$$


<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="https://jalammar.github.io/images/attention_process.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

Source: <https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/>

The score $e_i$ is computed using:

* the previous output of the decoder $\mathbf{h}^\text{decoder}_{t-1}$.
* the corresponding state vector $C^\text{encoder}_i$ of the encoder at step $i$.
* attentional weights $W_a$.

$$e_i = \text{tanh}(W_a \, [\mathbf{h}^\text{decoder}_{t-1}; C^\text{encoder}_i])$$

Everything is differentiable, these attentional weights can be learned with BPTT.

The attentional context vector $A^\text{decoder}_t$ is concatenated with the previous output $\mathbf{h}^\text{decoder}_{t-1}$ and used as the next input $\mathbf{x}^\text{decoder}_t$ of the decoder:


$$\mathbf{x}^\text{decoder}_t = [\mathbf{h}^\text{decoder}_{t-1} ; A^\text{decoder}_t]$$


<video width="100%" height="auto" loop="" autoplay="" controls="">
  <source src="https://jalammar.github.io/images/attention_tensor_dance.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

Source: <https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/>

```{figure} ../img/seq2seq-attention5.png
---
width: 100%
---
Seq2seq architecture with attention {cite}`Bahdanau2016`. Source: <https://towardsdatascience.com/day-1-2-attention-seq2seq-models-65df3f49e263>.
```

The attention scores or **alignment scores** $a_i$ are useful to interpret what happened. They show which words in the original sentence are the most important to generate the next word.

```{figure} ../img/seq2seq-attention7.png
---
width: 60%
---
Alignment scores during translation. Source: <https://towardsdatascience.com/day-1-2-attention-seq2seq-models-65df3f49e263>.
```

**Attentional mechanisms** are now central to NLP. The whole **history** of encoder states is passed to the decoder, which learns to decide which part is the most important using **attention**. This solves the bottleneck of seq2seq architectures, at the cost of much more operations. They require to use fixed-length sequences (generally 50 words). 

```{figure} ../img/seq2seq-comparison.png
---
width: 100%
---
Comparison of seq2seq and seq2seq with attention. Source: <https://towardsdatascience.com/day-1-2-attention-seq2seq-models-65df3f49e263>.
```

Google Neural Machine Translation (GNMT {cite}`Wu2016`) uses an attentional recurrent NN, with bidirectional GRUs, 8 recurrent layers on 8 GPUs for both the encoder and decoder.

```{figure} ../img/google-nmt-lstm.png
---
width: 100%
---
Google Neural Machine Translation (GNMT {cite}`Wu2016`)
```


## Transformers (optional)


### Transformer networks

<div class='embed-container'><iframe src='https://www.youtube.com/embed/AO-gqXNCEx0' frameborder='0' allowfullscreen></iframe></div>


#### Architecture

Attentional mechanisms are so powerful that recurrent networks are not even needed anymore.
**Transformer networks** {cite}`Vaswani2017` use **self-attention** in a purely feedforward architecture and outperform recurrent architectures.
They are used in Google BERT and OpenAI GPT-2/3 for text understanding (e.g. search engine queries).

```{figure} ../img/transformer_resideual_layer_norm_3.png
---
width: 100%
---
Architecture of the Transformer. Source: <http://jalammar.github.io/illustrated-transformer/>
```

Transformer networks use an **encoder-decoder** architecture, each with 6 stacked layers.

```{figure} ../img/transformer1.png
---
width: 100%
---
Encoder-decoder structure of the Transformer. Source: <http://jalammar.github.io/illustrated-transformer/>
```


Each layer of the encoder processes the $n$ words of the input sentence **in parallel**.
Word embeddings (as in word2vec) of dimension 512 are used as inputs (but learned end-to-end).

```{figure} ../img/encoder_with_tensors.png
---
width: 100%
---
Encoder layer. Source: <http://jalammar.github.io/illustrated-transformer/>
```

Two operations are performed on each word embedding $\mathbf{x}_i$: 

* self-attention vector $\mathbf{z}_i$ depending on the other words.
* a regular feedforward layer to obtain a new representation $\mathbf{r}_i$ (shared among all words). 

```{figure} ../img/encoder_with_tensors_2.png
---
width: 100%
---
Encoder layer. Source: <http://jalammar.github.io/illustrated-transformer/>
```

#### Self-attention

The first step of self-attention is to compute for each word three vectors of length $d_k = 64$ from the embeddings $\mathbf{x}_i$ or previous representations $\mathbf{r}_i$ (d = 512).

* The **query** $\mathbf{q}_i$ using $W^Q$.
* The **key** $\mathbf{k}_i$ using $W^K$.
* The **value** $\mathbf{v}_i$ using $W^V$.

```{figure} ../img/transformer_self_attention_vectors.png
---
width: 100%
---
Self-attention. Source: <http://jalammar.github.io/illustrated-transformer/>
```

This operation can be done in parallel over all words:

```{figure} ../img/self-attention-matrix-calculation.png
---
width: 60%
---
Query, key and value matrices. Source: <http://jalammar.github.io/illustrated-transformer/>
```


Why query / key / value? This a concept inspired from recommendation systems / databases.
A Python dictionary is a set of key / value entries:

```python
tel = {
    'jack': 4098, 
    'sape': 4139
}
```

The query would ask the dictionary to iterate over all entries and return the value associated to the key **equal or close to** the query.

```python
tel['jacky'] # 4098
```

This would be some sort of **fuzzy** dictionary.

In attentional RNNs, the attention scores were used by each word generated by the decoder to decide which **input word** is relevant.

If we apply the same idea to the **same sentence** (self-attention), the attention score tells how much words of the same sentence are related to each other (context).

> The animal didn't cross the street because it was too tired.

The goal is to learn a representation for the word `it` that contains information about `the animal`, not `the street`.

```{figure} ../img/transformer_self-attention_visualization.png
---
width: 100%
---
Source: <https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html>
```

Each word $\mathbf{x}_i$ of the sentence generates its query $\mathbf{q}_i$, key $\mathbf{k}_i$ and value $\mathbf{v}_i$.

For all other words $\mathbf{x}_j$, we compute the **match** between the query $\mathbf{q}_i$ and the keys $\mathbf{k}_j$ with a dot product:

$$e_{i, j} = \mathbf{q}_i^T \, \mathbf{k}_j$$ 

We normalize the scores by dividing by $\sqrt{d_k} = 8$ and apply a softmax:

$$a_{i, j} = \text{softmax}(\dfrac{\mathbf{q}_i^T \, \mathbf{k}_j}{\sqrt{d_k}})$$

The new representation $\mathbf{z}_i$ of the word $\mathbf{x}_i$ is a weighted sum of the values of all other words, weighted by the attention score:

$$\mathbf{z}_i = \sum_{j} a_{i, j} \, \mathbf{v}_j$$

```{figure} ../img/self-attention-output.png
---
width: 100%
---
Summary of self-attention on two words. Source: <http://jalammar.github.io/illustrated-transformer/>
```

If we concatenate the word embeddings into a $n\times d$ matrix $X$, self-attention only implies matrix multiplications and a row-based softmax:

$$
\begin{cases}
    Q = X \times W^Q \\
    K = X \times W^K \\
    V = X \times W^V \\
    Z = \text{softmax}(\dfrac{Q \times K^T}{\sqrt{d_k}}) \times V \\
\end{cases}
$$

```{figure} ../img/self-attention-matrix-calculation-2.png
---
width: 100%
---
Self-attention. Source: <http://jalammar.github.io/illustrated-transformer/>
```

Note 1: everything is differentiable, backpropagation will work.

Note 2: the weight matrices do not depend on the length $n$ of the sentence.

#### Multi-headed self-attention

In the sentence *The animal didn't cross the street because it was too tired.*, the new representation for the word `it` will hopefully contain features of the word `animal` after training.

But what if the sentence was *The animal didn't cross the street because it was too **wide**.*? The representation of `it` should be linked to `street` in that context.
This is not possible with a single set of matrices $W^Q$, $W^K$ and $W^V$, as they would average every possible context and end up being useless.

```{figure} ../img/transformer-needforheads.png
---
width: 100%
---
Source: <https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html>
```


The solution is to use **multiple attention heads** ($h=8$) with their own matrices $W^Q_k$, $W^K_k$ and $W^V_k$.

```{figure} ../img/transformer_attention_heads_qkv.png
---
width: 100%
---
Multiple attention heads. Source: <http://jalammar.github.io/illustrated-transformer/>
```


Each **attention head** will output a vector $\mathbf{z}_i$ of size $d=512$ for each word.
How do we combine them?

```{figure} ../img/transformer_attention_heads_z.png
---
width: 100%
---
Multiple attention heads. Source: <http://jalammar.github.io/illustrated-transformer/>
```

The proposed solution is based on **ensemble learning** (stacking): let another matrix $W^O$ decide which attention head to trust... $8 \times 512$ rows, 512 columns.

```{figure} ../img/transformer_attention_heads_weight_matrix_o.png
---
width: 100%
---
Multiple attention heads. Source: <http://jalammar.github.io/illustrated-transformer/>
```

```{figure} ../img/transformer_multi-headed_self-attention-recap.png
---
width: 100%
---
Summary of self-attention. Source: <http://jalammar.github.io/illustrated-transformer/>
```

Each attention head learns a different context:

* `it` refers to `animal`.
* `it` refers to `street`.
* etc.

The original transformer paper in 2017 used 8 attention heads. OpenAI's GPT-3 uses 96 attention heads...

#### Encoder layer

Multi-headed self-attention produces a vector $\mathbf{z}_i$ for each word of the sentence.
A regular feedforward MLP transforms it into a new representation $\mathbf{r}_i$. 

* one input layer with 512 neurons.
* one hidden layer with 2048 neurons and a ReLU activation function.
* one output layer with 512 neurons.

The same NN is applied on all words, it does not depend on the length $n$ of the sentence.

```{figure} ../img/encoder_with_tensors_2.png
---
width: 100%
---
Encoder layer. Source: <http://jalammar.github.io/illustrated-transformer/>
```

#### Positional encoding

As each word is processed in parallel, the order of the words in the sentence is lost.

> street was animal tired the the because it cross too didn't

We need to explicitly provide that information in the **input** using positional encoding.

A simple method would be to append an index $i = 1, 2, \ldots, n$ to the word embeddings, but it is not very robust.

```{figure} ../img/transformer_positional_encoding_vectors.png
---
width: 100%
---
Positional encoding. Source: <http://jalammar.github.io/illustrated-transformer/>
```

If the elements of the 512-d embeddings are numbers between 0 and 1, concatenating an integer between 1 and $n$ will unbalance the dimensions. 
Normalizing that integer between 0 and 1 requires to know $n$ in advance, this introduces a maximal sentence length...

How about we append the binary code of that integer?

```{figure} ../img/trasnformer-positionalencoding.png
---
width: 100%
---
Positional encoding. Source: <https://kazemnejad.com/blog/transformer_architecture_positional_encoding/>
```

Sounds good, we have numbers between 0 and 1, and we just need to use enough bits to encode very long sentences. 
However, representing a binary value (0 or 1) with a 64 bits floating number is a waste of computational resources.

We can notice that the bits of the integers oscillate at various frequencies:

* the lower bit oscillates every number.
* the bit before oscillates every two numbers.
* etc.

We could also represent the position of a word using sine and cosine functions at different frequencies (Fourier basis).
We create a vector, where each element oscillates at increasing frequencies.
The "code" for each position in the sentence is unique.

```{figure} ../img/positional_encoding.png
---
width: 100%
---
Positional encoding. Source: <https://kazemnejad.com/blog/transformer_architecture_positional_encoding/>
```

In practice, a 512-d vector is created using sine and cosine functions.

$$
    \begin{cases}
        t(\text{pos}, 2i) = \sin(\dfrac{\text{pos}}{10000^{2 i / 512}})\\
        t(\text{pos}, 2i + 1) = \cos(\dfrac{\text{pos}}{10000^{2 i / 512}})\\
    \end{cases}
$$

```{figure} ../img/attention-is-all-you-need-positional-encoding.png
---
width: 100%
---
Positional encoding. Source: <http://jalammar.github.io/illustrated-transformer/>
```

The positional encoding vector is **added** element-wise to the embedding, not concatenated!

$$\mathbf{x}_{i} = \mathbf{x}^\text{embed}_{i} + \mathbf{t}_i$$


#### Layer normalization

The last tricks of the encoder layers are:

* skip connections (residual layer)
* layer normalization

The input $X$ is added to the output of the multi-headed self-attention and normalized (zero mean, unit variance).

**Layer normalization** {cite}`Ba2016` is an alternative to batch normalization, where the mean and variance are computed over single vectors, not over a minibatch:

$$\mathbf{z} \leftarrow \dfrac{\mathbf{z} - \mu}{\sigma}$$

with $\mu = \dfrac{1}{d} \displaystyle\sum_{i=1}^d z_i$ and $\sigma = \dfrac{1}{d} \displaystyle\sum_{i=1}^d (z_i - \mu)^2$.


The feedforward network also uses a skip connection and layer normalization.

```{figure} ../img/transformer_resideual_layer_norm_2.png
---
width: 100%
---
Encoder layer with layer normalization and skip connections. Source: <http://jalammar.github.io/illustrated-transformer/>
```

We can now stack 6 (or more, 96 in GPT-3) of these encoder layers and use the final representation of each word as an input to the decoder.

```{figure} ../img/transformer_resideual_layer_norm_3.png
---
width: 100%
---
The encoder is a stack of encoder layers. Source: <http://jalammar.github.io/illustrated-transformer/>
```

#### Decoder

In the first step of decoding, the final representations of the encoder are used as query and value vectors of the decoder to produce the first word. 
The input to the decoder is a "start of sentence" symbol.

```{figure} ../img/transformer_decoding_1.gif
---
width: 100%
---
Encoding of a sentence. Source: <http://jalammar.github.io/illustrated-transformer/>
```


The decoder is **autoregressive**: it outputs words one at a time, using the previously generated words as an input.

```{figure} ../img/transformer_decoding_2.gif
---
width: 100%
---
Autoregressive generation of words. Source: <http://jalammar.github.io/illustrated-transformer/>
```

Each decoder layer has two multi-head attention sub-layers:

* A self-attention sub-layer with query/key/values coming from the generated sentence.
* An **encoder-decoder** attention sub-layer, with the query coming from the generated sentence and the key/value from the encoder.

The encoder-decoder attention is the regular attentional mechanism used in seq2seq architectures.
Apart from this additional sub-layer, the same residual connection and layer normalization mechanisms are used.

```{figure} ../img/transformer-architecture.png
---
width: 60%
---
Transformer architecture. Source {cite}`Vaswani2017`.
```

When the sentence has been fully generated (up to the `<eos>` symbol), **masked self-attention** has to applied in order for a word in the middle of the sentence to not "see" the solution in the input when learning. Learning occurs on minibatches of sentences, not on single words.

```{figure} ../img/self-attention-and-masked-self-attention.png
---
width: 100%
---
Masked self-attention. Source: <https://jalammar.github.io/illustrated-gpt2/>
```



The output of the decoder is a simple softmax classification layer, predicting the one-hot encoding of the word using a vocabulary (`vocab_size=25000`).

```{figure} ../img/transformer_decoder_output_softmax.png
---
width: 100%
---
Word production from the output. Source: <http://jalammar.github.io/illustrated-transformer/>
```

#### Results

The transformer is trained on the WMT datasets:

* English-French: 36M sentences, 32000 unique words.
* English-German: 4.5M sentences, 37000 unique words.

Cross-entropy loss, Adam optimizer with scheduling, dropout. Training took 3.5 days on 8 P100 GPUs.
The sentences can have different lengths, as the decoder is autoregressive.
The transformer network beat the state-of-the-art performance in translation with less computations and without any RNN.

```{figure} ../img/transformer-results.png
---
width: 100%
---
Performance of the Transformer on NLP tasks. Source {cite}`Vaswani2017`.
```

### Self-supervised Transformers

<div class='embed-container'><iframe src='https://www.youtube.com/embed/jYDEFZWn6co' frameborder='0' allowfullscreen></iframe></div>
<br>

The Transformer is considered as the **AlexNet** moment of natural language processing (NLP).
However, it is limited to supervised learning of sentence-based translation.

Two families of architectures have been developed from that idea to perform all NLP tasks using **unsupervised pretraining** or **self-supervised training**:

* BERT (Bidirectional Encoder Representations from Transformers) from Google {cite}`Devlin2019`.
* GPT (Generative Pre-trained Transformer) from OpenAI <https://openai.com/blog/better-language-models/>.

```{figure} ../img/gpt-2-transformer-xl-bert-3.png
---
width: 100%
---
Source: <https://jalammar.github.io/illustrated-gpt2/>
```


#### BERT

BERT {cite}`Devlin2019` only uses the encoder of the transformer (12 layers, 12 attention heads, $d = 768$). BERT is pretrained on two different unsupervised tasks before being fine-tuned on supervised tasks.

* Task 1: Masked language model. Sentences from BooksCorpus and Wikipedia (3.3G words) are presented to BERT during pre-training, with 15% of the words masked.
The goal is to predict the masked words from the final representations using a shallow FNN.

```{figure} ../img/BERT-language-modeling-masked-lm.png
---
width: 100%
---
Masked language model. Source: <https://jalammar.github.io/illustrated-bert/>
```


* Task 2: Next sentence prediction. Two sentences are presented to BERT. 
The goal is to predict from the first representation whether the second sentence should follow the first.

```{figure} ../img/bert-next-sentence-prediction.png
---
width: 100%
---
Next sentence prediction. Source: <https://jalammar.github.io/illustrated-bert/>
```

Once BERT is pretrained, one can use **transfer learning** with or without fine-tuning from the high-level representations to perform:

* sentiment analysis / spam detection
* question answering

```{figure} ../img/bert-classifier.png
---
width: 100%
---
Sentiment analysis with BERT. Source: <https://jalammar.github.io/illustrated-bert/>
```

```{figure} ../img/bert-transfer-learning.png
---
width: 100%
---
Transfer learning with BERT. Source: <https://jalammar.github.io/illustrated-bert/>
```

#### GPT 

GPT is an **autoregressive** language model learning to predict the next word using only the transformer's **decoder**.

```{figure} ../img/transformer-decoder-intro.png
---
width: 100%
---
GPT is the decoder of the Transformer. Source: <https://jalammar.github.io/illustrated-gpt2/>
```


**Autoregression** mimicks a LSTM that would output words one at a time.

```{figure} ../img/gpt-2-autoregression-2.gif
---
width: 100%
---
Autoregression. Source: <https://jalammar.github.io/illustrated-gpt2/>
```


GPT-2 comes in various sizes, with increasing performance.
GPT-3 is even bigger, with 175 **billion** parameters and a much larger training corpus.

```{figure} ../img/gpt2-sizes-hyperparameters-3.png
---
width: 100%
---
GPT sizes. Source: <https://jalammar.github.io/illustrated-gpt2/>
```

GPT can be fine-tuned (transfer learning) to perform **machine translation**.

```{figure} ../img/decoder-only-transformer-translation.png
---
width: 100%
---
Machine translation with GPT. Source: <https://jalammar.github.io/illustrated-gpt2/>```


GPT can be fine-tuned to summarize Wikipedia articles.

```{figure} ../img/wikipedia-summarization.png
---
width: 100%
---
Wikipedia summarization with GPT. Source: <https://jalammar.github.io/illustrated-gpt2/>
```


```{figure} ../img/decoder-only-summarization.png
---
width: 100%
---
Wikipedia summarization with GPT. Source: <https://jalammar.github.io/illustrated-gpt2/>
```


````{note}
Try transformers at <https://huggingface.co/>
```bash
pip install transformers
```
````


Github and OpenAI trained a GPT-3-like architecture on the available open source code.
Copilot is able to "autocomplete" the code based on a simple comment/docstring.



```{figure} ../img/githubcopliot.gif
---
width: 100%
---
Github Copilot. <https://copilot.github.com/>
```


All NLP tasks (translation, sentence classification, text generation) are now done using transformer-like architectures (BERT, GPT), **unsupervisedly** pre-trained on huge corpuses. 
BERT can be used for feature extraction, while GPT is more generative.
Transformer architectures seem to **scale**: more parameters = better performance. Is there a limit?

The price to pay is that these models are very expensive to train (training one instance of GPT-3 costs 12M$) and to use (GPT-3 is only accessible with an API). 
Many attempts have been made to reduce the size of these models while keeping a satisfying performance.

* DistilBERT, RoBERTa, BART, T5, XLNet...

See <https://medium.com/mlearning-ai/recent-language-models-9fcf1b5f17f5>.


### Vision transformers

<div class='embed-container'><iframe src='https://www.youtube.com/embed/mK9HMXZVSUQ' frameborder='0' allowfullscreen></iframe></div>
<br>

The transformer architecture can also be applied to computer vision, by splitting images into a **sequence** of small patches (16x16). The sequence of vectors can then be classified by the output of the transformer using labels.

```{figure} ../img/vision-transformer.gif
---
width: 100%
---
Vision Tranformer (ViT). Source: <https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html>
```


The Vision Transformer (ViT, {cite}`Dosovitskiy2021`) outperforms state-of-the-art CNNs while requiring less computations.

```{figure} ../img/ViTPerformance.png
---
width: 100%
---
ViT performance. Source: {cite}`Dosovitskiy2021`
```


```{figure} ../img/ViTPerformance2.png
---
width: 100%
---
ViT performance. Source: {cite}`Dosovitskiy2021`
```

ViT only works on big supervised datasets (ImageNet). Can we benefit from self-supervised learning as in BERT or GPT? The Self-supervised Vision Transformer (SiT) {cite}`Atito2021` has an denoising autoencoder-like structure, reconstructing corrupted patches autoregressively.


```{figure} ../img/SiT.png
---
width: 100%
---
SiT architecture {cite}`Atito2021`.
```

Self-supervised learning is possible through from **data augmentation** techniques. Various corruptions (masking, replacing, color distortion, blurring) are applied to the input image, but SiT must reconstruct the original image (denoising autoencoder). 

```{figure} ../img/SiT-training.png
---
width: 100%
---
Data augmentation for SiT {cite}`Atito2021`.
```

An auxiliary **rotation loss** forces SiT to predict the orientation of the image (e.g. 30°). Another auxiliary **contrastive loss** ensures that high-level representations are different for different images.

```{figure} ../img/SiT-results.png
---
width: 100%
---
Performance of SiT {cite}`Atito2021`.
```

A recent approach for self-supervised learning has been proposed by Facebook AI researchers using **self-distillation** (Self-distillation with no labels - DINO, {cite}`Caron2021`). The images are split into **global** and **local patches** at different scales. Global patches contain label-related information (whole objects) while local patches contain finer details.

```{figure} ../img/DINO-images.gif
---
width: 100%
---
Global and local views for DINO. Source: <https://towardsdatascience.com/on-dino-self-distillation-with-no-labels-c29e9365e382>
```

The idea of **self-distillation** in DINO is to use two similar ViT networks to classify the patches.
The **teacher** network gets the global views as an input, while the **student** network get both the local and global ones.
Both have a MLP head to predict the softmax probabilities, but do **not** use any labels.

```{figure} ../img/DINO-distillation.png
---
width: 60%
---
Self-distillation in DINO. {cite}`Caron2021`.
```

The student tries to imitate the output of the teacher, by minimizing the **cross-entropy** (or KL divergence) between the two probability distributions. The teacher slowly integrates the weights of the student (momentum or exponentially moving average ema):

$$\theta_\text{teacher} \leftarrow \beta \, \theta_\text{teacher} + (1 - \beta) \, \theta_\text{student}$$



```{figure} ../img/DINO-architecture2.gif
---
width: 100%
---
DINO training. Source: <https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/>
```

The predicted classes do not matter when pre-training, as there is no ground truth. 
The only thing that matters is the **high-level representation** of an image before the softmax output, which can be used for transfer learning.
Self-distillation forces the representations to be meaningful at both the global and local scales, as the teacher gets global views. 
ImageNet classes are already separated in the high-level representations: a simple kNN (k-nearest neighbour) classifier achieves 74.5% accuracy (vs. 79.3% for a supervised ResNet50).


```{figure} ../img/DINO-tsne.gif
---
width: 100%
---
t-SNE visualization of the feature representations in DINO. Source: <https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/>
```

More interestingly, by looking at the self-attention layers, one can obtain saliency maps that perform **object segmentation** without ever having been trained to!

<div class='embed-container'><iframe src='https://www.youtube.com/embed/8I1RelnsgMw' frameborder='0' allowfullscreen></iframe></div>

<br>

Transformers can also be used for time-series classification or forecasting instead of RNNs. Example: weather forecasting, market prices, etc.

```{figure} ../img/transformer-timeseries.png
---
width: 100%
---
Time series forecasting. {cite}`Wu2020`
```

```{figure} ../img/transformer-timeseries-architecture.png
---
width: 100%
---
Time series transformer. {cite}`Wu2020`
```
