This is the official "Spec Sheet" for your project. Follow this order. If you get stuck, consult the linked reference for that specific section.

**The Golden Rule:** Do not write the training loop until Step 5 is complete.

### Reference Material

* **For Exact Dimensions/Layer Names:** [Hugging Face `modeling_gpt2.py](https://www.google.com/search?q=%5Bhttps://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py%5D(https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py))`
* **For Clean Logic:** [Karpathy's `minGPT` (model.py)](https://www.google.com/search?q=%5Bhttps://github.com/karpathy/minGPT/blob/master/mingpt/model.py%5D(https://github.com/karpathy/minGPT/blob/master/mingpt/model.py))
*(Note: Use `minGPT`, NOT `nanochat` or `nanoGPT` for this phase. `minGPT` is the cleanest educational implementation of the 2019 architecture.)*

---

### Step 1: The Blueprint (`GPTConfig`)

Create `model.py`. Define the container for your hyperparameters.

* **Requirement:** A dataclass that holds the 5 magic numbers for GPT-2 Small.
* `vocab_size`: 50,257
* `n_embd`: 768
* `n_layer`: 12
* `n_head`: 12
* `block_size`: 1024


* **Check:** Verify you can instantiate it. `conf = GPTConfig()`.

### Step 2: The Muscle (`MLP`)

Implement the Feed-Forward Network.

* **Input Shape:** `(Batch, Time, 768)`
* **Architecture:**
1. Linear Projection: 768  3072
2. Activation: `GELU` (New approximate version if available, or standard).
3. Linear Projection: 3072  768
4. Dropout


* **Source Check:** Look at `GPT2MLP` in Hugging Face to see the order of operations.

### Step 3: The Brain (`CausalSelfAttention`)

Implement the Multi-Head Attention mechanism.

* **Input Shape:** `(Batch, Time, 768)`
* **Architecture:**
1. Linear `c_attn`: 768  2304 (Calculates Q, K, V simultaneously).
2. **Logic:** Split Q, K, V. Reshape heads to `(Batch, 12, Time, 64)`.
3. **Logic:** Calculate Attention Scores ().
4. **Logic:** Apply **Causal Mask** (Future positions = ).
5. **Logic:** Softmax  Dropout  Multiply by V.
6. Linear `c_proj`: 768  768 (Output projection).


* **Source Check:** Look at `GPT2Attention` in Hugging Face. Pay close attention to how they handle the mask buffering (`register_buffer`).

### Step 4: The Container (`Block`)

Implement a single Transformer Block.

* **Architecture:**
1. LayerNorm 1 (`ln_1`)
2. Attention
3. LayerNorm 2 (`ln_2`)
4. MLP


* **The "Gotcha" (Pre-Norm):**
OpenAI uses a specific order: `x = x + sublayer(norm(x))`.
*Do not* do `norm(x + sublayer(x))`. That is the original 2017 Transformer; it is unstable for deep networks like GPT-2.

### Step 5: The Body (`GPT`)

Implement the full model class.

* **Architecture:**
1. `wte`: Token Embeddings (Vocab  Emb Dim)
2. `wpe`: Position Embeddings (Block Size  Emb Dim)
3. `drop`: Embedding Dropout
4. `h`: A `ModuleList` of 12 `Block`s.
5. `ln_f`: Final LayerNorm.


* **The "Gotcha" (Weight Tying):**
You do *not* create a separate Linear layer for the final output head. You must manually set the weights of the output head to be identical to `wte.weight`.
* *Reference:* Look at `GPTLMHeadModel` in Hugging Face to see how they link `lm_head` and `wte`.



### Step 6: The Transplant (Loading Weights)

**This is the Boss Fight.**
Create a method `from_pretrained(cls, model_type='gpt2')`.

* **Logic:**
1. Initialize your model.
2. Download standard Hugging Face model (`gpt2`).
3. Iterate through your state dict keys and the HF state dict keys.
4. **Crucial Math:** If you used `nn.Linear` and OpenAI used `Conv1D`, you must **transpose** the weights (`.t()`) for every standard linear layer.
5. Copy the data.


* **Source Check:** This logic is tedious. I highly recommend looking at Karpathy's `minGPT` `model.py`  `from_pretrained`. He has the exact loop written out to handle the transposes.

### Step 7: The Sanity Check

Before LoRA, verify the brain works.

* **Test:**
```python
model = GPT.from_pretrained('gpt2')
input_ids = tokenizer.encode("Hello, my name is")
output = model.generate(input_ids)
print(tokenizer.decode(output))

```


* **Success Criteria:** It should complete the sentence grammatically. If it outputs "&&&&&&" or random words, you missed a `.transpose()` or a LayerNorm.

**Start Step 1 now.** Create the file.