# Shared Subtask C – Explaining Dominant Narratives

**Author:** Besher Hassan  
**MBZUAI Leaderboard Team Name:** Besher  
**Email:** Besher.Hassan@mbzuai.ac.ae

---

## Abstract

This report presents my approach to Subtask C of SemEval-2025 Task 10 (Organizers, 2025), which focuses on generating concise and evidence-supported explanations for dominant narratives in English news articles. I experimented with multiple transformer models, including GPT-2 (small, medium, large) (Radford et al., 2019) and BART (base, large) (Lewis et al., 2019), and ultimately selected BART-base for its balance of performance and efficiency. By integrating contextual sentence extraction with the `all-MiniLM-L6-v2` sentence encoder (Wang et al., 2020) and fine-tuning BART-base, the model achieved a BERTScore (Zhang et al., 2020) F1 of **0.8981** on the test set and **0.74662** on the development set.

---

## 1. Introduction

Narratives play a central role in shaping public opinion, especially in online news, where framing can influence perception. Understanding and explaining these narratives is critical for combating disinformation and aiding decision-makers. Subtask C focuses on generating concise explanations for dominant narratives in news articles using evidence from the text. This work develops a robust pipeline combining contextual sentence extraction and structured input formatting to produce high-quality explanations.

---

## 2. Dataset

The dataset comprises annotated articles from two domains: Ukraine-Russia War (URW) and Climate Change (CC). Each entry contains:
- `article_id`: A unique identifier for the article.  
- `dominant_narrative`: The overarching narrative conveyed by the article.  
- `dominant_subnarrative`: A finer-grained narrative under the dominant narrative (or `"none"` if unavailable).  
- `article_text`: The full text of the news article, with an average sentence length of 22 words.  
- `explanation`: A human-written explanation (gold label) justifying the narrative assignment, with an average length of 20 words.  

Dataset splits:
- **Training set:** 88 articles, internally split into 90% training and 10% validation/test.  
- **Development set:** 30 articles without human-annotated explanations; not used for training.

---

## 3. Methodology

### 3.1 Key Features for Best Performance

1. **Contextual Sentence Extraction:** Using `all-MiniLM-L6-v2`, the pipeline extracts the three most relevant sentences from each article based on cosine similarity with a query formed from the dominant narrative and subnarrative.
2. **Input Formatting:** Inputs are structured compactly to respect token limits:

   ```text
   <N> Dominant Narrative </N>
   <Sub-N> Subnarrative </Sub-N>
   <Art> Contextual sentences </Art>
   ```

   *Example:*  
   `<N> Criticism of Climate Movement </N> <Sub-N> Industrial Progress Opposition </Sub-N> <Art> Activists oppose industrial development without evidence. </Art>`

3. **Fine-Tuned BART-base:** Chosen for its balance of performance and efficiency compared to GPT-2 variants and BART-large.
4. **Custom Dataset Class:** Handles tokenization, truncation (512 tokens for inputs, 80 words for outputs), and batching.
5. **Training Configuration:** Learning rate `5 × 10^-5`, batch size `8`, `10` epochs, linear scheduler without warmup.

---

## 4. Experiments and Results

### 4.1 Quantitative Results

Multiple models were evaluated using BERTScore metrics (Precision, Recall, and F1). GPT-2 small served as the baseline with an F1 of 0.7956. BART-base provided the highest F1 of 0.8981 on the test set and 0.74662 on the development set.

![Model Performance Comparison](table.png)

| Model           | Size (M Params) | Precision | Recall | F1     |
|-----------------|-----------------|-----------|--------|--------|
| GPT-2 Small     | 124             | 0.8032    | 0.7891 | 0.7956 |
| GPT-2 Medium    | 355             | 0.8456    | 0.8302 | 0.8378 |
| GPT-2 Large     | 774             | 0.8621    | 0.8434 | 0.8526 |
| BART-Large      | 400             | 0.9042    | 0.8870 | 0.8955 |
| **BART-base**   | 140             | 0.9085    | 0.8879 | **0.8981** |

![Leaderboard Placement](leaderboard.png)

### 4.2 Qualitative Results

- **Climate Change (CC) Example**  
  - *Dominant Narrative:* Criticism of Institutions and Authorities  
  - *Dominant Subnarrative:* Criticism of national governments  
  - *Gold Explanation:* The article discusses resistance to the UK government's Climate Con Programme without detailing residents' concerns.  
  - *Generated Explanation:* Critiques the UK government for inadequate climate action and insufficient urban support.

- **Ukraine-Russia War (URW) Example**  
  - *Dominant Narrative:* Discrediting the West, Diplomacy  
  - *Dominant Subnarrative:* West is tired of Ukraine  
  - *Gold Explanation:* Highlights allies as hesitant to provide further military aid due to elections and funding concerns.  
  - *Generated Explanation:* Attributes aggression to Western countries and cites inadequate support for Ukraine.

---

## 5. Discussion

The fine-tuned BART-base model demonstrated robust performance, effectively capturing dominant narratives such as institutional criticism and geopolitical hesitancy, with average generated explanations of 18 words (aligned with the 80-token maximum and 20-word gold average). However, edge cases revealed limitations:
- **Climate Con Programme example:** The generated explanation aligned with the narrative but missed key programme details.
- **URW example:** The model misinterpreted the narrative, labeling the West as aggressors while correctly noting allied hesitance.

### Future Improvements

- **Data Augmentation:** Use GPT-4o to synthesize alternative explanations for underrepresented cases and to paraphrase existing samples.  
- **Enhanced Context Handling:** Integrate richer contextual embeddings or refined preprocessing to improve sentence-level comprehension.

---

## 6. Conclusion

This approach establishes an effective framework for explaining dominant narratives by combining contextual sentence extraction with a fine-tuned BART-base model. The method delivers competitive BERTScore performance, and targeted data augmentation plus improved context modeling offer promising avenues for further gains.

---

## References

- Lewis, M., Liu, Y., Goyal, N., et al. (2019). *BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension*. arXiv:1910.13461.  
- SemEval Task Organizers. (2025). *SemEval-2025 Task 10: Explaining dominant narratives in multilingual news articles*.  
- Radford, A., Wu, J., Child, R., et al. (2019). *Language models are unsupervised multitask learners*. OpenAI Blog.  
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). *MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers*. NeurIPS 33, 5776–5788.  
- Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). *BERTScore: Evaluating text generation with BERT*. arXiv:1904.09675.

