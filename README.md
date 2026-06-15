# Multimodal Sign Language Translation: ASL to English Prototype

**Álvaro García Mosqueda, Bersun Şipal, Inés Martínez Fernández**  

---

## Introduction

Sign Language Translation (SLT) aims to bridge communication between the Deaf community and non-signers. Most existing approaches rely on costly RGB video processing, requiring large-scale datasets and significant computational resources. This project proposes a prototype for ASL-to-English translation based on 2D skeletal keypoints — body, face, and hands — extracted from the How2Sign dataset, using a transformer-based architecture to encode spatio-temporal pose sequences and generate text.

Rather than following a gloss-based pipeline, where signs are first mapped to linguistic annotations before being translated to text, this system takes a gloss-free approach and directly translates pose sequences into English, avoiding error propagation between stages and reducing annotation requirements.

## Data and Preprocessing

The experiments use the How2Sign dataset (Duarte et al., 2021), a large-scale corpus for American Sign Language consisting of approximately 27,000 video clips paired with English sentence-level translations. Due to computational constraints, we work with the B-F-H 2D Keypoints subset, which provides pre-extracted 2D keypoints from the OpenPose framework including body, face, and hand landmarks — 137 joints per frame, encoded as a 274-dimensional vector per frame.

Several preprocessing steps were applied to standardize the data. All clips were resampled to a fixed length of 150 frames, with shorter clips zero-padded and longer ones truncated, resulting in tensors of shape (150, 274). To reduce inter-signer variability, clip-level normalization was applied: keypoints were translated to a coordinate system centered at the midpoint between the shoulders and scaled by the Euclidean distance between them, preserving temporal coherence across the clip. The target text was tokenized using a subword tokenizer adapted to the How2Sign corpus, resulting in a vocabulary of 11,711 tokens — a reduction from the original BERT vocabulary of ~30,000 that improves computational efficiency while preserving the ability to represent rare words through subword decomposition.


## Model Architecture

The proposed architecture combines a MotionBERT-based encoder (Zhu et al., 2022) with an autoregressive Transformer decoder.

The encoder is built on MotionBERT's Dual-Stream Transformer (DSTformer), which models spatial and temporal dependencies through two parallel attention streams: the spatial stream captures relationships between joints within a frame, while the temporal stream models each joint's evolution over time. Several modifications were introduced to adapt this backbone to sign language translation. Instead of standard mean pooling, a joint-attention pooling mechanism was added, using a learnable linear layer to weight joints by their importance for each sign. Additionally, an independent hand stream processes hand keypoints through a lightweight projection module, producing embeddings that are concatenated with the joint-attention output. The resulting fused representation is projected into a 256-dimensional latent space compatible with the decoder.

The decoder is an autoregressive Transformer with four layers, each consisting of causal self-attention, cross-attention over the encoder output, and a feed-forward network. At each generation step, the decoder predicts the next token and feeds it back autoregressively until an end-of-sequence token is produced. During inference, beam search with a beam width of 5 and a repetition penalty of 1.3 is used to improve output fluency.

---

## Training

Training follows a two-stage strategy over up to 40 epochs with early stopping. In the first stage (epochs 1–12), the MotionBERT backbone is frozen and only the newly introduced components are trained — joint-attention pooling, hand stream, projection layers, and decoder — using the Adam optimizer with a learning rate of 3×10⁻⁴. In the second stage, the backbone is unfrozen and the full model is fine-tuned end-to-end at the same learning rate. Cross-entropy loss with label smoothing of 0.15 is used throughout, with learning rate reduction by a factor of 0.5 when validation performance plateaus.



## Results

The model achieves a BLEU-4 score of **3.56** on the How2Sign test set. While low in absolute terms, this result is consistent with the difficulty of the task: 2D pose-only inputs discard important visual information such as depth, hand shape detail, and facial expressions, and the dataset presents high variability in signer appearance and motion dynamics.

Training dynamics show that the best validation performance is reached around the transition between the frozen and fine-tuning stages (epoch 11–12). After unfreezing, the validation loss plateaus while training loss continues to decrease, suggesting limited additional generalization benefit from full fine-tuning. Early models suffer from decoding collapse, producing repetitive outputs; the final configuration mitigates this and generates more diverse, input-conditioned sentences, though semantic accuracy remains limited.


## Limitations and Future Work

The main limitation of this approach is its reliance on 2D pose representations, which discard depth, hand shape, and fine-grained facial information. The conversion from OpenPose to H36M format may also introduce information loss. Computationally, the use of pre-extracted keypoints was motivated by storage and GPU constraints, which limited experimentation and scaling.

Future work could explore richer input modalities, stronger pretrained decoders, and improved multimodal fusion strategies to close the gap with full RGB-based systems.

---

## References

Duarte, A. et al. (2021). How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language. In *CVPR*.

Wong, R., Camgoz, N. C., & Bowden, R. (2024). Sign2GPT: Leveraging Large Language Models for Gloss-Free Sign Language Translation. In *ICLR*.

Zhou, B. et al. (2023). Gloss-Free Sign Language Translation: Improving from Visual-Language Pretraining. In *ICCV*.

Zhu, W. et al. (2022). MotionBERT: A Unified Perspective on Learning Human Motion Representations. *arXiv:2210.06551*.
