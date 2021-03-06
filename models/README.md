<h1 align="center">Lipreading Models</h1>

<div align="right">
  AIFFEL DAEGU 1 TEAM ๋ป๋๋ป๋
  <br>
  <i>#Tags: Lipreading, LipNet, ShuffleNet-TCN, CV, NLP</i>
</div>

---

## ๐ TABLE OF CONTENTS

- [I/O](#io)
- [๊ณตํต ๊ตฌ์กฐ](#๊ณตํต-๊ตฌ์กฐ)
- [LipNet (2016)](#lipnet-2016)
- [ShuffleNetTCN (2020)](#shufflenettcn-2020)

---

## I/O

![Lipreading Model Input Output][Lipreading Model Input Output]

[Lipreading Model Input Output]: imgs/Lipreading_input_and_output.png

- Input: Video(.avi), Align(.txt)
- Output: Text(.txt) โ Video(.gif, .avi)

## ๊ณตํต ๊ตฌ์กฐ

|LipNet|ShuffleNetTCN|
|:---:|:---:|
|![LipNet Architecture Description][LipNet Architecture Description]|![ShuffleNetTCN Architecture Description][ShuffleNetTCN Architecture Description]|

[LipNet Architecture Description]: imgs/LipNet_architecture_description.png
[ShuffleNetTCN Architecture Description]: imgs/ShuffleNetTCN_architecture_description.png

- ๋ฅ๋ฌ๋ 2๋จ๊ณ ์ ๊ทผ๋ฒ
  - Frontend: 3D-CNN(3D conv layer + deep 2D conv)
  - Backent: LSTM, Attention Mechanisms, self-attention module, TCN (Temporal Convolutional Networks)

**[โฌ back to top](#-table-of-contents)**

---

## LipNet (2016)

### Key Contributions

1. ์ต์ด์ end-to-end ๋ฌธ์ฅ ๋จ์ ๋ชจ๋ธ
2. GRID Corpus dataset

### Process

![LipNet Process][LipNet Process]

[LipNet Process]: imgs/LipNet_architecture.png

- STCNN: video ์์ ์๊ฐ์ ํ๋ฆ๊ณผ ๊ณต๊ฐ์ ์ฐจ์์ ๋ชจ๋ convolution
- bi-LSTM: STCNN ์ output sequence ์ ๋ณด๋ฅผ ์ ํํ๊ธฐ ์ํด ์ฌ์ฉ, ์ ๋ณด ํ๋ฆ ์ ์ด ํ์ต
- CTC Loss: Target sequence ์ output sequence ๊ธธ์ด๊ฐ ๋ค๋ฅผ ๋ ์ฌ์ฉ
- Label โ UNICODE (encoding) โ ํ๊ตญ์ด (decoding)

### ๋จ์  ํ์ ๋ฐ ๋ถ์

- ์ฐ์ฐ๋ ์ด์: GCP ํ๊ฒฝ์์ ํ์ต์ ๋๋ฆด ๋ GCP ๊ฐ ๊บผ์ง๋ ํ์ ๋ฐ์
- ์์ธ ๋ถ์
  - ํ๊ตญ์ด character ๊ฐ์๊ฐ ์์ด character ์ ๋นํด ๊ฐ์ง์๊ฐ ๋ง์
  - ๋ชจ๋ธ์ด ๋ฌด๊ฑฐ์ด ๊ฒ(ํ์ดํผ ํ๋ผ๋ฏธํฐ ๋ง์)์ผ๋ก ํ์๋จ
- ๊ฐ์  ๋ฐฉ๋ฒ
  - ์์  ๋ชจ๋ธ์ธ LipNet ๋ง๊ณ  ์ต์  ๋ชจ๋ธ์ ์ฐพ์๋ณผ ๊ฒ โ ์ต์  ๋ผ๋ฌธ ์ฐพ์๋ณด๊ธฐ
  - ์ฌ๋งํ๋ฉด ๊ฐ๋ฒผ์ด ๋ชจ๋ธ์ผ ๊ฒ โ ๋ผ๋ฌธ์์ ์ฐ์ฐ๋ ๋๋ ํ์ต ์๊ฐ์ ๋ํ ๋ด์ฉ ์ง์คํด์ ๋ณด๊ธฐ
  - ์ฑ๋ฅ์ด ์ข์ ๊ฒ โ SOTA ์ฐพ์๋ณด๊ธฐ

### ๊ฒฐ๋ก 

SOTA ๋ฅผ ๋ฌ์ฑํ ๊ฐ๋ณ๊ณ  ์ต์  ๋จ์ด ๋จ์ ๋ชจ๋ธ์ธ ShuffleNetTCN ์ฌ์ฉ

**[โฌ back to top](#-table-of-contents)**

---

## ShuffleNetTCN (2020)

### Key Contributions

1. Backbone ๊ต์ฒด
    - ResNet-18 โ ShuffleNetV2
    - ํ๋ผ๋ฏธํฐ 5๋ฐฐ ๊ฐ์, FLOPs 12๋ฐฐ ๊ฐ์
2. Shuffle Grouped Convolution
    - ShuffleNet ์์ ์ ์
    - ์ฃผ๊ธฐ์ ์ผ๋ก ๊ทธ๋ฃน ๊ฐ ์ฑ๋์ ์์ด์ ์ ๋ณด๊ฐ ๊ตํ๋๋๋ก ๋ง๋  Group Conv ๋ฐฉ์
3. TCN (Temporal Convolution Network)
    - 1D conv ๋ฅผ Sequence ๋ฐ์ดํฐ์ ์ ์ฉ
    - ๋ณ๋ ฌ ์ฐ์ฐ โ ์ง๋ ฌ RNN ๋ณด๋ค ๋น ๋ฆ
    - receptive field ํฌ๊ธฐ ์กฐ์  ๊ฐ๋ฅ
    - ํ๋์ layer ์ ๋ํ์ฌ ๊ฐ์ ํ๋ผ๋ฏธํฐ ๊ณต์  โ ๋ฉ๋ชจ๋ฆฌ ์์ ์ ์
4. DS-TCN (Depthwise Separable Temporal Convolution Network)
    - Depthwise Conv: ๊ณต๊ฐ ํน์ง ์ถ์ถ
    - Pointwise Conv: ์ฑ๋ ํน์ง ์ถ์ถ
    - ํ์ค Conv ๋ณด๋ค ์ฐ์ฐ๋ 8 ~ 9๋ฐฐ ๊ฐ์
    - ์ต์ข์ ์ผ๋ก DS-MS-TCN ์ฌ์ฉ
5. Knowledge Distillation
    - ํฐ ๋ชจ๋ธ(Teacher)๋ก๋ถํฐ ์ฆ๋ฅํ ์ง์ โ ์์ ๋ชจ๋ธ(Student)๋ก ์ ๋ฌ
    - Teacher Network ์ Student Network ๋ ๋ชจ๋ ๊ฐ์ ๋๋ฉ์ธ โ Self-Distillation ๊ณผ์ 

### Process

![ShuffleNetTCN Process][ShuffleNetTCN Process]

[ShuffleNetTCN Process]: imgs/ShuffleNetTCN_architecture.png

- 3D Conv: video ์ฒ๋ฆฌ
- ResNet-18 or ShuffleNet: image frames ์ฒ๋ฆฌ
- MS-TCN: text output ์์ฑ

**[โฌ back to top](#-table-of-contents)**
