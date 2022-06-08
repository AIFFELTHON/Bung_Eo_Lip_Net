

## 개발환경
* python 3.6
* Keras 2.0+
* Tensorflow 1.0+
* Ubuntu 16.04  
*  cuda 8.0  
*  cudnn 5
  

필수 패키지 설치
```
cd LipNet/
pip install -e .
```
비디오 파일을 처리하기 위해 Ubuntu에서 ffmpeg 설치
``apt install ffmpeg``

You're done!

Here is some ideas on what you can do next:
* Modify the package and make some improvements to it.
* Train the model using predefined training scenarios.
* Make your own training scenarios.
* Use [pre-trained weights](https://github.com/rizkiarm/LipNet/tree/master/evaluation/models) to do lipreading.
* Go crazy and experiment on other dataset! by changing some hyperparameters or modify the model.

## Dataset
This model uses GRID corpus (http://spandh.dcs.shef.ac.uk/gridcorpus/)

## Pre-trained weights
For those of you who are having difficulties in training the model (or just want to see the end results), you can download and use the weights provided here: https://github.com/rizkiarm/LipNet/tree/master/evaluation/models. 

More detail on saving and loading weights can be found in [Keras FAQ](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).

## Training
There are five different training scenarios that are (going to be) available:

### Prerequisites
1. Download all video (normal) and align from the GRID Corpus website.
2. Extracts all the videos and aligns.
3. Create ``datasets`` folder on each training scenario folder.
4. Create ``align`` folder inside the ``datasets`` folder.
5. All current ``train.py`` expect the videos to be in the form of 100x50px mouthcrop image frames.
You can change this by adding ``vtype = "face"`` and ``face_predictor_path`` (which can be found in ``evaluation/models``) in the instantiation of ``Generator`` inside the ``train.py``
6. The other way would be to extract the mouthcrop image using ``scripts/extract_mouth_batch.py`` (usage can be found inside the script).
7. Create symlink from each ``training/*/datasets/align`` to your align folder.
8. You can change the training parameters by modifying ``train.py`` inside its respective scenarios.

### Random split (Unmaintained)
Create symlink from ``training/random_split/datasets/video`` to your video dataset folder (which contains ``s*`` directory).

Train the model using the following command:
```
./train random_split [GPUs (optional)]
```

**Note:** You can change the validation split value by modifying the ``val_split`` argument inside the ``train.py``.
### Unseen speakers
Create the following folder:
* ``training/datasets/train``
* ``training/datasets/val``

Then, create symlink from ``training/datasets/[train|val]/s*`` to your selection of ``s*`` inside of the video dataset folder.

The paper used ``s1``, ``s2``, ``s20``, and ``s22`` for evaluation and the remainder for training.

Train the model using the following command:
```
./train [GPUs (optional)]
```
### Unseen speakers with curriculum learning
The same way you do unseen speakers.

**Note:** You can change the curriculum by modifying the ``curriculum_rules`` method inside the ``train.py``

```
./train curriculum [GPUs (optional)]
```

### Overlapped Speakers
Run the preparation script:
```
python prepare.py [Path to video dataset] [Path to align dataset] [Number of samples]
```
**Notes:**
- ``[Path to video dataset]`` should be a folder with structure: ``/s{i}/[video]``
- ``[Path to align dataset]`` should be a folder with structure: ``/[align].align``
- ``[Number of samples]`` should be less than or equal to ``min(len(ls '/s{i}/*'))``

Then run training for each speaker:
```
python training/overlapped_speakers/train.py s{i}
```

### Overlapped Speakers with curriculum learning
Copy the ``prepare.py`` from ``overlapped_speakers`` folder to ``overlapped_speakers_curriculum`` folder, 
and run it as previously described in overlapped speakers training explanation.

Then run training for each speaker:
```
python training/overlapped_speakers_curriculum/train.py s{i}
```
**Note:** As always, you can change the curriculum by modifying the ``curriculum_rules`` method inside the ``train.py``

## Evaluation
To evaluate and visualize the trained model on a single video / image frames, you can execute the following command:
```
./predict [path to weight] [path to video]
```
**Example:**
```
./predict evaluation/models/overlapped-weights368.h5 evaluation/samples/id2_vcd_swwp2s.mpg
```
## Work in Progress
This is a work in progress. Errors are to be expected.
If you found some errors in terms of implementation please report them by submitting issue(s) or making PR(s). Thanks!

**Some todos:**
- [X] Use ~~Stanford-CTC~~ Tensorflow CTC beam search
- [X] Auto spelling correction
- [X] Overlapped speakers (and its curriculum) training
- [ ] Integrate language model for beam search
- [ ] RGB normalization over the dataset.
- [X] Validate CTC implementation in training.
- [ ] Proper documentation
- [ ] Unit tests
- [X] (Maybe) better curriculum learning.
- [ ] (Maybe) some proper scripts to do dataset stuff.

## License
MIT License
