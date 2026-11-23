# Live demonstration script — about six minutes

## 0:00–0:45 · Introduction

“Our project is VocalLens: Hear the voice. See the emotion. We explore expression cues in speech and faces. A recording gives us acoustic patterns, a photo gives us a visible facial expression, and a video lets us compare both. These are estimates from models; we cannot establish how a person actually feels.”

Point out the three modes and the shared interface.

## 0:45–2:00 · Voice

Open Voice only, upload happy.wav and preview the recording. Click Analyze voice.

“This sample has an acted Happy label. The result shown here is the model’s actual prediction. The largest bar is its top score; other bars show competing classes. If it predicts another class, that is a model error on this sample, and it is useful to see it openly. We trained a logistic regression using 138 acoustic features, including MFCC, pitch and energy summaries. We kept speakers apart when selecting and testing the model.”

“On the historical six-actor RAVDESS comparison it reached 49.72% accuracy. Those actors were excluded from training, but their test results have been examined during development. We also tested transfer to TESS and German EmoDB. Those are acted datasets, so we do not claim this is real-world accuracy.”

## 2:00–3:10 · Photo

Switch to Face only and Photo. Upload happy.jpg, point out the preview, and click Analyze photo.

“YuNet detects the face and facial landmarks. We align the crop and pass it to the OpenCV Zoo expression model. The numbered box links this face to its result. In a group photo, every detected face gets a separate numbered result. The score is over expression labels, not certainty about a person’s feelings. Small or dim faces can be unreliable.”

Show the alternative label and probability bars. Describe what is actually displayed.

## 3:10–4:00 · Video

Choose Video, upload sad.mp4, preview it and click Analyze video faces.

“We sample at most 12 frames instead of processing every frame. The final face distribution averages the valid sampled face scores. This is not a continuous timeline. In a video with several people, the largest face can change, so the application warns about that.”

## 4:00–5:15 · Multimodal

Switch to Voice + Face and upload happy.mp4. Click Analyze both cues.

“Here we extract audio and sample frames from the same video. These two panels show separate voice and face predictions. They can agree or disagree, and both remain visible. The combined cue averages scores over the seven shared classes after excluding the speech-only Calm class. It is experimental, without joint calibration.”

“On the historical Actor 05 comparison, the updated combined result reached 65.38% accuracy. On two additional actors it reached 57.69% accuracy and 0.570 macro F1. The new voice model improved combined macro F1 slightly there but reduced accuracy by one clip. That mixed result is why we keep fusion experimental.”

## 5:15–6:00 · Conclusion

Show the Webcam interface if time allows. Say whether capture has actually been tested on this laptop.

“The main work combines trained speech classification, pretrained facial analysis, input validation, and transparent comparisons. We replaced FER+ after a labeled comparison: on Actor 05, face accuracy improved from 34.62% to 51.92%, and Sad recall improved from zero of eight clips to three of eight. There are still errors, especially with subtle expressions. We need more independent people, natural recordings and difficult lighting before making stronger claims. The reports and reproducible tests are included with the project.”

Leave about a minute for questions. If inference takes longer, shorten the introduction rather than skipping uncertainty or evaluation limitations.
