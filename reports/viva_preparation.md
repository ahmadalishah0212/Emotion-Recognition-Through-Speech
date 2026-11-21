# Viva preparation: Emotion Recognition Through Speech

## Project in plain language

The system takes a short speech recording and predicts which of eight **acted vocal expressions** it resembles. It uses the RAVDESS audio-only speech set because it provides clear labels, 24 actors for speaker-separated testing, and an official download that fits a normal laptop. Its actors speak controlled English statements, so it does not represent all real conversation. The input is audio; the output is a label and model score distribution. These are guesses about vocal expression, not proof of a person's feelings.

The dataset filename encodes modality, speech/song, emotion, intensity, statement, repetition and actor. We keep only `03-01` audio-only speech. Labels 01–08 are neutral, calm, happy, sad, angry, fearful, disgust and surprised. We check missing/unreadable WAVs, exclude byte-identical duplicates, use mono 16 kHz audio, limit very long clips, reject silence and normalize volume. There are no spreadsheet-like null feature cells to fill. We keep six actors entirely for testing, avoiding a model seeing the same speaker in both groups.

Each recording becomes 46 numbers. MFCCs summarize the shape of the sound spectrum, which relates to voice quality. RMS energy measures loudness; zero-crossing rate measures rapid waveform sign changes; spectral centroid measures how much energy is in higher frequencies. We use mean and standard deviation to capture both typical value and variation. We scale features for SVM and logistic regression using only training data. The saved pipeline applies the same scaling in the app.

SVM learns boundaries between classes and can model curved boundaries with an RBF kernel, but scaling matters and it may be harder to explain. Random forest votes across many decision trees, handles nonlinear patterns and needs no scaling, but can overfit and may be less smooth. Logistic regression is a fast, clear linear baseline, but may miss nonlinear class patterns. The actual chosen model is **pending training**. The script selects the highest held-out macro F1; read the generated comparison and confusion matrices before claiming why one won.

Accuracy is the share of all correct predictions. Precision asks: of clips predicted as one emotion, how many were right? Recall asks: of clips truly in that emotion, how many were found? F1 balances precision and recall. Macro F1 averages equally across emotions, so a weak small class is not hidden by a large class. A confusion matrix shows actual labels against predicted labels and reveals which emotions get mixed up.

In the app, the browser uploads a file or records microphone audio. The app validates the bytes, decodes the signal, applies the shared feature extractor, loads `models/selected.json` and its saved model, predicts, then shows the label and model scores. The model does not use the filename or a chosen label as input.

## Likely questions and short answers

1. **Why this dataset?** It has eight labeled speech expressions, 24 speakers, an official download and manageable size.
2. **What is its main limitation?** It is acted studio English speech, unlike spontaneous noisy conversation.
3. **What is the target variable?** The emotion label encoded in the RAVDESS filename.
4. **What are the inputs?** Numbers extracted from the actual speech waveform.
5. **Why not use the filename to predict?** The filename contains the answer; that would be cheating and unusable for new audio.
6. **What is an MFCC?** A compact measure of a recording's spectral shape, loosely related to vocal timbre.
7. **Why use mean and standard deviation?** They capture the average sound and how it changes during a clip.
8. **Why resample to 16 kHz?** To make recordings and app input use the same frequency scale.
9. **Why normalize volume?** Different microphone levels should not dominate the feature values.
10. **Why check duplicates?** Repeated recordings can bias evaluation and inflate scores.
11. **What about missing values?** We check for missing or damaged audio files; there are no tabular null cells to impute.
12. **What is training?** Fitting model parameters using labeled examples from the training actors.
13. **What is testing?** Measuring predictions on actors whose recordings were not used to fit the model.
14. **Why split by actor?** The same voice in training and test could make performance look better than it is for a new person.
15. **What is data leakage?** Information from test examples influencing training or preprocessing.
16. **What is overfitting?** Learning quirks of training audio that fail on unseen speakers.
17. **How does SVM work?** It finds boundaries separating classes in a scaled feature space.
18. **How does random forest work?** Many decision trees vote; averaging helps reduce a single tree's errors.
19. **How does logistic regression work?** It learns linear weighted scores for the classes and converts them to probabilities.
20. **Why these three?** They compare nonlinear boundaries, tree voting and a simple linear baseline on a CPU.
21. **What is accuracy?** Correct predictions divided by all test predictions.
22. **What is precision?** Correct predictions of a class divided by all predictions of that class.
23. **What is recall?** Correct predictions of a class divided by all actual examples of that class.
24. **What is F1?** The harmonic mean of precision and recall.
25. **Why use macro F1?** It gives every emotion equal influence despite class imbalance.
26. **What does a confusion matrix show?** Which true emotion classes the model confuses with others.
27. **Which model was best?** Check `results/model_comparison/comparison.csv` after training; no measured winner exists yet.
28. **Why might the winner beat the others?** Use the real per-class results to explain whether its boundaries capture the feature patterns better.
29. **How does an upload become a prediction?** Decode, validate, resample, extract 46 features, load selected model, predict.
30. **What are model score bars?** The classifier's estimated class probabilities, not certainty about genuine feelings.
31. **Can it identify anyone's true emotion?** No; voice is context dependent and the model was trained on acted examples.
32. **What would improve it?** More diverse spontaneous speech, more speakers, noise testing and group cross-validation.

Before the viva, all three members should inspect the *actual* dataset audit, comparison CSV and confusion matrices, then add the observed model name and numbers to this guide and the report.
