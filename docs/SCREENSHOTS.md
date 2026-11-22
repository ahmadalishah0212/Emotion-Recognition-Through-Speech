# Actual VocalLens screenshots

Captured from the running local Streamlit app, 23 November 2026, at a consistent 1280×720 browser viewport. These are JPEG browser screenshots, not mockups. Longer workflows use separate views because Streamlit scrolls within its main panel.

| File in `screenshots/` | Demonstrates | Inference |
|---|---|---|
| 01-vocallens-home.jpg | Branding, coral navigation and voice landing | No result |
| 02-voice-results.jpg | Uploaded Happy-labeled RAVDESS audio; Angry 43.9%, with alternatives | Actual speech inference; incorrect top class |
| 03-face-photo-upload.jpg | Photo preview and analysis control | Preview only |
| 04-face-photo-results.jpg | Detected face crop and Happy score distribution | Actual face inference, Happy 99.8% |
| 05-video-analysis.jpg | Sad-labeled video preview and Neutral face result | Actual 12-frame inference; incorrect top class |
| 05b-video-results.jpg | Video score distribution and sampling metadata | Actual 12-frame inference |
| 06-multimodal-results.jpg | Separate voice and face result panels | Actual inference, disagreeing outputs |
| 06b-experimental-fusion.jpg | Disagreement explanation and combined Happy 54.4% | Actual shared-class average, experimental |
| 07-webcam.jpg | Webcam control and permission/unavailable state | No live camera capture tested |

Photo and video source: Livingstone & Russo (2018), [RAVDESS](https://zenodo.org/records/1188976), Actor 05; CC BY-NC-SA 4.0. Screenshots containing those frames are adaptations shared under CC BY-NC-SA 4.0. The photo is a sampled video frame, and the WAV is its decoded audio. Other project code retains its existing licensing. No personal uploads are included. Samples were selected by filename before inference, not for accuracy. Model scores do not establish actual feelings.

See [demo instructions](../demo/README.md) to reproduce the inputs. These screenshots document behavior and visual presentation; they are not a new accuracy benchmark. Physical camera/microphone and GPU testing remain manual.
