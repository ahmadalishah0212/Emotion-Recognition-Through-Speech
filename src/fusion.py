"""Transparent experimental late fusion over shared expression labels."""

SHARED = ("neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised")


def experimental_fusion(voice, face):
    """Symmetric mean after conditioning each distribution on shared labels.

    This is a display baseline, not a calibrated or validated multimodal model.
    Calm (voice) and contempt (face) have no counterpart and are excluded.
    """
    if voice is None or face is None:
        return None
    left = voice["probabilities"]
    right = face["probabilities"]
    left_total = sum(left[label] for label in SHARED)
    right_total = sum(right[label] for label in SHARED)
    if left_total <= 0 or right_total <= 0:
        return None
    scores = {label: 0.5 * (left[label] / left_total + right[label] / right_total) for label in SHARED}
    return {"emotion": max(scores, key=scores.get), "probabilities": scores,
            "excluded_voice_mass": left.get("calm", 0), "excluded_face_mass": right.get("contempt", 0)}
