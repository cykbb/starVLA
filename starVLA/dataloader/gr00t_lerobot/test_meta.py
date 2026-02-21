import json

info_data = {
    "features": {
        "observation.images.image": {"dtype": "video"},
        "observation.state": {"dtype": "float32", "shape": [8]},
        "action": {"dtype": "float32", "shape": [7]},
        "segmentation.agentview": {"dtype": "string"}
    }
}

state_meta = {}
action_meta = {}
video_meta = {}
annotation_meta = {}

features = info_data.get("features", {})
for key, feat in features.items():
    if feat.get("dtype", "") == "video":
        video_meta[key] = {"original_key": key}
    elif key.startswith("observation.state"):
        shape = feat.get("shape", [1])
        state_meta[key] = {
            "start": 0, "end": shape[0], "absolute": False, 
            "rotation_type": "axis_angle", "dtype": feat.get("dtype", "float32"),
            "original_key": key
        }
    elif key.startswith("action"):
        shape = feat.get("shape", [1])
        action_meta[key] = {
            "start": 0, "end": shape[0], "absolute": False, 
            "rotation_type": "axis_angle", "dtype": feat.get("dtype", "float32"),
            "original_key": key
        }
    elif key.startswith("annotation"):
        annotation_meta[key] = {"original_key": key}

final_dict = {
    "state": state_meta,
    "action": action_meta,
    "video": video_meta,
    "annotation": annotation_meta if annotation_meta else None
}

from schema import LeRobotModalityMetadata
meta = LeRobotModalityMetadata.model_validate(final_dict)
print("SUCCESS!", meta)
