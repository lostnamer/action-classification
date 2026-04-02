import json
import numpy as np
import pickle
import os
import pandas as pd
from tqdm import tqdm


def format_acar_dets(opts):
    """Reads ACAR prediction CSV and writes per-frame .pkl files in the format
    that the Post-Processing evaluation pipeline expects.

    CSV format (9 columns, no header):
        video, frame, xmin, ymin, xmax, ymax, action, confidence, tube_id
    where:
        - xmin/ymin/xmax/ymax are in [0,1] normalized coordinates
        - action is 0-indexed (0..21 for 22 ACAR classes)

    MAGIC NUMBERS 512 and 682:
    RetinaNet input dimensions are 512x682.  The evaluation code has these
    hardcoded, so we scale normalized coords to pixel space here.

    Output pkl format per frame:
        {'main': ndarray(N, 23)}
        columns 0-3  : bbox [x1, y1, x2, y2] in pixel space (682 x 512)
        columns 4-22 : 19 ROAD action class confidence scores

    Parameters
    ----------
    opts : EasyDict
        Must contain:
            opts.prediction_path   – path to predict_epoch_*.csv
            opts.save_pickles_path – directory to write per-video/per-frame pkls
    """
    # ACAR trains on 22 classes; ROAD evaluation uses 19 classes.
    # Maps ACAR class index (0-indexed, length 22) → ROAD class index (-1 = skip).
    #   Skipped: Rev(6), MovRht(14), MovLft(15)
    action_class_selection_map = [0, 1, 2, 3, 4, 5, -1, 6, 7, 8, 9, 10, 11, 12, -1, -1, 13, 14, 15, 16, 17, 18]

    print(f"Formatting predictions from {opts.prediction_path}")

    predictions_df = pd.read_csv(opts.prediction_path, header=None)
    predictions_df.columns = ["video", "frame", "xmin", "ymin", "xmax", "ymax", "action", "confidence", "tube_id"]

    # Group all rows by (video, frame, bbox) to reconstruct per-box confidence vectors
    formatted_predictions = {}
    for _, row in tqdm(predictions_df.iterrows(), total=len(predictions_df)):
        vid = row['video']
        frm = int(row['frame'])
        box_hash = f"{row['xmin']}_{row['ymin']}_{row['xmax']}_{row['ymax']}"

        if vid not in formatted_predictions:
            formatted_predictions[vid] = {}
        if frm not in formatted_predictions[vid]:
            formatted_predictions[vid][frm] = {}
        if box_hash not in formatted_predictions[vid][frm]:
            formatted_predictions[vid][frm][box_hash] = {
                'bbox': [float(row['xmin']) * 682,
                         float(row['ymin']) * 512,
                         float(row['xmax']) * 682,
                         float(row['ymax']) * 512],
                'confidences': [0.0] * 19
            }

        action_acar = int(row['action'])   # 0-indexed, range 0..21
        road_idx = action_class_selection_map[action_acar]
        if road_idx < 0:
            continue
        formatted_predictions[vid][frm][box_hash]['confidences'][road_idx] = float(row['confidence'])

    if not os.path.exists(opts.save_pickles_path):
        os.makedirs(opts.save_pickles_path)

    print(f"Writing .pkl files to {opts.save_pickles_path}")

    for video_name, frames in formatted_predictions.items():
        vid_dir = os.path.join(opts.save_pickles_path, video_name)
        if not os.path.exists(vid_dir):
            os.makedirs(vid_dir)

        for frame_num, boxes in frames.items():
            n = len(boxes)
            save_data = np.zeros((n, 23), dtype=np.float32)
            for i, box_data in enumerate(boxes.values()):
                save_data[i, 0:4] = box_data['bbox']
                save_data[i, 4:23] = box_data['confidences']

            pkl_path = os.path.join(vid_dir, "%05d.pkl" % frame_num)
            with open(pkl_path, 'wb') as f:
                pickle.dump({'main': save_data}, f)

    print("Done!")

