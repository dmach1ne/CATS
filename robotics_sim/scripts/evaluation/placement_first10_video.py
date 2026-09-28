"""Persist evaluated frames: policy input, high-res view and arm7/gripper trace."""
from pathlib import Path
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from high_res_video import compose_policy_input_high_res, compose_action_trace, ACTION_TRACE_HEIGHT


def save_first10_video(path, frames, actions, boundaries):
    path = Path(path)
    # Observations keep float32 pixels; only video copies become uint8.
    frames = [(np.ascontiguousarray(np.clip(np.rint(policy), 0, 255), dtype=np.uint8),
               np.ascontiguousarray(full, dtype=np.uint8)) for policy, full in frames]
    values = np.asarray(actions)
    np.savez_compressed(path.with_suffix('.actions.npz'), actions=values,
                        trial_boundaries=np.asarray(boundaries), policy_rate_hz=30.)
    writers = {}
    try:
        for name, image in [('policy_input', frames[0][0]), ('high_res', frames[0][1]),
                            ('video', compose_action_trace(compose_policy_input_high_res(*frames[0]), values, 0))]:
            h, w = image.shape[:2]
            writer = imageio_ffmpeg.write_frames(str(path.with_name(path.stem + '.' + name + '.mp4')),
                       (w, h), fps=30, macro_block_size=1, quality=8, ffmpeg_log_level='error')
            writer.send(None)
            writers[name] = writer
        for index, (policy, full) in enumerate(frames):
            canvas = compose_policy_input_high_res(policy, full)
            top_h, width = canvas.shape[:2]
            canvas = Image.fromarray(compose_action_trace(canvas, values, index,
                          action_labels=('J1','J2','J3','J4','J5','J6','J7','GRIPPER')))
            draw = ImageDraw.Draw(canvas)
            for boundary in boundaries[:-1]:
                x = 66 + (width - 18 - 66) * boundary / max(len(values)-1, 1)
                draw.line((x, top_h+60, x, top_h+ACTION_TRACE_HEIGHT-26), fill=(140,140,140), width=1)
            writers['video'].send(np.asarray(canvas).tobytes())
            writers['policy_input'].send(policy.tobytes())
            writers['high_res'].send(full.tobytes())
    finally:
        for writer in writers.values():
            writer.close()
