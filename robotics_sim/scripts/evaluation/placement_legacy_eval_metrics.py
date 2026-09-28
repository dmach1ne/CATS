"""Old release/hold success and orientation-aware cube height metrics."""
import numpy as np


def update_old_success(frame, hold, dt):
    released = (abs(frame['height_m'] - .0505) <= .005 + 1e-9
                and frame['table_contact'] and frame['width_m'] >= .06 - 1e-9
                and not frame['left_finger_contact'] and not frame['right_finger_contact'])
    thresholds = np.array([.02, .05, .10])
    hold = np.where(released & (frame['xy_error_m'] <= thresholds + 1e-9), hold + dt, 0.)
    for i, key in enumerate(('success_2cm_1s', 'success_5cm_1s', 'success_10cm_1s')):
        frame[key] = bool(hold[i] >= 1 - 1e-9)
    return hold


def cube_bottom_above_table(center_z, quat_wxyz, side=.049, table_z=.026):
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    z_row = np.array([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
    return float(center_z - side / 2 * np.abs(z_row).sum() - table_z)
