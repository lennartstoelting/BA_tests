import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    file_name = "imu_test3.csv"

    try:
        imu_data = pd.read_csv(file_name, engine="python")

        columns_to_drop = [
            "recording id",
            "roll [deg]",
            "pitch [deg]",
            "yaw [deg]",
            "quaternion x",
            "quaternion y",
            "quaternion z",
            "quaternion w",
        ]
        imu_data = imu_data.drop(columns_to_drop, axis=1, errors="ignore")

        # Initialize our filter instance (500 rows is roughly 5 seconds of data, we sample at somewhere between 100 and 110Hz)
        imu_filter = RealTimeIMUFilter(calibration_duration_samples=500)

        print("Starting real-time simulation...\n")

        previous_time = imu_data["timestamp [ns]"].iloc[0]

        for index, row in imu_data.iterrows():
            current_time = row["timestamp [ns]"]

            gyro_sample = np.array(
                [row["gyro x [deg/s]"], row["gyro y [deg/s]"], row["gyro z [deg/s]"]]
            )
            accel_sample = np.array(
                [
                    row["acceleration x [g]"],
                    row["acceleration y [g]"],
                    row["acceleration z [g]"],
                ]
            )

            output = imu_filter.process_new_sample(
                gyro_sample, accel_sample, previous_time, current_time
            )

            previous_time = current_time

            # Once calibration is done, we can start printing EKF outputs
            # if imu_filter.is_calibrated and index % 100 == 0:
            #     print(f"Row {index}: {output}")

    except FileNotFoundError:
        print(f"Error: Could not find '{file_name}'.")


class RealTimeIMUFilter:

    def __init__(self, calibration_duration_samples=200):
        self.cal_total_samples = calibration_duration_samples
        self.is_calibrated = False

        # 2. Buffers to hold streaming calibration data
        self.accel_cal_buffer = []

        self.state_x = None  # Will hold position, velocity, quaternion, and biases

    def process_new_sample(self, gyro, accel, previous_time_ns, current_time_ns):
        # one row of data at a time, like a real-time loop
        if not self.is_calibrated:
            self._load_calibration(gyro, accel)
            return None

        dt = (current_time_ns - previous_time_ns) / 1e9
        gyro_rad_s = gyro * (np.pi / 180.0)

        qw, qx, qy, qz = self.state_x[0:4]

        dw = 1.0
        dx = 0.5 * gyro_rad_s[0] * dt
        dy = 0.5 * gyro_rad_s[1] * dt
        dz = 0.5 * gyro_rad_s[2] * dt

        w_new = qw * dw - qx * dx - qy * dy - qz * dz
        x_new = qw * dx + qx * dw + qy * dz - qz * dy
        y_new = qw * dy - qx * dz + qy * dw + qz * dx
        z_new = qw * dz + qx * dy - qy * dx + qz * dw

        mag = np.sqrt(w_new**2 + x_new**2 + y_new**2 + z_new**2)
        self.state_x[0:4] = [w_new / mag, x_new / mag, y_new / mag, z_new / mag]
        print(self.state_x[0:4])
        # EKF for later
        # return self.state_x
        return "EKF running..."

    def _load_calibration(self, gyro, accel):
        self.accel_cal_buffer.append(accel)

        current_count = len(self.accel_cal_buffer)

        # temp progress printing every 100 steps
        if current_count % 100 == 0:
            print(
                f"Calibrating... gathered {current_count}/{self.cal_total_samples} samples."
            )

        if current_count >= self.cal_total_samples:
            self._finalize_calibration()

    def _finalize_calibration(self):
        accel_array = np.array(self.accel_cal_buffer)

        # Calculate Initial Tilt
        avg_accel = np.mean(accel_array, axis=0)

        print("\n---")
        print(
            f"Average Accel Gravity Vector: {avg_accel[0]:.4f}, {avg_accel[1]:.4f}, {avg_accel[2]:.4f}"
        )

        # calculate pitch, yaw and roll
        yaw = 0.0
        pitch = np.arctan2(avg_accel[1], avg_accel[2])
        roll = np.arctan2(-avg_accel[0], np.sqrt(avg_accel[1] ** 2 + avg_accel[2] ** 2))

        print("\n---")
        print(f"Yaw: {yaw}, \nPitch: {pitch},\nRoll: {roll}")

        # calculate quaternions
        print("\n---")
        r = R.from_euler("zyx", [yaw, roll, pitch], degrees=False)
        # print(f"rotation as quat {r.as_quat()}")
        # print(f"rotation as euler {r.as_euler('zyx', degrees=True)}")

        scipy_quat = r.as_quat()
        qx, qy, qz, qw = scipy_quat[0], scipy_quat[1], scipy_quat[2], scipy_quat[3]

        # Initialize our 16-State Vector here next...
        self.state_x = np.array(
            [
                qw,
                qx,
                qy,
                qz,  # Orientation quaternion
                0.0,
                0.0,
                0.0,  # Initial velocities (vx, vy, vz)
                0.0,
                0.0,
                0.0,  # Initial positions  (x, y, z)
            ]
        )

        # Free up memory by clearing buffers
        self.accel_cal_buffer = []
        self.is_calibrated = True


if __name__ == "__main__":
    main()
