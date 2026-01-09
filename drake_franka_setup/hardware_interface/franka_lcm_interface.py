"""
FrankaStation: High-level hardware interface using Drake diagrams.

This class provides a clean API for controlling the Franka robot hardware,
similar to the IIWAStation pattern. It uses Drake's simulator and diagram
architecture for proper timing and control.
"""

import numpy as np
from typing import List, Optional, Union, TYPE_CHECKING

import pydrake.all as pd
import pybezier as pb
from drake_franka_setup.planning.utils import (
    composite_bezier_to_drake,
    add_stationary_start_and_end
)
from drake import lcmt_panda_command, lcmt_panda_status, lcmt_schunk_wsg_command, lcmt_schunk_wsg_status
from manipulation.station import _WireDriverStatusReceiversToToPose

from drake_franka_setup.hardware_interface.franka_params import FRANKA_MAX_Q, FRANKA_MIN_Q


class FrankaStatusToState(pd.LeafSystem):
    """
    Converts FrankaStatusReceiver outputs (position and velocity) to
    a single MultibodyPlant state vector [q, v].
    """
    def __init__(self, num_positions: int, num_velocities: int):
        pd.LeafSystem.__init__(self)
        self.num_positions = num_positions
        self.num_velocities = num_velocities

        # Input ports for position and velocity from FrankaStatusReceiver
        self.position_input = self.DeclareVectorInputPort(
            "position", pd.BasicVector(num_positions)
        )
        self.velocity_input = self.DeclareVectorInputPort(
            "velocity", pd.BasicVector(num_velocities)
        )

        # Output port for combined state [q, v]
        self.DeclareVectorOutputPort(
            "state",
            pd.BasicVector(num_positions + num_velocities),
            self.CalcStateOutput
        )

    def CalcStateOutput(self, context, output):
        """Combine position and velocity into state vector."""
        q = self.position_input.Eval(context)
        v = self.velocity_input.Eval(context)
        output.SetFromVector(np.hstack([q, v]))


class FrankaLCMInterface:
    """
    High-level interface for Franka robot using Drake diagrams.

    This class wraps Drake's LCM communication systems and provides a simple API
    for trajectory execution, similar to the IIWAStation pattern used with the
    iiwa robots.

    Example:
        station = FrankaLCMInterface()
        q_current = station.get_config()
        station.goto_config(q_goal, duration=2.0)
        station.play_trajectory(composite_trajectory)
        station.close()
    """

    def __init__(self,
                 control_mode: str = "position",
                 lcm_url: str = "",
                 enable_logging: bool = False,
                 control_gripper: bool = False):
        """
        Initialize the FrankaStation.

        Args:
            control_mode: "position", "position_velocity", or "torque"
            lcm_url: LCM URL (empty string uses default)
            enable_logging: If True, log commanded and actual positions/velocities
            control_gripper: If True, enable gripper control methods
        """
        self.control_mode = control_mode
        self.lcm_url = lcm_url
        self.enable_logging = enable_logging
        self.control_gripper = control_gripper
        self.domain = pd.HPolyhedron.MakeBox(FRANKA_MIN_Q, FRANKA_MAX_Q)

        # Read boot configuration from LCM FRANKA_STATUS
        self.boot_config = self._read_boot_config_from_lcm()
        print(f"Boot configuration: {self.boot_config}")

        if control_mode == "position":
            self._control_mode_enum = pd.PandaControlMode.kPosition
        elif control_mode == "position_velocity":
            #raise NotImplementedError(" torque mode not implemented yet")
            # Combined mode: send both position and velocity for better tracking
            self._control_mode_enum = pd.PandaControlMode.kPosition | pd.PandaControlMode.kVelocity
        elif control_mode == "torque":
            raise NotImplementedError(" torque mode not implemented yet")
            self._control_mode_enum = pd.PandaControlMode.kTorque
        else:
            raise ValueError(f"Invalid control mode: {control_mode}")

        # Build the Drake diagram
        self._build_diagram()

        # Create simulator
        self.simulator = pd.Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)

        # Set max step size to match command rate (5ms for 200Hz)
        # This ensures commands are published frequently enough
        self.simulator.get_mutable_integrator().set_maximum_step_size(0.005)

        # Get contexts
        self.diagram_context = self.simulator.get_mutable_context()

        # Setup gripper control if enabled (outside diagram - independent from arm)
        self._gripper_subscriber = None
        if self.control_gripper:
            from pydrake.lcm import Subscriber
            self._gripper_subscriber = Subscriber(
                self.drake_lcm, "PANDA_HAND_STATUS", lcmt_schunk_wsg_status
            )
            print(f"✅ Gripper control enabled")
            print(f"   Hand Status Channel: PANDA_HAND_STATUS")
            print(f"   Hand Command Channel: PANDA_HAND_COMMAND")

        print(f"✅ FrankaStation initialized (control_mode={control_mode})")
        print(f"   LCM Status Channel: PANDA_STATUS")
        print(f"   LCM Command Channel: PANDA_COMMAND")

    def _read_boot_config_from_lcm(self, timeout: float = 5.0) -> np.ndarray:
        """
        Read the current robot configuration from LCM PANDA_STATUS messages.

        This is used during initialization to get the boot configuration without
        creating a FrankaHardwareInterface (which would conflict with the driver).

        Args:
            timeout: Maximum time to wait for messages [s]

        Returns:
            np.ndarray: Current joint positions [rad] (7,)
        """
        from pydrake.lcm import DrakeLcm, Subscriber
        import time

        print(f"Reading boot configuration from PANDA_STATUS LCM...")

        lcm = DrakeLcm(self.lcm_url)
        subscriber = Subscriber(lcm, "PANDA_STATUS", lcmt_panda_status)

        start = time.time()
        configs = []

        # Collect a few messages to ensure we have a valid reading
        while time.time() - start < timeout and len(configs) < 5:
            lcm.HandleSubscriptions(100)  # 100ms timeout
            msg = subscriber.message
            if msg is not None and msg.num_joints == 7:
                config = np.array(msg.joint_position)

                # Sanity check: ensure we're getting real robot data, not zeros
                # Real robot joints should have non-zero values (especially joint 4 which is typically ~-2.3 rad)
                if np.allclose(config, 0.0, atol=1e-6):
                    print(f"  ⚠️  Received all-zero config, skipping (likely uninitialized data)")
                    continue

                # Additional sanity: check if values are within reasonable joint limits

                if not self.domain.PointInSet(config):
                    print(f"  ⚠️  Received out-of-range config: {config.round(4)}, skipping")
                    continue

                configs.append(config)
                print(f"  ✓ Valid config {len(configs)}/5: {config.round(4)}")

        if len(configs) == 0:
            raise RuntimeError(
                f"Failed to read valid boot configuration from LCM in {timeout}s. "
                "Is the driver running and publishing PANDA_STATUS with real robot data?"
            )

        # Average the configs to smooth out any noise
        boot_config = np.mean(configs, axis=0)
        print(f"  Boot config (averaged): {boot_config.round(4)}")

        return boot_config

    def _build_diagram(self):
        """Build the Drake diagram with LCM communication systems."""
        builder = pd.DiagramBuilder()

        # Create LCM interface
        from pydrake.lcm import DrakeLcm
        self.drake_lcm = DrakeLcm(self.lcm_url)

        # Add LcmInterfaceSystem to handle message pumping during simulation
        lcm_interface = builder.AddSystem(pd.LcmInterfaceSystem(self.drake_lcm))

        self.status_receiver = builder.AddNamedSystem("franka1.status_receiver",
            pd.PandaStatusReceiver(num_joints=7)
        )

        status_subscriber = builder.AddNamedSystem("franka1.status_subscriber",
            pd.LcmSubscriberSystem.Make(
                channel="PANDA_STATUS",
                lcm_type=lcmt_panda_status,
                lcm=self.drake_lcm,
                use_cpp_serializer=True,
                wait_for_message_on_initialization_timeout=10.0  # Wait for first message!
            )
        )

        builder.Connect(
            status_subscriber.get_output_port(),
            self.status_receiver.get_input_port()
        )

        # This thing poupulates the lcm message
        self.command_sender = builder.AddNamedSystem("franka1.command_sender",
            pd.PandaCommandSender(
                num_joints=7,
                control_mode=self._control_mode_enum
            )
        )

        # This thing sends the lcm command
        command_publisher = builder.AddNamedSystem("franka1.command_publisher",
            pd.LcmPublisherSystem.Make(
                channel="PANDA_COMMAND",
                lcm_type=lcmt_panda_command,
                lcm=self.drake_lcm,
                use_cpp_serializer=True,
                publish_period=0.005  # 200 Hz
            )
        )

        builder.Connect(
            self.command_sender.get_output_port(),
            command_publisher.get_output_port()
        )

        # boot config is read out during initialization,
        # if not populated correctly throws an error
        q_zero = self.boot_config.copy()
        zero_traj = pd.BezierCurve(
            start_time=0.0,
            end_time=0.1,
            control_points=np.column_stack([q_zero, q_zero])
        )

        # For position+velocity mode, we need to output derivatives
        # Otherwise just output positions (derivative_order=0)
        if self.control_mode == "position_velocity":
            output_derivative_order = 1  # Output position and velocity
        else:
            output_derivative_order = 0  # Output position only

        self.traj_source = builder.AddSystem(
            pd.TrajectorySource(trajectory=zero_traj, output_derivative_order=output_derivative_order)
        )

        # Connect trajectory source to command sender
        if self.control_mode == "position_velocity":
            # When output_derivative_order=1, TrajectorySource outputs [position; velocity] as single vector
            # Use Demultiplexer to split into separate position and velocity vectors
            demux = builder.AddSystem(pd.Demultiplexer(output_ports_sizes=[7, 7]))
            builder.Connect(
                self.traj_source.get_output_port(0),
                demux.get_input_port(0)
            )
            builder.Connect(
                demux.get_output_port(0),  # Position output
                self.command_sender.get_position_input_port()
            )
            builder.Connect(
                demux.get_output_port(1),  # Velocity output
                self.command_sender.get_velocity_input_port()
            )
        else:
            # For other modes, just connect position
            builder.Connect(
                self.traj_source.get_output_port(0),  # Position output
                self.command_sender.get_position_input_port()
            )

        # Add logging systems if enabled
        self.position_command_logger = None
        self.velocity_command_logger = None
        self.position_status_logger = None
        self.velocity_status_logger = None
        self.last_executed_trajectory = None  # Store trajectory that was actually executed

        if self.enable_logging:
            # Log commanded positions
            self.position_command_logger = builder.AddSystem(
                pd.VectorLogSink(7)  # 7 joints
            )

            if self.control_mode == "position_velocity":
                # Connect from demux outputs (already split)
                builder.Connect(
                    demux.get_output_port(0),  # Position output from demux
                    self.position_command_logger.get_input_port()
                )

                # Log commanded velocities
                self.velocity_command_logger = builder.AddSystem(
                    pd.VectorLogSink(7)
                )
                builder.Connect(
                    demux.get_output_port(1),  # Velocity output from demux
                    self.velocity_command_logger.get_input_port()
                )
            else:
                # Connect directly from trajectory source (position only)
                builder.Connect(
                    self.traj_source.get_output_port(0),
                    self.position_command_logger.get_input_port()
                )

            # Log actual positions (from status receiver)
            self.position_status_logger = builder.AddSystem(
                pd.VectorLogSink(7)
            )
            builder.Connect(
                self.status_receiver.get_position_measured_output_port(),
                self.position_status_logger.get_input_port()
            )

            # Log actual velocities (from status receiver)
            self.velocity_status_logger = builder.AddSystem(
                pd.VectorLogSink(7)
            )
            builder.Connect(
                self.status_receiver.get_velocity_measured_output_port(),
                self.velocity_status_logger.get_input_port()
            )


        self.diagram = builder.Build()


        self._status_receiver_name = self.status_receiver.get_name()
        self._command_sender_name = self.command_sender.get_name()
        self._traj_source_name = self.traj_source.get_name()

    def get_config(self) -> np.ndarray:
        """
        Get the current joint configuration.

        This reads directly from LCM without advancing the simulator,
        so it won't trigger any command publishing.

        Returns:
            np.ndarray: Current joint positions [rad] (7,)
        """
        from pydrake.lcm import Subscriber
        import time

        subscriber = Subscriber(self.drake_lcm, "PANDA_STATUS", lcmt_panda_status)

        # Wait for a fresh message (timeout 500ms)
        start = time.time()
        latest_msg = None
        while time.time() - start < 0.5:
            self.drake_lcm.HandleSubscriptions(50)  # 50ms timeout
            msg = subscriber.message
            if msg is not None and msg.num_joints == 7:
                config = np.array(msg.joint_position)
                # Sanity check
                if not np.allclose(config, 0.0, atol=1e-6) and np.all(np.abs(config) <= 3.0):
                    return config
                latest_msg = msg

        # Fallback: return last message even if sanity checks failed
        if latest_msg is not None:
            return np.array(latest_msg.joint_position)

        raise RuntimeError("Failed to read robot configuration from LCM")

    def goto_config(self, configuration: np.ndarray, duration: Optional[float] = None):
        """
        Move to a target joint configuration using a smooth Bezier trajectory.

        Args:
            configuration: Target joint positions [rad] (7,)
            duration: Optional duration [s]. If None, computed from distance.
        """
        configuration = np.asarray(configuration).squeeze()
        if configuration.shape != (7,):
            raise ValueError(f"Expected configuration shape (7,), got {configuration.shape}")

        # Reset simulator time
        self.diagram_context.SetTime(0.0)
        self.simulator.Initialize()

        # Get current configuration
        current_config = self.get_config()

        # Check if already at target
        dist = np.linalg.norm(current_config - configuration)
        if dist <= 1e-3:
            print(f"Already at target configuration (dist={dist:.6f} rad)")
            return

        # Compute duration if not specified (0.9 rad/s average speed)
        if duration is None:
            duration = np.max([0.5, dist / 0.9])

        print(f"Moving to target configuration (dist={dist:.3f} rad, duration={duration:.2f}s)")

        # Create smooth Bezier curve from current to target
        # Use 5 control points for smooth motion
        control_points = np.column_stack([
            current_config,
            current_config,
            current_config,
            (current_config + configuration) / 2,  # midpoint
            configuration,
            configuration,
            configuration
        ])

        traj = pd.BezierCurve(
            start_time=0.0,
            end_time=duration,
            control_points=control_points
        )

        # Execute trajectory
        self._execute_trajectory(traj)

    def play_trajectory(self, trajectory: Union[pd.CompositeTrajectory, pb.CompositeBezierCurve],
                        add_stationary_bookends: bool = True,
                        start_duration: float = 0.05,
                        end_duration: float = 0.05):
        """
        Execute a Drake CompositeTrajectory on the robot.

        Args:
            trajectory: Drake CompositeTrajectory to execute
            add_stationary_bookends: If True, add stationary segments at start and end to prevent lurching (default: True)
            start_duration: Duration of starting stationary segment in seconds (default: 0.05s)
            end_duration: Duration of ending stationary segment in seconds (default: 0.05s)
        """
        if isinstance(trajectory, pb.CompositeBezierCurve):
            trajectory: pd.CompositeTrajectory = composite_bezier_to_drake(trajectory)

        # Add stationary segments at start and end to prevent lurching
        if add_stationary_bookends:
            trajectory = add_stationary_start_and_end(trajectory, start_duration, end_duration)
            print(f"Playing trajectory (duration={trajectory.end_time():.2f}s, with {start_duration:.2f}s start + {end_duration:.2f}s end holds)")
        else:
            print(f"Playing trajectory (duration={trajectory.end_time():.2f}s)")

        # Reset simulator time
        self.diagram_context.SetTime(0.0)
        self.simulator.Initialize()
        context_time = self.diagram_context.get_time()

        # Time-shift trajectory if needed to align with context time
        if abs(trajectory.start_time() - context_time) > 1e-6:
            time_shift = context_time - trajectory.start_time()

            # Rebuild trajectory with shifted times
            segments: List[pd.Trajectory] = [
                trajectory.segment(i)
                for i in range(trajectory.get_number_of_segments())
            ]

            shifted_segments: List[pd.BezierCurve] = []
            for seg in segments:
                if isinstance(seg, pd.BezierCurve):
                    shifted_segments.append(
                        pd.BezierCurve(
                            start_time=seg.start_time() + time_shift,
                            end_time=seg.end_time() + time_shift,
                            control_points=seg.control_points()
                        )
                    )

            trajectory = pd.CompositeTrajectory(shifted_segments)

        # Execute trajectory
        self._execute_trajectory(trajectory)

    def _execute_trajectory(self, trajectory: Union[pd.BezierCurve, pd.CompositeTrajectory]):
        """
        Internal method to execute a trajectory using the simulator.

        Args:
            trajectory: Drake trajectory (BezierCurve or CompositeTrajectory)
        """
        # Store the trajectory that was actually executed (for plotting reference)
        self.last_executed_trajectory = trajectory

        # Update trajectory source
        # Note: UpdateTrajectory modifies the system's internal trajectory
        self.traj_source.UpdateTrajectory(trajectory)

        # Run simulator until trajectory ends
        # Drake handles all timing and LCM communication automatically
        self.simulator.AdvanceTo(trajectory.end_time())

        print(f"✅ Trajectory execution complete")

    def get_logs(self) -> dict:
        """
        Get logged data from the trajectory execution.

        Returns:
            dict with keys:
                - 'time': Time samples [s] (N,)
                - 'q_commanded': Commanded positions [rad] (N, 7)
                - 'v_commanded': Commanded velocities [rad/s] (N, 7) - only if position_velocity mode
                - 'q_actual': Actual positions [rad] (N, 7)
                - 'v_actual': Actual velocities [rad/s] (N, 7)
                - 'q_error': Position tracking error [rad] (N, 7)
                - 'v_error': Velocity tracking error [rad/s] (N, 7) - only if position_velocity mode
        """
        if not self.enable_logging:
            raise RuntimeError("Logging was not enabled. Set enable_logging=True in constructor.")

        # Get the diagram context to access logger contexts
        position_cmd_log = self.position_command_logger.FindLog(self.diagram_context)
        position_status_log = self.position_status_logger.FindLog(self.diagram_context)
        velocity_status_log = self.velocity_status_logger.FindLog(self.diagram_context)

        # Extract data, skipping first sample to avoid initialization transient
        time = position_cmd_log.sample_times()[1:]  # Skip first sample
        q_commanded = position_cmd_log.data().T[1:]  # Transpose to (N, 7) and skip first

        # Get commanded velocities if available
        v_commanded = None
        if self.velocity_command_logger is not None:
            velocity_cmd_log = self.velocity_command_logger.FindLog(self.diagram_context)
            v_commanded = velocity_cmd_log.data().T[1:]  # Skip first sample

        # Status logs might have different sample times due to LCM timing
        # We'll use the command log times as reference
        time_status = position_status_log.sample_times()[1:]  # Skip first sample
        q_actual = position_status_log.data().T[1:]  # Skip first sample
        v_actual = velocity_status_log.data().T[1:]  # Skip first sample

        # Compute tracking error
        # Note: We need to interpolate if sample times don't match exactly
        if len(time_status) != len(time) or not np.allclose(time_status, time):
            print(f"⚠️  Command and status logs have different sample times.")
            print(f"   Command samples: {len(time)}, Status samples: {len(time_status)}")
            # For simplicity, just use status log timing
            time = time_status
            # Interpolate commanded positions to status times
            q_commanded_interp = np.zeros((len(time_status), 7))
            for i in range(7):
                q_commanded_interp[:, i] = np.interp(
                    time_status,
                    position_cmd_log.sample_times()[1:],  # Skip first sample in source too
                    position_cmd_log.data()[i, 1:]  # Skip first sample
                )
            q_commanded = q_commanded_interp

            # Interpolate commanded velocities if available
            if v_commanded is not None:
                v_commanded_interp = np.zeros((len(time_status), 7))
                for i in range(7):
                    v_commanded_interp[:, i] = np.interp(
                        time_status,
                        velocity_cmd_log.sample_times()[1:],
                        velocity_cmd_log.data()[i, 1:]
                    )
                v_commanded = v_commanded_interp

        q_error = q_actual - q_commanded

        result = {
            'time': time,
            'q_commanded': q_commanded,
            'q_actual': q_actual,
            'v_actual': v_actual,
            'q_error': q_error,
        }

        # Add velocity data if available
        if v_commanded is not None:
            result['v_commanded'] = v_commanded
            result['v_error'] = v_actual - v_commanded

        return result

    def save_logs(self, filepath: str):
        """
        Save logged data to a numpy file.

        Args:
            filepath: Path to save the logs (will save as .npz)
        """
        logs = self.get_logs()
        np.savez(filepath, **logs)
        print(f"✅ Logs saved to {filepath}")

    def clear_logs(self):
        """
        Clear all logged data.

        This removes all samples from the VectorLogSinks, allowing you to start
        fresh logging for a new trajectory.
        """
        if not self.enable_logging:
            return

        # Access and clear each log
        position_cmd_log = self.position_command_logger.FindLog(self.diagram_context)
        position_status_log = self.position_status_logger.FindLog(self.diagram_context)
        velocity_status_log = self.velocity_status_logger.FindLog(self.diagram_context)

        position_cmd_log.Clear()
        position_status_log.Clear()
        velocity_status_log.Clear()

        # Clear velocity command log if it exists
        if self.velocity_command_logger is not None:
            velocity_cmd_log = self.velocity_command_logger.FindLog(self.diagram_context)
            velocity_cmd_log.Clear()

        print("🗑️  Logs cleared")

    def plot_logs(self, station: PandaStation, reference_traj: Optional[Union[pd.CompositeTrajectory, pb.CompositeBezierCurve]] = None,
                  save_path: str = 'tmp/trajectory_tracking_plot.png'):
        """
        Plot logged trajectory tracking data including joint space and Cartesian tracking.

        Args:
            station: PandaStation instance for forward kinematics
            reference_traj: Optional reference trajectory to plot alongside commanded trajectory.
                          If None, uses the last executed trajectory (with bookends) automatically.
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt

        # Use last executed trajectory by default (includes bookends if they were added)
        if reference_traj is None:
            reference_traj = self.last_executed_trajectory

        if isinstance(reference_traj, pb.CompositeBezierCurve):
            reference_traj = composite_bezier_to_drake(reference_traj)

        # Get logged data
        logs = self.get_logs()
        time = logs['time']
        q_commanded = logs['q_commanded']
        q_actual = logs['q_actual']
        v_actual = logs['v_actual']
        q_error = logs['q_error']
        v_commanded = logs.get('v_commanded', None)  # May be None if not in position_velocity mode

        # Compute end-effector positions
        pos_commanded = station.get_endeff_positions(q_commanded)
        pos_actual = station.get_endeff_positions(q_actual)
        pos_error = pos_actual - pos_commanded

        # Create plots with 5 subplots
        fig, axes = plt.subplots(5, 1, figsize=(12, 16))

        # Plot 1: Joint Positions
        ax = axes[0]
        for i in range(7):
            ax.plot(time, q_commanded[:, i], '--', label=f'q{i}_cmd', alpha=0.7)
            ax.plot(time, q_actual[:, i], '-', label=f'q{i}_actual')
        ax.set_ylabel('Position [rad]')
        ax.set_title('Joint Positions: Commanded vs Actual')
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True)

        # Plot 2: Joint Velocities
        ax = axes[1]
        if v_commanded is not None:
            # Show commanded and actual velocities
            for i in range(7):
                ax.plot(time, v_commanded[:, i], '--', label=f'v{i}_cmd', alpha=0.7)
                ax.plot(time, v_actual[:, i], '-', label=f'v{i}_actual')
            ax.set_title('Joint Velocities: Commanded vs Actual')
        else:
            # Show only actual velocities (no feedforward)
            for i in range(7):
                ax.plot(time, v_actual[:, i], '-', label=f'v{i}_actual')
            ax.set_title('Joint Velocities (Actual only - no feedforward)')
        ax.set_ylabel('Velocity [rad/s]')
        ax.legend(ncol=7, fontsize=8)
        ax.grid(True)
        ax.axhline(y=2.62, color='r', linestyle='--', alpha=0.5, label='Velocity limit')
        ax.axhline(y=-2.62, color='r', linestyle='--', alpha=0.5)

        # Plot 3: Joint Tracking Errors
        ax = axes[2]
        for i in range(7):
            ax.plot(time, q_error[:, i], '-', label=f'error{i}')
        ax.set_ylabel('Position Error [rad]')
        ax.set_title('Joint Tracking Errors')
        ax.legend(ncol=7, fontsize=8)
        ax.grid(True)

        # Plot 4: Cartesian Tracking
        ax = axes[3]
        labels = ['X', 'Y', 'Z']
        colors = ['r', 'g', 'b']

        # Plot reference trajectory if provided
        if reference_traj is not None:
            t_ref = np.linspace(reference_traj.start_time(), reference_traj.end_time(), len(time))
            q_ref = reference_traj.vector_values(t_ref).T
            pos_ref = station.get_endeff_positions(q_ref)
            for i, (label, color) in enumerate(zip(labels, colors)):
                ax.plot(time, pos_ref[:, i], ':', color=color, label=f'{label}_ref', alpha=0.5, linewidth=2)

        # Plot commanded and actual
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(time, pos_commanded[:, i], '--', color=color, label=f'{label}_cmd', alpha=0.7)
            ax.plot(time, pos_actual[:, i], '-', color=color, label=f'{label}_actual')

        ax.set_ylabel('Position [m]')
        ax.set_title('End-Effector Cartesian Tracking')
        ax.legend(ncol=3, fontsize=8)
        ax.grid(True)

        # Plot 5: Cartesian Tracking Errors
        ax = axes[4]
        for i, (label, color) in enumerate(zip(labels, colors)):
            ax.plot(time, pos_error[:, i] * 1000, '-', color=color, label=f'{label}_error')

        # Plot norm of error
        error_norm = np.linalg.norm(pos_error, axis=1) * 1000
        ax.plot(time, error_norm, 'k-', linewidth=2, label='Total error', alpha=0.7)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position Error [mm]')
        ax.set_title('End-Effector Cartesian Tracking Errors')
        ax.legend(ncol=4, fontsize=8)
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.show(block = False)
        # Print statistics
        print(f"\nTracking Statistics:")
        print(f"  Max joint velocities: {np.max(np.abs(v_actual), axis=0)}")
        print(f"  Max joint errors: {np.max(np.abs(q_error), axis=0)}")
        print(f"  Max Cartesian error: {np.max(np.linalg.norm(pos_error, axis=1)):.6f} m")
        print(f"  Mean Cartesian error: {np.mean(np.linalg.norm(pos_error, axis=1)):.6f} m")

        return fig, axes

    # ========== Gripper Control Methods ==========

    def wait_for_gripper_status(self, timeout: float = 5.0):
        """
        Wait until gripper status is received.

        Args:
            timeout: Maximum time to wait [s]

        Raises:
            TimeoutError: If no gripper status received within timeout
        """
        if not self.control_gripper:
            raise RuntimeError("Gripper control not enabled. Set control_gripper=True in constructor.")

        import time
        start = time.time()

        while self._gripper_subscriber.message is None:
            self.drake_lcm.HandleSubscriptions(100)  # 100ms timeout
            if time.time() - start > timeout:
                raise TimeoutError(
                    f"Did not receive gripper status in {timeout}s. "
                    "Is the hand driver running?"
                )

    def get_gripper_width(self) -> Optional[float]:
        """
        Get current gripper width in meters.

        Returns:
            float: Current gripper width [m], or None if gripper not enabled/available
                    """
        if not self.control_gripper:
            raise RuntimeError("Gripper control not enabled. Set control_gripper=True in constructor.")

        # Handle pending messages
        self.drake_lcm.HandleSubscriptions(0)

        msg = self._gripper_subscriber.message
        if msg is None:
            return None

        # Convert from mm to m
        return msg.actual_position_mm / 1000.0

    def set_gripper_width(self, width_m: float, force: float = 40.0, wait: bool = True, timeout: float = 10.0):
        """
        Command gripper to move to target width.

        This method sends a single command and optionally waits for completion.
        The gripper driver executes the command asynchronously.

        Args:
            width_m: Target gripper width [m] (typically 0.0 to 0.08)
            force: Grasp force [N] (default 40N, only used if driver is in grasp mode)
            wait: If True, block until gripper reaches target (default True)
            timeout: Maximum time to wait [s] (default 10s)
        """
        if not self.control_gripper:
            raise RuntimeError("Gripper control not enabled. Set control_gripper=True in constructor.")

        import time

        # Clamp width to valid range
        width_m = np.clip(width_m, 0.0, 0.08)

        # Create and publish command
        cmd = lcmt_schunk_wsg_command()
        cmd.utime = int(time.time() * 1e6)
        cmd.target_position_mm = float(width_m * 1000.0)  # Convert m to mm
        cmd.force = float(force)

        self.drake_lcm.Publish("PANDA_HAND_COMMAND", cmd.encode())

        if wait:
            # Wait until gripper reaches target (within 1mm tolerance)
            start = time.time()
            while time.time() - start < timeout:
                current_width = self.get_gripper_width()
                if current_width is not None:
                    error_mm = abs((current_width - width_m) * 1000.0)
                    if error_mm < 1.0:  # Within 1mm tolerance
                        return

                time.sleep(0.05)  # Check at 20Hz

            print(f"⚠️  Warning: Gripper did not reach target width within {timeout}s timeout")

    def open_gripper(self, wait: bool = True):
        """
        Open gripper to maximum width (~80mm).

        Args:
            wait: If True, block until motion completes (default True)
        """
        self.set_gripper_width(width_m=0.08, wait=wait)

    def close_gripper(self, wait: bool = True):
        """
        Close gripper to minimum width (~0mm).

        Args:
            wait: If True, block until motion completes (default True)
        """
        self.set_gripper_width(width_m=0.0, wait=wait)

    def close(self):
        """Clean shutdown of the station."""
        print("🛑 FrankaStation closing...")
        # Simulator cleanup happens automatically
