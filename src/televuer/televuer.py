from vuer import Vuer
from vuer.schemas import ImageBackground, Hands, MotionControllers, WebRTCVideoPlane, WebRTCStereoVideoPlane
from multiprocessing import Value, Array, Process, shared_memory
from msgpack import ExtType
import numpy as np
import asyncio
import threading
import cv2
import os
import time
from pathlib import Path
from typing import Literal


HAND_TRACKING_STATES = ("missing", "tracking", "invalid")


class HandPoseGuard:
    def __init__(self):
        self.pose = None
        self.timestamp = 0.0
        self.tracking = False
        self.status = "missing"

    def lose(self, status):
        self.tracking = False
        self.status = status

    def update(self, data, now):
        if isinstance(data, ExtType):
            # msgpackr represents JavaScript undefined as fixext1 type 0, byte 0.
            if data.code == 0 and data.data == b"\x00":
                data = None
            else:
                raise ValueError(f"unsupported hand extension code={data.code}, bytes={len(data.data)}")
        if data is None or isinstance(data, (list, tuple)) and len(data) == 0:
            self.lose("missing")
            return None
        flat = np.asarray(data, dtype=float)
        if flat.shape != (400,) or not np.isfinite(flat).all():
            raise ValueError("hand poses must contain 400 finite numbers")
        matrices = flat.reshape(25, 4, 4).transpose(0, 2, 1)
        rotations = matrices[:, :3, :3]
        if (not np.allclose(matrices[:, 3, :], [0, 0, 0, 1], atol=1e-4)
                or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-3)
                or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-3)):
            raise ValueError("hand joint transforms must be rigid poses")
        if np.max(np.linalg.norm(matrices[:, :3, 3] - matrices[0, :3, 3], axis=1)) > 0.35:
            raise ValueError("hand joints exceed wrist-relative hand size")
        self.accept_wrist(matrices[0], now)
        return flat

    def accept_wrist(self, wrist, now):
        # Input validity is checked above; motion amplitude is not tracking validity.
        self.pose, self.timestamp = wrist.copy(), now
        self.tracking = True
        self.status = "tracking"
        return True


class TeleVuer:
    def __init__(self, use_hand_tracking: bool, binocular: bool=True, img_shape: tuple=None, display_fps: float=30.0,
                       display_mode: Literal["immersive", "pass-through", "ego"]="immersive", zmq: bool=False, webrtc: bool=False, webrtc_url: str=None, 
                       cert_file: str=None, key_file: str=None,
                       wrist_panels: tuple=(), wrist_panel_height: float=0.26, wrist_panel_distance: float=1.2,
                       wrist_panel_offset: tuple=(0.40, 0.40), wrist_panel_aspect: float=4.0 / 3.0,
                       wrist_panel_shape: tuple=(240, 320)):
        """
        TeleVuer class for OpenXR-based XR teleoperate applications.
        This class handles the communication with the Vuer server and manages image and pose data.

        :param use_hand_tracking: bool, whether to use hand tracking or controller tracking.
        :param binocular: bool, whether the application is binocular (stereoscopic) or monocular.
        :param img_shape: tuple, shape of the head image (height, width).
        :param display_fps: float, target frames per second for display updates (default: 30.0).
        
        :param display_mode: str, controls the VR viewing mode. Options are "immersive", "pass-through", and "ego".
        :param zmq: bool, whether to use zmq for image transmission.
        :param webrtc: bool, whether to use webrtc for real-time communication.
        :param webrtc_url: str, URL for the webrtc offer. must be provided if webrtc is True.
        :param wrist_panels: sequence of sides ("left"/"right") that get an independent HUD panel.
            The frames are pushed from the main loop with render_wrist_to_xr(), which is the only
            element type this headset client renders next to the head background: a second WebRTC
            element never negotiates (measured on 2026-09-15), so the panels ride the scene channel.
        :param wrist_panel_height: float, panel height in meters at `wrist_panel_distance`.
        :param wrist_panel_distance: float, meters in front of the eyes where the panels sit.
        :param wrist_panel_offset: tuple, (x, y) panel centre offset in meters; x is mirrored per side
            (left panel at -x, right panel at +x) and y is applied downwards.
        :param wrist_panel_aspect: float, panel width/height (4:3 for the 640x480 wrist cameras).
        :param wrist_panel_shape: tuple, (height, width) the wrist frames are scaled to before they
            are JPEG-encoded into the scene stream; small keeps the tracking channel light.
        :param cert_file: str, path to the SSL certificate file.
        :param key_file: str, path to the SSL key file.

        Note:

        - display_mode controls what the VR headset displays:
            * "immersive": fully immersive mode; VR shows the robot's first-person view (zmq or webrtc must be enabled).
            * "pass-through": VR shows the real world through the VR headset cameras; no image from zmq or webrtc is displayed (even if enabled).
            * "ego": a small window in the center shows the robot's first-person view, while the surrounding area shows the real world.
        
        - Only one image mode is active at a time.
        - Image transmission to VR occurs only if display_mode is "immersive" or "ego" and the corresponding zmq or webrtc option is enabled.
        - If zmq and webrtc simultaneously enabled, webrtc will be prioritized.

        --------------              -------------------           --------------       -----------------                     -------
         display_mode       |        display behavior         |    image to VR     |      image source        |               Notes
        --------------              -------------------           --------------       -----------------                     ------- 
           immersive        |   fully immersive view (robot)  |     Yes (full)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------
         pass-through       |       Real world view (VR)      |         No         |          N/A             |  even if image source enabled, don't display
        --------------              -------------------           --------------       -----------------                     -------
              ego           |      ego view (robot + VR)      |    Yes (small)     |     zmq or webrtc        |   if both enabled, webrtc prioritized
        --------------              -------------------           --------------       -----------------                     -------

        """
        self.use_hand_tracking = use_hand_tracking
        self.binocular = binocular
        if img_shape is None:
            raise ValueError("[TeleVuer] img_shape must be provided.")
        self.img_shape = (img_shape[0], img_shape[1], 3)
        self.display_fps = display_fps
        self.img_height = self.img_shape[0]
        if self.binocular:
            self.img_width  = self.img_shape[1] // 2
        else:
            self.img_width  = self.img_shape[1]
        self.aspect_ratio = self.img_width / self.img_height

        # SSL certificate path resolution
        env_cert = os.getenv("XR_TELEOP_CERT")
        env_key = os.getenv("XR_TELEOP_KEY")
        if cert_file is None or key_file is None:
            # 1.Try environment variables
            if env_cert and env_key:
                cert_file = cert_file or env_cert
                key_file = key_file or env_key
            else:
                # 2.Try ~/.config/xr_teleoperate/
                user_conf_dir = Path.home() / ".config" / "xr_teleoperate"
                cert_path_user = user_conf_dir / "cert.pem"
                key_path_user = user_conf_dir / "key.pem"

                if cert_path_user.exists() and key_path_user.exists():
                    cert_file = cert_file or str(cert_path_user)
                    key_file = key_file or str(key_path_user)
                else:
                    # 3.Fallback to package root (current logic)
                    current_module_dir = Path(__file__).resolve().parent.parent.parent
                    cert_file = cert_file or str(current_module_dir / "cert.pem")
                    key_file = key_file or str(current_module_dir / "key.pem")

        self.vuer = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if self.use_hand_tracking:
            self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        else:
            self.vuer.add_handler("CONTROLLER_MOVE")(self.on_controller_move)

        self.display_mode = display_mode
        self.zmq = zmq
        self.webrtc = webrtc
        self.webrtc_url = webrtc_url

        # Wrist camera HUD panels: one independent panel per side, fed with BGR
        # frames by the main loop. The frames cross into the vuer process through
        # shared memory (same trick as the zmq background) because they are
        # published as ImageBackground elements next to the head view.
        self.wrist_panel_sides = tuple(side for side in wrist_panels if side in ("left", "right"))
        self.wrist_panel_height = float(wrist_panel_height)
        self.wrist_panel_distance = float(wrist_panel_distance)
        self.wrist_panel_offset = (float(wrist_panel_offset[0]), float(wrist_panel_offset[1]))
        self.wrist_panel_aspect = float(wrist_panel_aspect)
        self.wrist_panel_shape = (int(wrist_panel_shape[0]), int(wrist_panel_shape[1]), 3)
        self.wrist_panel_shm = {}
        self.wrist_panel_frames = {}
        self.wrist_panel_seq = {}
        for side in self.wrist_panel_sides:
            panel_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.wrist_panel_shape)))
            self.wrist_panel_shm[side] = panel_shm
            self.wrist_panel_frames[side] = np.ndarray(self.wrist_panel_shape, dtype=np.uint8, buffer=panel_shm.buf)
            self.wrist_panel_seq[side] = Value('L', 0, lock=True)

        if self.display_mode == "immersive":
            if self.webrtc:
                fn = self.main_image_binocular_webrtc if self.binocular else self.main_image_monocular_webrtc
            elif self.zmq:
                self.img2display_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
                self.img2display = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img2display_shm.buf)
                self.latest_frame = None
                self.new_frame_event = threading.Event()
                self.stop_writer_event = threading.Event()
                self.writer_thread = threading.Thread(target=self._xr_render_loop, daemon=True)
                self.writer_thread.start()
                fn = self.main_image_binocular_zmq if self.binocular else self.main_image_monocular_zmq
            else:
                raise ValueError("[TeleVuer] immersive mode requires zmq=True or webrtc=True.")
        elif self.display_mode == "ego":
            if self.webrtc:
                fn = self.main_image_binocular_webrtc_ego if self.binocular else self.main_image_monocular_webrtc_ego
            elif self.zmq:
                self.img2display_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
                self.img2display = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.img2display_shm.buf)
                self.latest_frame = None
                self.new_frame_event = threading.Event()
                self.stop_writer_event = threading.Event()
                self.writer_thread = threading.Thread(target=self._xr_render_loop, daemon=True)
                self.writer_thread.start()
                fn = self.main_image_binocular_zmq_ego if self.binocular else self.main_image_monocular_zmq_ego
            else:
                raise ValueError("[TeleVuer] ego mode requires zmq=True or webrtc=True.")
        elif self.display_mode == "pass-through":
            fn = self.main_pass_through
        else:
            raise ValueError(f"[TeleVuer] Unknown display_mode: {self.display_mode}")
        
        self.vuer.spawn(start=False)(fn)

        self.head_pose_shared = Array('d', 16, lock=True)
        self.left_arm_pose_shared = Array('d', 16, lock=True)
        self.right_arm_pose_shared = Array('d', 16, lock=True)
        self.motion_data_ready_shared = Value('b', False, lock=True)
        self.motion_data_timestamp_shared = Value('d', 0.0, lock=True)
        self.left_hand_timestamp_shared = Value('d', 0.0, lock=True)
        self.right_hand_timestamp_shared = Value('d', 0.0, lock=True)
        self.motion_sample_seq_shared = Value('L', 0, lock=True)
        # Events, accepted hands, errors, missing hands, and suspect hands.
        self.tracking_event_counts_shared = Array('L', 9, lock=True)
        self.tracking_hand_status_shared = Array('i', 2, lock=True)
        if self.use_hand_tracking:
            self.left_hand_position_shared = Array('d', 75, lock=True)
            self.right_hand_position_shared = Array('d', 75, lock=True)
            self.left_hand_orientation_shared = Array('d', 25 * 9, lock=True)
            self.right_hand_orientation_shared = Array('d', 25 * 9, lock=True)

            self.left_hand_pinch_shared = Value('b', False, lock=True)
            self.left_hand_pinchValue_shared = Value('d', 0.0, lock=True)
            self.left_hand_squeeze_shared = Value('b', False, lock=True)
            self.left_hand_squeezeValue_shared = Value('d', 0.0, lock=True)

            self.right_hand_pinch_shared = Value('b', False, lock=True)
            self.right_hand_pinchValue_shared = Value('d', 0.0, lock=True)
            self.right_hand_squeeze_shared = Value('b', False, lock=True)
            self.right_hand_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.hand_pose_guards = (HandPoseGuard(), HandPoseGuard())
        else:
            self.left_ctrl_trigger_shared = Value('b', False, lock=True)
            self.left_ctrl_triggerValue_shared = Value('d', 0.0, lock=True)
            self.left_ctrl_squeeze_shared = Value('b', False, lock=True)
            self.left_ctrl_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.left_ctrl_thumbstick_shared = Value('b', False, lock=True)
            self.left_ctrl_thumbstickValue_shared = Array('d', 2, lock=True)
            self.left_ctrl_aButton_shared = Value('b', False, lock=True)
            self.left_ctrl_bButton_shared = Value('b', False, lock=True)

            self.right_ctrl_trigger_shared = Value('b', False, lock=True)
            self.right_ctrl_triggerValue_shared = Value('d', 0.0, lock=True)
            self.right_ctrl_squeeze_shared = Value('b', False, lock=True)
            self.right_ctrl_squeezeValue_shared = Value('d', 0.0, lock=True)
            self.right_ctrl_thumbstick_shared = Value('b', False, lock=True)
            self.right_ctrl_thumbstickValue_shared = Array('d', 2, lock=True)
            self.right_ctrl_aButton_shared = Value('b', False, lock=True)
            self.right_ctrl_bButton_shared = Value('b', False, lock=True)

        self.process = Process(target=self._vuer_run)
        self.process.daemon = True
        self.process.start()
    
    def _vuer_run(self):
        try:
            self.vuer.run()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Vuer encountered an error: {e}")
        finally:
            if hasattr(self, "stop_writer_event"):
                self.stop_writer_event.set()

    def _xr_render_loop(self):
        while not self.stop_writer_event.is_set():
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()
            if self.latest_frame is None:
                continue
            latest_frame = self.latest_frame
            latest_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
            self.img2display[:] = latest_frame
    
    def render_to_xr(self, image):
        if self.webrtc or self.display_mode == "pass-through":
            print("[TeleVuer] Warning: render_to_xr is ignored when webrtc is enabled or pass_through is True.")
            return
        self.latest_frame = image
        self.new_frame_event.set()

    def close(self):
        self.process.terminate()
        self.process.join(timeout=0.5)
        if self.display_mode in ("immersive", "ego") and not self.webrtc:
            self.stop_writer_event.set()
            self.new_frame_event.set()
            self.writer_thread.join(timeout=0.5)
            try:
                self.img2display_shm.close()
                self.img2display_shm.unlink()
            except:
                pass
        for panel_shm in self.wrist_panel_shm.values():
            try:
                panel_shm.close()
                panel_shm.unlink()
            except:
                pass
        self.wrist_panel_shm = {}
        self.wrist_panel_frames = {}
        self.wrist_panel_seq = {}

    async def on_cam_move(self, event, session, fps=60):
        with self.tracking_event_counts_shared.get_lock():
            self.tracking_event_counts_shared[0] += 1
        try:
            with self.head_pose_shared.get_lock():
                self.head_pose_shared[:] = event.value["camera"]["matrix"]
        except:
            pass

    async def on_controller_move(self, event, session, fps=60):
        """https://docs.vuer.ai/en/latest/examples/20_motion_controllers.html"""
        try:
            # ControllerData
            with self.left_arm_pose_shared.get_lock():
                self.left_arm_pose_shared[:] = event.value["left"]
            with self.right_arm_pose_shared.get_lock():
                self.right_arm_pose_shared[:] = event.value["right"]
            # ControllerState
            left_controller = event.value["leftState"]
            right_controller = event.value["rightState"]

            def extract_controllers(controllerState, prefix):
                # trigger
                with getattr(self, f"{prefix}_ctrl_trigger_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_trigger_shared").value = bool(controllerState.get("trigger", False))
                with getattr(self, f"{prefix}_ctrl_triggerValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_triggerValue_shared").value = float(controllerState.get("triggerValue", 0.0))
                # squeeze
                with getattr(self, f"{prefix}_ctrl_squeeze_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_squeeze_shared").value = bool(controllerState.get("squeeze", False))
                with getattr(self, f"{prefix}_ctrl_squeezeValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_squeezeValue_shared").value = float(controllerState.get("squeezeValue", 0.0))
                # thumbstick
                with getattr(self, f"{prefix}_ctrl_thumbstick_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_thumbstick_shared").value = bool(controllerState.get("thumbstick", False))
                with getattr(self, f"{prefix}_ctrl_thumbstickValue_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_thumbstickValue_shared")[:] = controllerState.get("thumbstickValue", [0.0, 0.0])
                # buttons
                with getattr(self, f"{prefix}_ctrl_aButton_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_aButton_shared").value = bool(controllerState.get("aButton", False))
                with getattr(self, f"{prefix}_ctrl_bButton_shared").get_lock():
                    getattr(self, f"{prefix}_ctrl_bButton_shared").value = bool(controllerState.get("bButton", False))

            extract_controllers(left_controller, "left")
            extract_controllers(right_controller, "right")
            with self.motion_data_timestamp_shared.get_lock():
                self.motion_data_timestamp_shared.value = time.monotonic()
            with self.motion_data_ready_shared.get_lock():
                self.motion_data_ready_shared.value = True
        except:
            pass

    def _begin_motion_sample(self):
        with self.motion_sample_seq_shared.get_lock():
            current = self.motion_sample_seq_shared.value
            sample_seq = current + 1 if current % 2 == 0 else current + 2
            self.motion_sample_seq_shared.value = sample_seq
            return sample_seq

    def _commit_motion_sample(self, sample_seq):
        with self.motion_sample_seq_shared.get_lock():
            if self.motion_sample_seq_shared.value == sample_seq:
                self.motion_sample_seq_shared.value = sample_seq + 1

    async def on_hand_move(self, event, session, fps=60):
        with self.tracking_event_counts_shared.get_lock():
            self.tracking_event_counts_shared[1] += 1
        sample_seq = self._begin_motion_sample()
        now = time.monotonic()
        errors = []
        timestamps = []
        try:
            for index, (side, guard) in enumerate(zip(("left", "right"), self.hand_pose_guards)):
                flat = None
                try:
                    if not isinstance(event.value, dict):
                        raise ValueError("hand tracking payload must be an object")
                    flat = guard.update(event.value.get(side), now)
                    if flat is not None:
                        state = event.value.get(side + "State")
                        state = state if isinstance(state, dict) else {}
                        pinch_value = float(state.get("pinchValue", 0.0))
                        squeeze_value = float(state.get("squeezeValue", 0.0))
                        if not np.isfinite([pinch_value, squeeze_value]).all():
                            raise ValueError("hand gestures must be finite")
                except (TypeError, ValueError, OverflowError) as exc:
                    flat = None
                    guard.lose("invalid")
                    errors.append(f"{side}: {exc}")

                # Invalidate this side in the same snapshot as the event that lost tracking.
                timestamp = now if flat is not None else 0.0
                with getattr(self, side + "_hand_timestamp_shared").get_lock():
                    getattr(self, side + "_hand_timestamp_shared").value = timestamp
                timestamps.append(timestamp)
                with self.tracking_hand_status_shared.get_lock():
                    self.tracking_hand_status_shared[index] = HAND_TRACKING_STATES.index(guard.status)
                with self.tracking_event_counts_shared.get_lock():
                    self.tracking_event_counts_shared[2 + index] += int(flat is not None)
                    self.tracking_event_counts_shared[5 + index] += int(guard.status == "missing")
                    self.tracking_event_counts_shared[7 + index] += int(guard.status == "invalid")
                if flat is None:
                    continue
                matrices = flat.reshape(25, 4, 4)
                with getattr(self, side + "_arm_pose_shared").get_lock():
                    getattr(self, side + "_arm_pose_shared")[:] = flat[:16]
                with getattr(self, side + "_hand_position_shared").get_lock():
                    getattr(self, side + "_hand_position_shared")[:] = matrices[:, 3, :3].flatten()
                with getattr(self, side + "_hand_orientation_shared").get_lock():
                    getattr(self, side + "_hand_orientation_shared")[:] = matrices[:, :3, :3].flatten()
                for field, value in (
                    ("pinch", bool(state.get("pinch", False))), ("pinchValue", pinch_value),
                    ("squeeze", bool(state.get("squeeze", False))), ("squeezeValue", squeeze_value),
                ):
                    shared = getattr(self, side + "_hand_" + field + "_shared")
                    with shared.get_lock():
                        shared.value = value

            with self.motion_data_timestamp_shared.get_lock():
                self.motion_data_timestamp_shared.value = min(timestamps)
            # Any single side is enough to call the stream ready: each side's own
            # timestamp is zeroed above when that side is missing, so per-side
            # staleness is decided by the consumer, not by this shared flag.
            # Requiring both sides here coupled them: one lost hand made the other
            # one look stale too and froze both arms.
            if any(timestamps):
                with self.motion_data_ready_shared.get_lock():
                    self.motion_data_ready_shared.value = True
        finally:
            self._commit_motion_sample(sample_seq)
        if errors:
            with self.tracking_event_counts_shared.get_lock():
                self.tracking_event_counts_shared[4] += 1
            if now - getattr(self, "_last_hand_move_error_log", 0.0) >= 1.0:
                print("[TeleVuer] HAND_MOVE malformed: " + "; ".join(errors), flush=True)
                self._last_hand_move_error_log = now

    # ==================== wrist camera panels ====================
    #: Quarter turns applied to each palm frame before it reaches its panel.
    #: numpy's rot90 counts counter-clockwise, so +1 is a quarter turn CCW and
    #: -1 a quarter turn CW.
    #:
    #: The two cameras sit mirrored in the hands, so the base quarter turns are
    #: opposite; a further half turn is added on top of both. Written as base
    #: plus half rather than as the folded result so the intent stays readable
    #: (rot90 reduces k modulo 4, so +2 is exactly the half turn).
    #: Callers size the panel from the rotated shape.
    WRIST_PANEL_ROTATION = {"left": 1 + 2, "right": -1 + 2}

    def render_wrist_to_xr(self, side, image):
        """Publish one wrist camera frame (BGR) to the matching HUD panel.

        Called from the main teleop loop; the frame is scaled down once here and
        then re-encoded into the scene stream by the vuer process. Safe to call
        before the headset connects (the panel simply shows the latest frame).
        """
        frame = self.wrist_panel_frames.get(side)
        if frame is None or image is None:
            return
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            return
        turns = self.WRIST_PANEL_ROTATION.get(side, 0)
        if turns:
            image = np.ascontiguousarray(np.rot90(image, turns))
        if image.shape[:2] != frame.shape[:2]:
            image = cv2.resize(image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
        frame[:] = image
        sequence = self.wrist_panel_seq[side]
        with sequence.get_lock():
            sequence.value += 1

    def wrist_panel_ready(self, side):
        sequence = self.wrist_panel_seq.get(side)
        if sequence is None:
            return False
        with sequence.get_lock():
            return sequence.value > 0

    def _wrist_panel_elements(self):
        """HUD panels for the wrist cameras that have received a frame.

        `ImageBackground` is the overlay element the headset renders next to the
        head view; `position`/`distanceToCamera`/`height` place it at the lower
        left/right of the field of view and mirror the offset per side.
        """
        elements = []
        for side, sign in (("left", -1.0), ("right", 1.0)):
            frame = self.wrist_panel_frames.get(side)
            if frame is None or not self.wrist_panel_ready(side):
                continue
            elements.append(
                ImageBackground(
                    frame,
                    aspect=self.wrist_panel_aspect,
                    height=self.wrist_panel_height,
                    distanceToCamera=self.wrist_panel_distance,
                    position=[sign * self.wrist_panel_offset[0], -self.wrist_panel_offset[1], 0.0],
                    format="jpeg",
                    quality=75,
                    key=f"wrist-{side}",
                )
            )
        return elements

    def _upsert_wrist_panels(self, session):
        """(Re)publish the wrist panels; called every frame by the render loops
        because a one-shot upsert races the headset's page load."""
        elements = self._wrist_panel_elements()
        if elements:
            session.upsert(elements, to="bgChildren")

    ## immersive MODE
    async def main_image_binocular_zmq(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display[:, :self.img_width],
                        aspect=self.aspect_ratio,
                        height=1,
                        distanceToCamera=1,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=80,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        self.img2display[:, self.img_width:],
                        aspect=self.aspect_ratio,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        format="jpeg",
                        quality=80,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            self._upsert_wrist_panels(session)
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_zmq(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display,
                        aspect=self.aspect_ratio,
                        height=1,
                        distanceToCamera=1,
                        format="jpeg",
                        quality=80,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            self._upsert_wrist_panels(session)
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_binocular_webrtc(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCStereoVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[], 
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height = 11,
                    layout="stereo-left-right"
                ),
                to="bgChildren",
            )
            self._upsert_wrist_panels(session)
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_webrtc(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[],
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height = 7,
                ),
                to="bgChildren",
            )
            self._upsert_wrist_panels(session)
            await asyncio.sleep(1.0 / self.display_fps)

    ## ego MODE
    async def main_image_binocular_zmq_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True,
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )
        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display[:, :self.img_width],
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera. 
                        # Below we set the two image planes, left and right, to layers=1 and layers=2. 
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=80,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        self.img2display[:, self.img_width:],
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        layers=2,
                        format="jpeg",
                        quality=80,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            self._upsert_wrist_panels(session)
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_zmq_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                [
                    ImageBackground(
                        self.img2display,
                        aspect=self.aspect_ratio,
                        height=0.75,
                        distanceToCamera=2,
                        format="jpeg",
                        quality=80,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            self._upsert_wrist_panels(session)
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_binocular_webrtc_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCStereoVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[], 
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height=3,
                    layout="stereo-left-right"
                ),
                to="bgChildren",
            )
            self._upsert_wrist_panels(session)
            await asyncio.sleep(1.0 / self.display_fps)

    async def main_image_monocular_webrtc_ego(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            session.upsert(
                WebRTCVideoPlane(
                    src=self.webrtc_url,
                    iceServer=None,
                    iceServers=[],
                    key="video-quad",
                    aspect=self.aspect_ratio,
                    height=3,
                ),
                to="bgChildren",
            )
            self._upsert_wrist_panels(session)
            await asyncio.sleep(1.0 / self.display_fps)

    ## pass-through MODE
    async def main_pass_through(self, session):
        if self.use_hand_tracking:
            session.upsert(
                Hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True
                ),
                to="bgChildren",
            )
        else:
            session.upsert(
                MotionControllers(
                    stream=True, 
                    key="motionControllers",
                    left=True,
                    right=True,
                ),
                to="bgChildren",
            )

        while True:
            await asyncio.sleep(1.0 / self.display_fps)

    # ==================== common data ====================
    @property
    def head_pose(self):
        """np.ndarray, shape (4, 4), head SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.head_pose_shared.get_lock():
            return np.array(self.head_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def left_arm_pose(self):
        """np.ndarray, shape (4, 4), left arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.left_arm_pose_shared.get_lock():
            return np.array(self.left_arm_pose_shared[:]).reshape(4, 4, order="F")

    @property
    def right_arm_pose(self):
        """np.ndarray, shape (4, 4), right arm SE(3) pose matrix from Vuer (basis OpenXR Convention)."""
        with self.right_arm_pose_shared.get_lock():
            return np.array(self.right_arm_pose_shared[:]).reshape(4, 4, order="F")

    # ==================== Hand Tracking Data ====================
    @property
    def left_hand_positions(self):
        """np.ndarray, shape (25, 3), left hand 25 landmarks' 3D positions."""
        with self.left_hand_position_shared.get_lock():
            return np.array(self.left_hand_position_shared[:]).reshape(25, 3)

    @property
    def right_hand_positions(self):
        """np.ndarray, shape (25, 3), right hand 25 landmarks' 3D positions."""
        with self.right_hand_position_shared.get_lock():
            return np.array(self.right_hand_position_shared[:]).reshape(25, 3)

    @property
    def left_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), left hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.left_hand_orientation_shared.get_lock():
            return np.array(self.left_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def right_hand_orientations(self):
        """np.ndarray, shape (25, 3, 3), right hand 25 landmarks' orientations (flattened 3x3 matrices, column-major)."""
        with self.right_hand_orientation_shared.get_lock():
            return np.array(self.right_hand_orientation_shared[:]).reshape(25, 9).reshape(25, 3, 3, order="F")

    @property
    def left_hand_pinch(self):
        """bool, whether left hand is pinching."""
        with self.left_hand_pinch_shared.get_lock():
            return self.left_hand_pinch_shared.value

    @property
    def left_hand_pinchValue(self):
        """float, pinch strength of left hand."""
        with self.left_hand_pinchValue_shared.get_lock():
            return self.left_hand_pinchValue_shared.value

    @property
    def left_hand_squeeze(self):
        """bool, whether left hand is squeezing."""
        with self.left_hand_squeeze_shared.get_lock():
            return self.left_hand_squeeze_shared.value

    @property
    def left_hand_squeezeValue(self):
        """float, squeeze strength of left hand."""
        with self.left_hand_squeezeValue_shared.get_lock():
            return self.left_hand_squeezeValue_shared.value

    @property
    def right_hand_pinch(self):
        """bool, whether right hand is pinching."""
        with self.right_hand_pinch_shared.get_lock():
            return self.right_hand_pinch_shared.value

    @property
    def right_hand_pinchValue(self):
        """float, pinch strength of right hand."""
        with self.right_hand_pinchValue_shared.get_lock():
            return self.right_hand_pinchValue_shared.value

    @property
    def right_hand_squeeze(self):
        """bool, whether right hand is squeezing."""
        with self.right_hand_squeeze_shared.get_lock():
            return self.right_hand_squeeze_shared.value

    @property
    def right_hand_squeezeValue(self):
        """float, squeeze strength of right hand."""
        with self.right_hand_squeezeValue_shared.get_lock():
            return self.right_hand_squeezeValue_shared.value

    # ==================== Controller Data ====================
    @property
    def left_ctrl_trigger(self):
        """bool, left controller trigger pressed or not."""
        with self.left_ctrl_trigger_shared.get_lock():
            return self.left_ctrl_trigger_shared.value

    @property
    def left_ctrl_triggerValue(self):
        """float, left controller trigger analog value (0.0 ~ 1.0)."""
        with self.left_ctrl_triggerValue_shared.get_lock():
            return self.left_ctrl_triggerValue_shared.value

    @property
    def left_ctrl_squeeze(self):
        """bool, left controller squeeze pressed or not."""
        with self.left_ctrl_squeeze_shared.get_lock():
            return self.left_ctrl_squeeze_shared.value

    @property
    def left_ctrl_squeezeValue(self):
        """float, left controller squeeze analog value (0.0 ~ 1.0)."""
        with self.left_ctrl_squeezeValue_shared.get_lock():
            return self.left_ctrl_squeezeValue_shared.value

    @property
    def left_ctrl_thumbstick(self):
        """bool, whether left thumbstick is touched or clicked."""
        with self.left_ctrl_thumbstick_shared.get_lock():
            return self.left_ctrl_thumbstick_shared.value

    @property
    def left_ctrl_thumbstickValue(self):
        """np.ndarray, shape (2,), left thumbstick 2D axis values (x, y)."""
        with self.left_ctrl_thumbstickValue_shared.get_lock():
            return np.array(self.left_ctrl_thumbstickValue_shared[:])

    @property
    def left_ctrl_aButton(self):
        """bool, left controller 'A' button pressed."""
        with self.left_ctrl_aButton_shared.get_lock():
            return self.left_ctrl_aButton_shared.value

    @property
    def left_ctrl_bButton(self):
        """bool, left controller 'B' button pressed."""
        with self.left_ctrl_bButton_shared.get_lock():
            return self.left_ctrl_bButton_shared.value

    @property
    def right_ctrl_trigger(self):
        """bool, right controller trigger pressed or not."""
        with self.right_ctrl_trigger_shared.get_lock():
            return self.right_ctrl_trigger_shared.value

    @property
    def right_ctrl_triggerValue(self):
        """float, right controller trigger analog value (0.0 ~ 1.0)."""
        with self.right_ctrl_triggerValue_shared.get_lock():
            return self.right_ctrl_triggerValue_shared.value

    @property
    def right_ctrl_squeeze(self):
        """bool, right controller squeeze pressed or not."""
        with self.right_ctrl_squeeze_shared.get_lock():
            return self.right_ctrl_squeeze_shared.value

    @property
    def right_ctrl_squeezeValue(self):
        """float, right controller squeeze analog value (0.0 ~ 1.0)."""
        with self.right_ctrl_squeezeValue_shared.get_lock():
            return self.right_ctrl_squeezeValue_shared.value

    @property
    def right_ctrl_thumbstick(self):
        """bool, whether right thumbstick is touched or clicked."""
        with self.right_ctrl_thumbstick_shared.get_lock():
            return self.right_ctrl_thumbstick_shared.value

    @property
    def right_ctrl_thumbstickValue(self):
        """np.ndarray, shape (2,), right thumbstick 2D axis values (x, y)."""
        with self.right_ctrl_thumbstickValue_shared.get_lock():
            return np.array(self.right_ctrl_thumbstickValue_shared[:])

    @property
    def right_ctrl_aButton(self):
        """bool, right controller 'A' button pressed."""
        with self.right_ctrl_aButton_shared.get_lock():
            return self.right_ctrl_aButton_shared.value

    @property
    def right_ctrl_bButton(self):
        """bool, right controller 'B' button pressed."""
        with self.right_ctrl_bButton_shared.get_lock():
            return self.right_ctrl_bButton_shared.value

    @property
    def motion_data_ready(self):
        """bool, whether at least one hand or controller motion data event has been received."""
        with self.motion_data_ready_shared.get_lock():
            return self.motion_data_ready_shared.value

    @property
    def motion_data_timestamp(self):
        """Monotonic timestamp of the older hand sample or latest controller event."""
        with self.motion_data_timestamp_shared.get_lock():
            return self.motion_data_timestamp_shared.value

    def get_tracking_diagnostics(self):
        with self.tracking_event_counts_shared.get_lock():
            result = dict(zip(
                ("camera_events", "hand_events", "left_valid", "right_valid", "rejected",
                 "left_missing", "right_missing", "left_suspect", "right_suspect"),
                self.tracking_event_counts_shared[:],
            ))
        now = time.monotonic()
        for name, shared in (
            ("left_age_ms", self.left_hand_timestamp_shared),
            ("right_age_ms", self.right_hand_timestamp_shared),
            ("pair_age_ms", self.motion_data_timestamp_shared),
        ):
            with shared.get_lock():
                timestamp = shared.value
            result[name] = round((now - timestamp) * 1000.0, 1) if timestamp > 0 else None
        result["ready"] = self.motion_data_ready
        with self.tracking_hand_status_shared.get_lock():
            for index, side in enumerate(("left", "right")):
                result[side + "_state"] = HAND_TRACKING_STATES[self.tracking_hand_status_shared[index]]
                age = result[side + "_age_ms"]
                result[side + "_fresh"] = age is not None and 0.0 <= age <= 250.0
        with self.motion_sample_seq_shared.get_lock():
            result["sample_seq"] = self.motion_sample_seq_shared.value
        return result

    def get_hand_motion_snapshot(self, include_orientations=False, max_attempts=3):
        for _ in range(max_attempts):
            with self.motion_sample_seq_shared.get_lock():
                seq_before = self.motion_sample_seq_shared.value
            if seq_before % 2 != 0:
                continue

            snapshot = {
                "left_arm_pose": self.left_arm_pose,
                "right_arm_pose": self.right_arm_pose,
                "left_hand_positions": self.left_hand_positions,
                "right_hand_positions": self.right_hand_positions,
                "left_hand_pinch": self.left_hand_pinch,
                "left_hand_pinchValue": self.left_hand_pinchValue,
                "left_hand_squeeze": self.left_hand_squeeze,
                "left_hand_squeezeValue": self.left_hand_squeezeValue,
                "right_hand_pinch": self.right_hand_pinch,
                "right_hand_pinchValue": self.right_hand_pinchValue,
                "right_hand_squeeze": self.right_hand_squeeze,
                "right_hand_squeezeValue": self.right_hand_squeezeValue,
                "motion_data_ready": self.motion_data_ready,
                "motion_data_timestamp": self.motion_data_timestamp,
                "motion_sample_seq": seq_before,
            }
            with self.left_hand_timestamp_shared.get_lock():
                snapshot["left_hand_timestamp"] = self.left_hand_timestamp_shared.value
            with self.right_hand_timestamp_shared.get_lock():
                snapshot["right_hand_timestamp"] = self.right_hand_timestamp_shared.value
            if include_orientations:
                snapshot["left_hand_orientations"] = self.left_hand_orientations
                snapshot["right_hand_orientations"] = self.right_hand_orientations

            with self.motion_sample_seq_shared.get_lock():
                seq_after = self.motion_sample_seq_shared.value
            if seq_before == seq_after and seq_after % 2 == 0:
                return snapshot
        return None
