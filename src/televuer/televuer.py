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


#: Upper bounds, in milliseconds, of the hand_move callback inter-arrival bins.
#: The last bin collects everything above the final edge.
MOTION_INTERVAL_EDGES_MS = (5, 10, 20, 30, 40, 50, 70, 90, 120, 160, 220, 300, 450, 600, 1000)

#: Upper bounds, in milliseconds, of the Vuer event-loop lag bins. The hand and
#: camera handlers run on that loop together with the scene updates that push
#: images to the headset, so if those updates block it, incoming hand samples
#: cannot be dispatched and arrive in bursts instead.
EVENT_LOOP_LAG_EDGES_MS = (1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1600)

#: Slots in the motion-sample queue. Hand samples arrive in bursts -- measured
#: 2026-09-16, 44% of callbacks land within 5 ms of the previous one -- and a
#: consumer that reads a latest-value slot keeps only the last frame of each
#: burst, halving the rate at which the arm reference can advance. Four slots
#: cover any realistic burst while staying small.
MOTION_QUEUE_SLOTS = 4

#: A consumer more than this far behind is resynchronised to the newest samples
#: rather than playing out history.
MOTION_QUEUE_MAX_BACKLOG = 3

#: Per slot: the three timestamps, then both arm poses, hand positions and hand
#: orientations, laid out in the order MOTION_QUEUE_FIELDS reads them.
MOTION_QUEUE_STRIDE = 3 + 2 * 16 + 2 * 75 + 2 * 225
MOTION_QUEUE_FIELDS = ("left_arm_pose", "right_arm_pose",
                       "left_hand_positions", "right_hand_positions",
                       "left_hand_orientations", "right_hand_orientations")
MOTION_QUEUE_SHAPES = ((4, 4), (4, 4), (25, 3), (25, 3), (25, 3, 3), (25, 3, 3))


def bin_labels(edges):
    """Labels for the histogram bins an edge tuple describes: below the first
    edge, each edge-to-edge span, and everything at or above the final edge."""
    labels = ["<=%d" % edges[0]]
    labels += ["%d-%d" % (edges[i], edges[i + 1]) for i in range(len(edges) - 2)]
    labels.append(">=%d" % edges[-1])
    return labels


def bin_index(value, edges):
    index = len(edges) - 1
    for position, edge in enumerate(edges):
        if value < edge:
            index = position
            break
    return index


HAND_TRACKING_STATES = ("missing", "tracking", "invalid")


def webxr_session_mode_for_display(display_mode: str) -> str:
    """Request AR for passthrough; the client verifies device support and blending."""
    if display_mode in ("ego", "pass-through"):
        return "immersive-ar"
    return "immersive-vr"


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
                       wrist_panel_shape: tuple=(240, 320),
                       torque_hud: bool=False):
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
        :param torque_hud: bool, overlay Linker O6 joint-torque number strips on the
            scene channel (the same ImageBackground path as the wrist panels). Needed
            because the headset watches WebRTC 60001; painting the ZMQ JPEG never
            reaches the display.
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
        # Content counters for the XR scene. The render loop runs at display_fps
        # while the head camera runs near 7.5 Hz, so an unconditional upsert
        # re-encodes and re-sends the same JPEG four times: measured 111.5 KB per
        # frame, 3.27 MB/s pushed against 0.83 MB/s of real content, on the same
        # wireless link the hand-tracking uplink uses. The sequence is bumped by the
        # writer thread once a new frame is in shared memory, so a reader that sees
        # a new value is guaranteed to find that frame there.
        self.xr_frame_sequence = Value('L', 0, lock=True)
        self.wrist_panel_sent = {}
        self.xr_frame_sent = None
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

        # Vision Pro: ego/pass-through need immersive-ar or the surround is opaque black.
        self.webxr_session_mode = webxr_session_mode_for_display(display_mode)
        self.vuer = Vuer(
            host='0.0.0.0',
            cert=cert_file,
            key=key_file,
            queries=dict(grid=False, xrMode=self.webxr_session_mode),
            queue_len=3,
        )
        self._install_webxr_mode_index()
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

        # Torque number strips ride the same scene channel as the wrist panels so
        # they show up on top of WebRTC, not only on the unused ZMQ JPEG path.
        self.torque_hud_enabled = bool(torque_hud)
        # Digits are centred in this strip. The old 512x40 plane was 1.28 m
        # wide, so the glyphs sat on its left edge, outside the video window.
        self.torque_hud_shape = (48, 280, 3)
        self.torque_hud_aspect = self.torque_hud_shape[1] / self.torque_hud_shape[0]
        self.torque_hud_height = 0.09
        self.torque_hud_distance = 1.0
        # (right_x, y). Same inset on each side, just above the middle.
        self.torque_hud_offset = (0.36, 0.06)
        self.torque_hud_left_x = 0.36
        self.torque_hud_shm = {}
        self.torque_hud_frames = {}
        self.torque_hud_seq = {}
        self.torque_hud_sent = None
        if self.torque_hud_enabled:
            for side in ("left", "right"):
                panel_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(self.torque_hud_shape)))
                self.torque_hud_shm[side] = panel_shm
                self.torque_hud_frames[side] = np.ndarray(self.torque_hud_shape, dtype=np.uint8, buffer=panel_shm.buf)
                self.torque_hud_seq[side] = Value('L', 0, lock=True)

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
        # Where the motion samples are lost, stage by stage:
        #   0 hand_move events seen, 1 samples begun, 2 committed,
        #   3 motion timestamp written, 4 of those written as 0.0 because one
        #   hand was missing (min() over a list containing the 0.0 marker),
        #   5 snapshots the consumer took successfully, 6 snapshots that gave up,
        #   7 consumer attempts that found a sample mid-flight.
        self.motion_sample_counts_shared = Array('L', 8, lock=True)
        # Inter-arrival histogram of hand_move callbacks, in milliseconds. The
        # counters alone cannot tell a steady 27 Hz stream from the same rate
        # delivered in bursts, and only the burst case explains why the control
        # loop sees fresh poses at ~11 Hz while 27 callbacks/s arrive.
        self.motion_interval_bins_shared = Array('L', len(MOTION_INTERVAL_EDGES_MS), lock=True)
        self._motion_interval_previous = None
        # How long the Vuer event loop is unavailable, sampled inside the Vuer
        # process. See EVENT_LOOP_LAG_EDGES_MS.
        self.event_loop_lag_bins_shared = Array('L', len(EVENT_LOOP_LAG_EDGES_MS), lock=True)
        self._heartbeat_started = False
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
            # Ordered history of hand samples, for consumers that want every frame
            # of a burst instead of only the newest. See MOTION_QUEUE_SLOTS.
            self.motion_queue_data_shared = Array('d', MOTION_QUEUE_SLOTS * MOTION_QUEUE_STRIDE, lock=True)
            self.motion_queue_marks_shared = Array('L', MOTION_QUEUE_SLOTS, lock=True)
            self.motion_queue_head_shared = Value('L', 0, lock=True)
            self.motion_queue_tail_shared = Value('L', 0, lock=True)
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
            with self.xr_frame_sequence.get_lock():
                self.xr_frame_sequence.value += 1
    
    def render_to_xr(self, image, sequence=None):
        if self.webrtc or self.display_mode == "pass-through":
            print("[TeleVuer] Warning: render_to_xr is ignored when webrtc is enabled or pass_through is True.")
            return
        if sequence is not None:
            # Same camera frame as last time: the colour conversion and the shared
            # memory copy below would both be redone for nothing.
            if sequence == getattr(self, "xr_frame_input", None):
                return
            self.xr_frame_input = sequence
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
        for panel_shm in getattr(self, "torque_hud_shm", {}).values():
            try:
                panel_shm.close()
                panel_shm.unlink()
            except:
                pass
        self.torque_hud_shm = {}
        self.torque_hud_frames = {}
        self.torque_hud_seq = {}

    async def _event_loop_heartbeat(self):
        """Sample how long this event loop is unavailable, from inside this process.

        A starved loop cannot dispatch hand samples as they arrive, so they queue
        and are handled in a burst. The motion_interval histogram shows the burst;
        this shows whether the loop is the reason for it.
        """
        interval = 0.005
        while True:
            started = time.monotonic()
            await asyncio.sleep(interval)
            bins = getattr(self, "event_loop_lag_bins_shared", None)
            if bins is None:
                continue
            lag_ms = (time.monotonic() - started - interval) * 1000.0
            if not np.isfinite(lag_ms) or lag_ms < 0.0:
                continue
            with bins.get_lock():
                bins[bin_index(lag_ms, EVENT_LOOP_LAG_EDGES_MS)] += 1

    def _install_webxr_mode_index(self):
        """Serve one consistent client build with the selected XR session mode."""
        from aiohttp.hdrs import UPGRADE
        from aiohttp.web import HTTPFound, Response
        from vuer.server import Vuer as _Vuer

        vuer = self.vuer
        mode = self.webxr_session_mode

        async def socket_index(request):
            if "websocket" == request.headers.get(UPGRADE, "").lower().strip():
                return await _Vuer.socket_index(vuer, request)
            if request.rel_url.query.get("xrMode") != mode:
                items = [(k, v) for k, v in request.rel_url.query.items() if k != "xrMode"]
                items.append(("xrMode", mode))
                raise HTTPFound(str(request.rel_url.with_query(items)))
            index_path = Path(vuer.client_root) / "assets/xr-session-fix/index.html"
            html = index_path.read_text(encoding="utf-8")
            snippet = f'<script>window.__TELEVUER_XR_MODE__="{mode}";</script>'
            html = html.replace("</head>", snippet + "\n</head>", 1)
            return Response(text=html, content_type="text/html",
                            headers={"Cache-Control": "no-store"})

        vuer.socket_index = socket_index

    async def on_cam_move(self, event, session, fps=60):
        if not getattr(self, "_heartbeat_started", False):
            # First camera callback proves the loop is running, which is the only
            # place a task can be scheduled from.
            try:
                asyncio.create_task(self._event_loop_heartbeat())
                self._heartbeat_started = True
            except RuntimeError:
                pass
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

    def _count_motion(self, index, amount=1):
        """Bump one stage counter if the diagnostic array exists.

        Tolerant on purpose: some tests build a TeleVuer without running
        __init__, and a missing counter must never break the control path.
        """
        counts = getattr(self, "motion_sample_counts_shared", None)
        if counts is None:
            return
        if index >= len(counts):
            return
        with counts.get_lock():
            counts[index] += amount

    def _record_motion_interval(self, now):
        """Bin the gap since the previous hand_move callback."""
        bins = getattr(self, "motion_interval_bins_shared", None)
        if bins is None:
            return
        previous = getattr(self, "_motion_interval_previous", None)
        self._motion_interval_previous = now
        if previous is None:
            return
        elapsed_ms = (now - previous) * 1000.0
        if not np.isfinite(elapsed_ms) or elapsed_ms < 0.0:
            return
        with bins.get_lock():
            bins[bin_index(elapsed_ms, MOTION_INTERVAL_EDGES_MS)] += 1

    def _begin_motion_sample(self):
        self._count_motion(1)
        with self.motion_sample_seq_shared.get_lock():
            current = self.motion_sample_seq_shared.value
            sample_seq = current + 1 if current % 2 == 0 else current + 2
            self.motion_sample_seq_shared.value = sample_seq
            return sample_seq

    def _commit_motion_sample(self, sample_seq):
        committed = False
        with self.motion_sample_seq_shared.get_lock():
            if self.motion_sample_seq_shared.value == sample_seq:
                self.motion_sample_seq_shared.value = sample_seq + 1
                committed = True
        self._count_motion(2, int(committed))

    async def on_hand_move(self, event, session, fps=60):
        with self.tracking_event_counts_shared.get_lock():
            self.tracking_event_counts_shared[1] += 1
        self._count_motion(0)
        self._record_motion_interval(time.monotonic())
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

            oldest = min(timestamps)
            with self.motion_data_timestamp_shared.get_lock():
                self.motion_data_timestamp_shared.value = oldest
            self._count_motion(3)
            self._count_motion(4, int(oldest <= 0.0))
            # Any single side is enough to call the stream ready: each side's own
            # timestamp is zeroed above when that side is missing, so per-side
            # staleness is decided by the consumer, not by this shared flag.
            # Requiring both sides here coupled them: one lost hand made the other
            # one look stale too and froze both arms.
            if any(timestamps):
                with self.motion_data_ready_shared.get_lock():
                    self.motion_data_ready_shared.value = True
            self._append_motion_sample()
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

    def render_wrist_to_xr(self, side, image, sequence=None):
        """Publish one wrist camera frame (BGR) to the matching HUD panel.

        Called from the main teleop loop; the frame is scaled down once here and
        then re-encoded into the scene stream by the vuer process. Safe to call
        before the headset connects (the panel simply shows the latest frame).

        ``sequence`` is the camera's own frame counter: when it has not moved there
        is nothing new to show, and leaving the panel sequence alone also lets the
        render loop skip re-encoding the panel.
        """
        frame = self.wrist_panel_frames.get(side)
        if frame is None or image is None:
            return
        if sequence is not None:
            last = getattr(self, "_wrist_frame_input", None)
            if last is None:
                last = {}
                self._wrist_frame_input = last
            if last.get(side) == sequence:
                return
            last[side] = sequence
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

    def _head_plane_geometry(self):
        """Plane size and distance for the head image, per display mode."""
        if self.display_mode == "ego":
            return 0.75, 2
        return 1, 1

    def _head_plane_elements(self):
        """Image planes for the head frame currently in shared memory."""
        display = getattr(self, "img2display", None)
        if display is None:
            return []
        height, distance = self._head_plane_geometry()
        if not self.binocular:
            return [
                ImageBackground(
                    display,
                    aspect=self.aspect_ratio,
                    height=height,
                    distanceToCamera=distance,
                    format="jpeg",
                    quality=80,
                    key="background-mono",
                    interpolate=True,
                ),
            ]
        return [
            ImageBackground(
                display[:, :self.img_width],
                aspect=self.aspect_ratio,
                height=height,
                distanceToCamera=distance,
                # The rendering engine takes a layer bitmask for both objects and
                # the camera, so the two eye planes are layers 1 and 2: the left
                # eye's camera shows one and the right eye's shows the other.
                layers=1,
                format="jpeg",
                quality=80,
                key="background-left",
                interpolate=True,
            ),
            ImageBackground(
                display[:, self.img_width:],
                aspect=self.aspect_ratio,
                height=height,
                distanceToCamera=distance,
                layers=2,
                format="jpeg",
                quality=80,
                key="background-right",
                interpolate=True,
            ),
        ]

    def _upsert_head_planes(self, session, force=False):
        """Publish the head planes only when the camera actually produced a frame.

        The render loop ticks at display_fps while the head camera runs near 7.5 Hz,
        so upserting unconditionally re-encodes and re-sends the same JPEG about
        four times per frame, on a wireless link shared with the hand-tracking
        uplink. Nothing is drawn before the first frame, so the first call always
        publishes.
        """
        sequence = getattr(self, "xr_frame_sequence", None)
        if sequence is None:
            elements = self._head_plane_elements()
            if elements:
                session.upsert(elements, to="bgChildren")
            return True
        with sequence.get_lock():
            current = sequence.value
        if not force and current == getattr(self, "xr_frame_sent", None):
            return False
        elements = self._head_plane_elements()
        if not elements:
            return False
        self.xr_frame_sent = current
        session.upsert(elements, to="bgChildren")
        return True

    def _upsert_wrist_panels(self, session, force=False):
        """(Re)publish the wrist panels; called every frame by the render loops
        because a one-shot upsert races the headset's page load.

        Skipped while every panel still holds the frame that was last sent, which
        happens whenever the control loop outruns the wrist cameras.
        """
        elements = self._wrist_panel_elements()
        if not elements:
            return False
        current = {}
        for side, sequence in self.wrist_panel_seq.items():
            with sequence.get_lock():
                current[side] = sequence.value
        if not force and current == getattr(self, "wrist_panel_sent", None):
            return False
        self.wrist_panel_sent = current
        session.upsert(elements, to="bgChildren")
        return True

    def render_torque_hud_to_xr(self, side, image):
        """Publish one torque number strip (BGR) onto the scene overlay."""
        frame = self.torque_hud_frames.get(side)
        if frame is None or image is None:
            return
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            return
        if image.shape[:2] != frame.shape[:2]:
            image = cv2.resize(image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
        frame[:] = image
        sequence = self.torque_hud_seq[side]
        with sequence.get_lock():
            sequence.value += 1

    def _torque_hud_elements(self):
        elements = []
        for side, sign in (("left", -1.0), ("right", 1.0)):
            frame = self.torque_hud_frames.get(side)
            sequence = self.torque_hud_seq.get(side)
            if frame is None or sequence is None:
                continue
            with sequence.get_lock():
                ready = sequence.value > 0
            if not ready:
                continue
            elements.append(
                ImageBackground(
                    frame,
                    aspect=self.torque_hud_aspect,
                    height=self.torque_hud_height,
                    distanceToCamera=self.torque_hud_distance,
                    position=[
                        -self.torque_hud_left_x if side == "left" else self.torque_hud_offset[0],
                        self.torque_hud_offset[1],
                        0.0,
                    ],
                    format="jpeg",
                    quality=70,
                    key=f"torque-hud-{side}",
                )
            )
        return elements

    def _upsert_torque_hud(self, session, force=False):
        elements = self._torque_hud_elements()
        if not elements:
            return False
        current = {}
        for side, sequence in self.torque_hud_seq.items():
            with sequence.get_lock():
                current[side] = sequence.value
        # Resend every display frame. The stereo video plane is upserted on
        # that same loop, and a skipped strip disappears from the headset.
        # These images are 280x48, not the wrist panels.
        self.torque_hud_sent = current
        session.upsert(elements, to="bgChildren")
        return True

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
            self._upsert_head_planes(session)
            self._upsert_wrist_panels(session)
            self._upsert_torque_hud(session)
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
            self._upsert_head_planes(session)
            self._upsert_wrist_panels(session)
            self._upsert_torque_hud(session)
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
                    height = 7,
                    layout="stereo-left-right"
                ),
                to="bgChildren",
            )
            self._upsert_wrist_panels(session)
            self._upsert_torque_hud(session)
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
            self._upsert_torque_hud(session)
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
            self._upsert_head_planes(session)
            self._upsert_wrist_panels(session)
            self._upsert_torque_hud(session)
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
            self._upsert_head_planes(session)
            self._upsert_wrist_panels(session)
            self._upsert_torque_hud(session)
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
            self._upsert_torque_hud(session)
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
            self._upsert_torque_hud(session)
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
        counts = getattr(self, "motion_sample_counts_shared", None)
        if counts is not None:
            with counts.get_lock():
                result.update(dict(zip(
                    ("motion_events", "motion_begun", "motion_committed",
                     "motion_stamped", "motion_stamp_zero",
                     "snapshot_ok", "snapshot_none", "snapshot_inflight"),
                    counts[:],
                )))
        bins = getattr(self, "motion_interval_bins_shared", None)
        if bins is not None:
            with bins.get_lock():
                values = list(bins[:])
            result["motion_interval_ms"] = {
                label: count
                for label, count in zip(bin_labels(MOTION_INTERVAL_EDGES_MS), values) if count
            }
        lags = getattr(self, "event_loop_lag_bins_shared", None)
        if lags is not None:
            with lags.get_lock():
                values = list(lags[:])
            result["event_loop_lag_ms"] = {
                label: count
                for label, count in zip(bin_labels(EVENT_LOOP_LAG_EDGES_MS), values) if count
            }
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

    def _append_motion_sample(self):
        """Push this callback's sample onto the ordered queue.

        Reads back the shared arrays rather than plumbing the values through the
        loop above, so a slot is byte-identical to what get_hand_motion_snapshot
        would have returned at this instant -- including a side that is holding
        its previous pose because its hand is missing.
        """
        data = getattr(self, "motion_queue_data_shared", None)
        if data is None:
            return
        # The per-side timestamps have no property; they live in the shared values.
        with self.left_hand_timestamp_shared.get_lock():
            left_timestamp = self.left_hand_timestamp_shared.value
        with self.right_hand_timestamp_shared.get_lock():
            right_timestamp = self.right_hand_timestamp_shared.value
        row = [self.motion_data_timestamp, left_timestamp, right_timestamp]
        for name in MOTION_QUEUE_FIELDS:
            row.extend(np.asarray(getattr(self, name), dtype=np.float64).ravel())
        if len(row) != MOTION_QUEUE_STRIDE:
            return
        with self.motion_queue_head_shared.get_lock():
            index = self.motion_queue_head_shared.value
            self.motion_queue_head_shared.value = index + 1
        slot = index % MOTION_QUEUE_SLOTS
        base = slot * MOTION_QUEUE_STRIDE
        with data.get_lock():
            data[base:base + MOTION_QUEUE_STRIDE] = row
        # Written last: a reader that sees this mark knows the slot is complete.
        with self.motion_queue_marks_shared.get_lock():
            self.motion_queue_marks_shared[slot] = index

    def pop_hand_motion_sample(self):
        """Oldest sample this consumer has not taken yet, or None if caught up.

        Each caller advances the single tail, so one consumer must own the queue;
        a second consumer that keeps up with the newest frame should stay on
        get_hand_motion_snapshot instead.
        """
        data = getattr(self, "motion_queue_data_shared", None)
        if data is None:
            return None
        with self.motion_queue_head_shared.get_lock():
            head = self.motion_queue_head_shared.value
        with self.motion_queue_tail_shared.get_lock():
            tail = self.motion_queue_tail_shared.value
        # Bound the latency: a consumer that fell behind is resynchronised rather
        # than fed a history the operator has already moved past.
        if head - tail > MOTION_QUEUE_MAX_BACKLOG:
            tail = head - MOTION_QUEUE_MAX_BACKLOG
            with self.motion_queue_tail_shared.get_lock():
                self.motion_queue_tail_shared.value = tail
        if tail >= head:
            return None
        slot = tail % MOTION_QUEUE_SLOTS
        with self.motion_queue_marks_shared.get_lock():
            if self.motion_queue_marks_shared[slot] != tail:
                # The ring wrapped over this sample before it was read.
                with self.motion_queue_tail_shared.get_lock():
                    self.motion_queue_tail_shared.value = tail + 1
                return None
        base = slot * MOTION_QUEUE_STRIDE
        with data.get_lock():
            row = list(data[base:base + MOTION_QUEUE_STRIDE])
        with self.motion_queue_marks_shared.get_lock():
            if self.motion_queue_marks_shared[slot] != tail:
                return None
        with self.motion_queue_tail_shared.get_lock():
            self.motion_queue_tail_shared.value = tail + 1
        snapshot = {
            "motion_data_timestamp": row[0],
            "left_hand_timestamp": row[1],
            "right_hand_timestamp": row[2],
            "motion_data_ready": True,
            "motion_sample_seq": tail,
        }
        cursor = 3
        for name, shape in zip(MOTION_QUEUE_FIELDS, MOTION_QUEUE_SHAPES):
            size = int(np.prod(shape))
            snapshot[name] = np.asarray(row[cursor:cursor + size], dtype=np.float64).reshape(shape)
            cursor += size
        # Gestures are low bandwidth and 25 ms of lag on a pinch is not felt, so
        # they stay on the newest value rather than costing queue space.
        for name in ("left_hand_pinch", "left_hand_pinchValue",
                     "left_hand_squeeze", "left_hand_squeezeValue",
                     "right_hand_pinch", "right_hand_pinchValue",
                     "right_hand_squeeze", "right_hand_squeezeValue"):
            snapshot[name] = getattr(self, name)
        return snapshot

    def get_hand_motion_snapshot(self, include_orientations=False, max_attempts=3):
        for _ in range(max_attempts):
            with self.motion_sample_seq_shared.get_lock():
                seq_before = self.motion_sample_seq_shared.value
            if seq_before % 2 != 0:
                # A callback is mid-write. The retry spins with no sleep, so the three
                # attempts normally observe the same value; if this fires often the
                # control loop falls back to an older snapshot.
                self._count_motion(7)
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
                self._count_motion(5)
                return snapshot
        self._count_motion(6)
        return None
