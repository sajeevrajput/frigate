import datetime
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from multiprocessing import Queue, Value
from multiprocessing.synchronize import Event as MpEvent

import numpy as np
import zmq

from frigate.comms.object_detector_signaler import (
    ObjectDetectorPublisher,
    ObjectDetectorSubscriber,
)
from frigate.config import FrigateConfig
from frigate.const import PROCESS_PRIORITY_HIGH
from frigate.detectors import create_detector
from frigate.detectors.detector_config import (
    BaseDetectorConfig,
    InputDTypeEnum,
    ModelConfig,
)
from frigate.util.builtin import EventsPerSecond, load_labels
from frigate.util.image import SharedMemoryFrameManager, UntrackedSharedMemory
from frigate.util.process import FrigateProcess

from .util import tensor_transform

logger = logging.getLogger(__name__)


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, tensor_input, threshold: float = 0.4):
        pass


class BaseLocalDetector(ObjectDetector):
    def __init__(
        self,
        detector_config: BaseDetectorConfig = None,
        labels: str = None,
    ):
        self.fps = EventsPerSecond()
        if labels is None:
            self.labels = {}
        else:
            self.labels = load_labels(labels)

        if detector_config:
            self.input_transform = tensor_transform(detector_config.model.input_tensor)

            self.dtype = detector_config.model.input_dtype
        else:
            self.input_transform = None
            self.dtype = InputDTypeEnum.int

        self.detect_api = create_detector(detector_config)

    def _transform_input(self, tensor_input: np.ndarray) -> np.ndarray:
        if self.input_transform:
            tensor_input = np.transpose(tensor_input, self.input_transform)

        if self.dtype == InputDTypeEnum.float:
            tensor_input = tensor_input.astype(np.float32)
            tensor_input /= 255
        elif self.dtype == InputDTypeEnum.float_denorm:
            tensor_input = tensor_input.astype(np.float32)

        return tensor_input

    def detect(self, tensor_input: np.ndarray, threshold=0.4):
        detections = []

        raw_detections = self.detect_raw(tensor_input)

        for d in raw_detections:
            if int(d[0]) < 0 or int(d[0]) >= len(self.labels):
                logger.warning(f"Raw Detect returned invalid label: {d}")
                continue
            if d[1] < threshold:
                break
            detections.append(
                (self.labels[int(d[0])], float(d[1]), (d[2], d[3], d[4], d[5]))
            )
        self.fps.update()
        return detections


class LocalObjectDetector(BaseLocalDetector):
    def detect_raw(self, tensor_input: np.ndarray):
        tensor_input = self._transform_input(tensor_input)
        return self.detect_api.detect_raw(tensor_input=tensor_input)


class AsyncLocalObjectDetector(BaseLocalDetector):
    def async_send_input(self, tensor_input: np.ndarray, connection_id: str):
        tensor_input = self._transform_input(tensor_input)
        return self.detect_api.send_input(connection_id, tensor_input)

    def async_receive_output(self):
        return self.detect_api.receive_output()


class DetectorRunner(FrigateProcess):
    def __init__(
        self,
        name,
        detection_queue: Queue,
        cameras: list[str],
        avg_speed: Value,
        start_time: Value,
        config: FrigateConfig,
        detector_config: BaseDetectorConfig,
        stop_event: MpEvent,
    ) -> None:
        super().__init__(stop_event, PROCESS_PRIORITY_HIGH, name=name, daemon=True)
        self.detection_queue = detection_queue
        self.cameras = cameras
        self.avg_speed = avg_speed
        self.start_time = start_time
        self.config = config
        self.detector_config = detector_config
        self.outputs: dict = {}
        self.BATCH_SIZE = 8

    def create_output_shm(self, name: str):
        out_shm = UntrackedSharedMemory(name=f"out-{name}", create=False)
        out_np = np.ndarray((20, 6), dtype=np.float32, buffer=out_shm.buf)
        self.outputs[name] = {"shm": out_shm, "np": out_np}

    def run(self) -> None:
        self.pre_run_setup(self.config.logger)

        frame_manager = SharedMemoryFrameManager()
        object_detector = LocalObjectDetector(detector_config=self.detector_config)
        detector_publisher = ObjectDetectorPublisher()

        while not self.stop_event.is_set():
            detection_queue_size = self.detection_queue.qsize()
            # self.logger.info(f"[{datetime.datetime.now().timestamp()}]: detection_queue size: {self.detection_queue.qsize()}")


            input_frames = []
            connection_ids = []

            # pull items from queue of max size equal to BATCH_SIZE
            for i in range(min(detection_queue_size, self.BATCH_SIZE)):
                try:
                    connection_id = self.detection_queue.get(timeout=1)
                except queue.Empty:
                    continue
                self.create_output_shm(connection_id)   # this assumes, all connection ids to be batched are unique
                input_frame = frame_manager.get(
                    connection_id,
                    (
                        1,
                        self.detector_config.model.height,
                        self.detector_config.model.width,
                        3,
                    ),
                )

                if input_frame is None:
                    logger.warning(f"Failed to get frame {connection_id} from SHM")
                    continue
                input_frames.append(input_frame)
                connection_ids.append(connection_id)
            
            if not input_frames:
                continue
            
            # batch the inputs
            input_tensor = np.vstack(input_frames)
            

            # detect and send the output
            self.start_time.value = datetime.datetime.now().timestamp()
            logger.warning(f"[{datetime.datetime.now().timestamp()}]: Input tensor shape: {input_tensor.shape}")
            detections = object_detector.detect_raw(input_tensor)
            logger.warning(f"[{datetime.datetime.now().timestamp()}]: Detections: {detections.shape}")
            duration = datetime.datetime.now().timestamp() - self.start_time.value
            logger.info(
                f"[{datetime.datetime.now().timestamp()}]: Inference took {duration:.3f}s for {connection_id}"
            )
            # release input buffer
            for i, connection_id in enumerate(connection_ids):
                frame_manager.close(connection_id)
                
                # fetch outputs and publish
                if connection_id not in self.outputs:
                    self.create_output_shm(connection_id)

                self.outputs[connection_id]["np"][:] = detections[i][:]
                detector_publisher.publish(connection_id)
                self.start_time.value = 0.0

                self.avg_speed.value = (self.avg_speed.value * 9 + duration) / 10

        detector_publisher.stop()
        logger.info("Exited detection process...")


class AsyncDetectorRunner(FrigateProcess):
    def __init__(
        self,
        name,
        detection_queue: Queue,
        cameras: list[str],
        avg_speed: Value,
        start_time: Value,
        config: FrigateConfig,
        detector_config: BaseDetectorConfig,
        stop_event: MpEvent,
    ) -> None:
        super().__init__(stop_event, PROCESS_PRIORITY_HIGH, name=name, daemon=True)
        self.detection_queue = detection_queue
        self.cameras = cameras
        self.avg_speed = avg_speed
        self.start_time = start_time
        self.config = config
        self.detector_config = detector_config
        self.outputs: dict = {}
        self._frame_manager: SharedMemoryFrameManager | None = None
        self._publisher: ObjectDetectorPublisher | None = None
        self._detector: AsyncLocalObjectDetector | None = None
        self.send_times = deque()

    def create_output_shm(self, name: str):
        out_shm = UntrackedSharedMemory(name=f"out-{name}", create=False)
        out_np = np.ndarray((20, 6), dtype=np.float32, buffer=out_shm.buf)
        self.outputs[name] = {"shm": out_shm, "np": out_np}

    def _detect_worker(self) -> None:
        logger.info("Starting Detect Worker Thread")
        while not self.stop_event.is_set():
            try:
                connection_id = self.detection_queue.get(timeout=1)
            except queue.Empty:
                continue

            input_frame = self._frame_manager.get(
                connection_id,
                (
                    1,
                    self.detector_config.model.height,
                    self.detector_config.model.width,
                    3,
                ),
            )

            if input_frame is None:
                logger.warning(f"Failed to get frame {connection_id} from SHM")
                continue

            # mark start time and send to accelerator
            self.send_times.append(time.perf_counter())
            self._detector.async_send_input(input_frame, connection_id)

    def _result_worker(self) -> None:
        logger.info("Starting Result Worker Thread")
        while not self.stop_event.is_set():
            connection_id, detections = self._detector.async_receive_output()

            if not self.send_times:
                # guard; shouldn't happen if send/recv are balanced
                continue
            ts = self.send_times.popleft()
            duration = time.perf_counter() - ts

            # release input buffer
            self._frame_manager.close(connection_id)

            if connection_id not in self.outputs:
                self.create_output_shm(connection_id)

            # write results and publish
            if detections is not None:
                self.outputs[connection_id]["np"][:] = detections[:]
            self._publisher.publish(connection_id)

            # update timers
            self.avg_speed.value = (self.avg_speed.value * 9 + duration) / 10
            self.start_time.value = 0.0

    def run(self) -> None:
        self.pre_run_setup(self.config.logger)

        self._frame_manager = SharedMemoryFrameManager()
        self._publisher = ObjectDetectorPublisher()
        self._detector = AsyncLocalObjectDetector(detector_config=self.detector_config)

        for name in self.cameras:
            self.create_output_shm(name)

        t_detect = threading.Thread(target=self._detect_worker, daemon=True)
        t_result = threading.Thread(target=self._result_worker, daemon=True)
        t_detect.start()
        t_result.start()

        while not self.stop_event.is_set():
            time.sleep(0.5)

        self._publisher.stop()
        logger.info("Exited async detection process...")


class ObjectDetectProcess:
    def __init__(
        self,
        name: str,
        detection_queue: Queue,
        cameras: list[str],
        config: FrigateConfig,
        detector_config: BaseDetectorConfig,
        stop_event: MpEvent,
    ):
        self.name = name
        self.cameras = cameras
        self.detection_queue = detection_queue
        self.avg_inference_speed = Value("d", 0.01)
        self.detection_start = Value("d", 0.0)
        self.detect_process: FrigateProcess | None = None
        self.config = config
        self.detector_config = detector_config
        self.stop_event = stop_event
        self.start_or_restart()

    def stop(self):
        # if the process has already exited on its own, just return
        if self.detect_process and self.detect_process.exitcode:
            return
        self.detect_process.terminate()
        logging.info("Waiting for detection process to exit gracefully...")
        self.detect_process.join(timeout=30)
        if self.detect_process.exitcode is None:
            logging.info("Detection process didn't exit. Force killing...")
            self.detect_process.kill()
            self.detect_process.join()
        logging.info("Detection process has exited...")

    def start_or_restart(self):
        self.detection_start.value = 0.0
        if (self.detect_process is not None) and self.detect_process.is_alive():
            self.stop()

        # Async path for MemryX
        if self.detector_config.type == "memryx":
            self.detect_process = AsyncDetectorRunner(
                f"frigate.detector:{self.name}",
                self.detection_queue,
                self.cameras,
                self.avg_inference_speed,
                self.detection_start,
                self.config,
                self.detector_config,
                self.stop_event,
            )
        else:
            self.detect_process = DetectorRunner(
                f"frigate.detector:{self.name}",
                self.detection_queue,
                self.cameras,
                self.avg_inference_speed,
                self.detection_start,
                self.config,
                self.detector_config,
                self.stop_event,
            )
        self.detect_process.start()


class RegionSHM:
    def __init__(self, 
                 name: str, input_array_shape):
        self.inp_shm_name = name
        self.out_shm_name = f"out-{name}"
        
        # create input shm
        try:
            self.inp_shm = UntrackedSharedMemory(name=self.inp_shm_name, create=True, size=input_array_shape[0] * input_array_shape[1] * 3)
        except FileExistsError:
            self.inp_shm = UntrackedSharedMemory(name=self.inp_shm_name)
        self.inp_shm_np = np.ndarray(
            (1, input_array_shape[0], input_array_shape[1], 3),
            dtype=np.uint8,
            buffer=self.inp_shm.buf,
        )
        
        # create detections shm
        try:
            self.out_shm = UntrackedSharedMemory(name=self.out_shm_name, create=True, size=20 * 6 * 4)
        except FileExistsError:
            self.out_shm = UntrackedSharedMemory(name=self.out_shm_name)
        self.out_shm_np = np.ndarray((20, 6), dtype=np.float32, buffer=self.out_shm.buf)
        
        # create subscriber
        self.detector_subscriber = ObjectDetectorSubscriber(self.inp_shm_name)

class UntrackedSharedMemoryPool:
    """Create a pool of shared memory regions for object detection inputs and outputs.
    This only creates the shared memory regions and does book keeping. It does not manage their usage. 
    Use SharedMemoryFrameManager to manage usage. 
    """
    def __init__(self, name: str, input_array_shape, pool_size: int):
        self.name = name
        self.inp_shape = input_array_shape
        self.pool_size = pool_size
        self.shm_pool: dict[int, RegionSHM] = {}  # dict of inp output shm name pair for each region
        for i in range(pool_size):
            shm_name = f"{self.name}-{i}"  # e.g. 'camera1-0'
            self.shm_pool[shm_name] = RegionSHM(name=shm_name, input_array_shape=input_array_shape)

    def __getitem__(self, index: int) -> RegionSHM:
        """index here gives an impression of a shm region from the pool of shm regions created for a camera"""
        return self.shm_pool[index]

    def __setitem__(self, index: int, value: RegionSHM) -> None:
        """Set the shm region info at the given index."""
        raise NotImplementedError("Setting region shm is not supported.")
    
    def cleanup_all(self):
        for reg_shm in self.shm_pool.values():
            reg_shm.detector_subscriber.stop()            
            reg_shm.inp_shm.unlink()
            reg_shm.out_shm.unlink()


class RemoteObjectDetector:
    def __init__(
        self,
        name: str,
        labels: dict[int, str],
        detection_queue: Queue,
        model_config: ModelConfig,
        stop_event: MpEvent,
        shm_pool_size: int = 4,
    ):
        self.labels = labels
        self.name = name
        self.fps = EventsPerSecond()
        self.detection_queue = detection_queue
        self.stop_event = stop_event
        self.shm_pool_size = shm_pool_size
        self.shm_pool = UntrackedSharedMemoryPool(name=self.name, input_array_shape=(model_config.height, model_config.width), pool_size=shm_pool_size) # Pool of SHM regions

    def detect(self, tensor_inputs, threshold=0.4):
        detections = []

        if self.stop_event.is_set():
            return detections

        # Drain any stale detection results from the ZMQ buffer before making a new request
        # This prevents reading detection results from a previous request
        # NOTE: This should never happen, but can in some rare cases
        while True:
            try:
                for i in range(self.shm_pool_size):
                    self.shm_pool[f"{self.name}-{i}"].detector_subscriber.socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        logger.warning(f"RemoteObjectDetector: {self.name} Processing {len(tensor_inputs)} inputs with SHM pool size {self.shm_pool_size}")
        # map output detections to their corresponding input regions
        region_detections_map = defaultdict(list)
        
        # break inputs into chunks of shm_pool_size
        for i in range(0,len(tensor_inputs),self.shm_pool_size):
            names = []
            # track current region shm names in chunk to original input index
            pool_region_map = {}
            # process shm_pool_size inputs at a time
            for j in range(self.shm_pool_size):
                region_shm = self.shm_pool[f"{self.name}-{j}"]
                index = i + j
                if index < len(tensor_inputs):
                    # copy input to shared memory
                    region_shm.inp_shm_np[:] = tensor_inputs[index][:]
                    names.append(region_shm.inp_shm_name)

                    self.detection_queue.put(region_shm.inp_shm_name)
                    pool_region_map[region_shm.inp_shm_name] = index
            
            # monitor for outputs for each region, post proc immediately if one's available.
            pending_regions = names.copy()
            start_time = time.time()
            TIMEOUT = 5.0  # seconds timeout for all regions in the pool to be processed
            
            while pending_regions and (time.time() - start_time) < TIMEOUT:
                for r in list(pending_regions):
                    region_shm = self.shm_pool[r]
                    result = region_shm.detector_subscriber.check_for_update(timeout=0.001)
                    if result is not None:
                        region_detection=[]
                        for d in region_shm.out_shm_np:
                            if d[1] < threshold:
                                break
                            region_detection.append((self.labels[int(d[0])], float(d[1]), (d[2], d[3], d[4], d[5])))
                            # assign detections to correct region
                        region_detections_map[pool_region_map[region_shm.inp_shm_name]] = region_detection
                        self.fps.update()
                        pending_regions.remove(r)
            # end of while pending_regions
            if pending_regions:
                self.logger.warning(f"{self.name}: Timeout waiting for regions: {pending_regions}")
        
        logger.debug(f"{self.name}: Region detections keys: {region_detections_map.keys()}")
        logger.debug(f"{self.name}: Region detections : {region_detections_map}")

        # return detections in order of arrived input tensor list
        for index in sorted(region_detections_map.keys()):
            detections.append(region_detections_map[index])
        logger.debug(f"{self.name}: Total detections returned: {detections}")
        return detections

    def cleanup(self):
        self.shm_pool.cleanup_all()
