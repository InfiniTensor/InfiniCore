import infinicore.device
from infinicore.lib import _infinicore


class DeviceEvent:
    """A device event for timing operations and synchronization across devices.

    Similar to torch.cuda.Event, this class provides functionality to:
    - Record events on specific device streams
    - Synchronize with events
    - Measure elapsed time between events
    - Query event completion status
    - Make streams wait for events

    Args:
        device: Target device for this event. If None, uses current device.
        flags: Event creation flags (e.g., for timing, blocking sync). Default is 0.
        enable_timing: Whether the event should be created with timing enabled.
    """

    def __init__(self, device=None, enable_timing=True, flags=0):
        if not enable_timing:
            # You might want to handle this differently based on your flag system
            flags = flags  # Adjust flags if timing is disabled

        if device is None:
            # Use current device
            if flags == 0:
                self._underlying = _infinicore.DeviceEvent()
            else:
                self._underlying = _infinicore.DeviceEvent(flags)
        elif flags == 0:
            # Construct with device only
            self._underlying = _infinicore.DeviceEvent(device._underlying)
        else:
            # Construct with both device and flags
            self._underlying = _infinicore.DeviceEvent(device._underlying, flags)

    def record(self, stream=None):
        """Record the event.

        Args:
            stream: Stream to record the event on. If None, uses current stream.
        """
        if stream is None:
            self._underlying.record()
        else:
            self._underlying.record(stream)

    def synchronize(self):
        """Wait for the event to complete (blocking)."""
        self._underlying.synchronize()

    def query(self):
        """Check if the event has been completed.

        Returns:
            bool: True if completed, False otherwise.
        """
        return self._underlying.query()

    def elapsed_time(self, other):
        """Calculate elapsed time between this event and another event.

        Args:
            other: The other DeviceEvent to compare with

        Returns:
            float: Elapsed time in milliseconds between this event and the other event

        Raises:
            RuntimeError: If events are on different devices or not recorded
        """
        return self._underlying.elapsed_time(other._underlying)

    def wait(self, stream=None):
        """Make a stream wait for this event to complete.

        Args:
            stream: Stream to make wait for this event. If None, uses current stream.
        """
        self._underlying.wait(stream)

    @property
    def device(self):
        """Get the device where this event was created."""
        return infinicore.device._from_infinicore_device(self._underlying.device)

    @property
    def is_recorded(self):
        """Check if the event has been recorded."""
        return self._underlying.is_recorded

    def __repr__(self):
        return f"DeviceEvent(device={self.device}, recorded={self.is_recorded})"
