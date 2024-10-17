from temibox.tracker import Tracker

#######################
# Mocks
#######################

class CustomTracker(Tracker):
    def __init__(self, logs: list[str]):
        super().__init__()
        self._logs = logs

    def log(self, message: str):
        super().log(message)
        self._logs.append(message)


#######################
# Tests
#######################

def test_tracker():
    log_messages = []
    progress_messages = []

    tracker = CustomTracker(log_messages)
    tracker.add_progress_callback(lambda i, total, message: progress_messages.append((i, total, message)))

    tracker.task("Test-1")
    tracker.log("1->2")
    tracker.task("Test-2")
    tracker.log("2->3")
    tracker.task("Test-3")

    assert log_messages == ["Starting task 'Test-1'",
                            "1->2",
                            "Starting task 'Test-2'",
                            "2->3",
                            "Starting task 'Test-3'"], "Incorrect log entries"

    total = 5
    for i in range(total):
        tracker.progress(i + 1, total, f"Unit Test in progress {i+1}")


    assert progress_messages == [(1, 5, 'Unit Test in progress 1'),
                                 (2, 5, 'Unit Test in progress 2'),
                                 (3, 5, 'Unit Test in progress 3'),
                                 (4, 5, 'Unit Test in progress 4'),
                                 (5, 5, 'Unit Test in progress 5')], "Progres entries are wrong"

    log_messages_callback = []
    tracker.add_log_callback(lambda m: log_messages_callback.append(m))

    tracker.log("Log callback initiated")

    log_copy      = log_messages_callback.copy()
    progress_copy = progress_messages.copy()

    tracker.clear_log_callbacks()
    tracker.clear_progress_callbacks()

    for i in range(10, 10+total):
        tracker.log(f"Test {i+1}")
        tracker.progress(i + 1, 10+total, f"Unit Test in progress {i + 1}")

    assert log_messages_callback == log_copy, "Log callback was active"
    assert progress_messages == progress_copy, "Progress callback was active"