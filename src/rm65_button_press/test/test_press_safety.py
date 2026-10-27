import pytest

from rm65_button_press.press_button import PressButton


class FakePublisher:

    def __init__(self, subscribers=1):
        self.subscribers = subscribers
        self.messages = []

    def get_subscription_count(self):
        return self.subscribers

    def publish(self, message):
        self.messages.append(message)


class FakeLogger:

    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, message):
        self.errors.append(message)

    def warn(self, message):
        self.warnings.append(message)


def make_node(motion_active):
    node = PressButton.__new__(PressButton)
    node.motion_active = motion_active
    node.stop_sent = False
    node.exit_code = None
    node.move_stop_pub = FakePublisher()
    logger = FakeLogger()
    node.get_logger = lambda: logger
    return node, logger


def test_safe_stop_publishes_true_once():
    node, _ = make_node(True)

    assert node.safe_stop('test timeout') is True
    assert node.safe_stop('duplicate') is False
    assert len(node.move_stop_pub.messages) == 1
    assert node.move_stop_pub.messages[0].data is True


def test_safe_stop_does_nothing_when_not_moving():
    node, _ = make_node(False)

    assert node.safe_stop('idle failure') is False
    assert node.move_stop_pub.messages == []


def test_fail_stops_active_motion_and_sets_exit_code():
    node, logger = make_node(True)

    node.fail(2, 'return timeout')

    assert node.exit_code == 2
    assert len(node.move_stop_pub.messages) == 1
    assert logger.errors == ['return timeout']


def test_press_failure_stops_instead_of_starting_retreat():
    node, _ = make_node(True)
    node.press_timeout = 20.0
    node.segment_status = lambda timeout: 'timeout'
    node.start_retreat = lambda reason: pytest.fail(
        'press failure must not start a new retreat command')

    node.tick_press()

    assert node.exit_code == 1
    assert len(node.move_stop_pub.messages) == 1
