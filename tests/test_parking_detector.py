import numpy as np
from src.parking_detector import ParkingSlot, slot_occupancy


def test_slot_occupancy():
    slots = [ParkingSlot(1, np.array([[0, 0], [100, 0], [100, 100], [0, 100]]))]
    assert slot_occupancy(slots, [(10, 10, 90, 90)], threshold=0.2)[1] is True
    assert slot_occupancy(slots, [(200, 200, 300, 300)], threshold=0.2)[1] is False
