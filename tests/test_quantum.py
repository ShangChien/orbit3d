import numpy as np
from orbit3d.quantum.ocelot import ip_objective, frontier_orbitals, OcelotTheory


def test_ip_tuning_uses_positive_vertical_ionization_energy():
    assert ip_objective(-100.0, -99.5, -0.5) == 0
    assert np.isclose(ip_objective(-100.0, -99.5, -0.4), 0.1)


def test_spin_frontiers_take_highest_occupied_and_lowest_unoccupied():
    energy = np.array([[-1.0, -0.4, 0.1], [-0.9, -0.2, 0.2]])
    occupancy = np.array([[1, 1, 0], [1, 0, 0]])
    assert frontier_orbitals(energy, occupancy) == (-0.4, -0.2)
    assert OcelotTheory().xc == 'HYB_GGA_XC_LC_WPBE_WHS'
    assert OcelotTheory().basis == 'def2-svp'
