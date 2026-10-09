import numpy as np
import pytest

from resolution_sweep.degrade import degrade, effective_gsd, target_size


def test_target_sizes_for_650px_chip():
    levels = [0.3, 0.45, 0.6, 0.9, 1.2, 1.8, 2.4, 3.6, 5.0, 10.0]
    got = [target_size(650, 0.3, g) for g in levels]
    assert got == [650, 433, 325, 217, 163, 108, 81, 54, 39, 20]


def test_effective_gsd_reflects_integer_rounding():
    assert effective_gsd(650, 0.3, 10.0) == pytest.approx(9.75)
    assert effective_gsd(650, 0.3, 0.6) == pytest.approx(0.6)


@pytest.mark.parametrize("method", ["area", "blur_resize"])
def test_shape_dtype_and_identity(method):
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    out = degrade(img, 0.3, 1.2, method)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(degrade(img, 0.3, 0.3, method), img)


@pytest.mark.parametrize("method", ["area", "blur_resize"])
def test_constant_image_is_preserved(method):
    img = np.full((80, 80, 3), (200, 60, 40), dtype=np.uint8)
    out = degrade(img, 0.3, 2.4, method)
    assert np.abs(out.astype(int) - img.astype(int)).max() <= 1


@pytest.mark.parametrize("method", ["area", "blur_resize"])
def test_checkerboard_collapses_to_mid_grey_at_two_times_gsd(method):
    yy, xx = np.indices((64, 64))
    board = (((yy + xx) % 2) * 255).astype(np.uint8)
    img = np.dstack([board] * 3)
    out = degrade(img, 0.3, 0.6, method)
    interior = out[8:-8, 8:-8].astype(float)
    assert np.abs(interior - 127.5).max() <= 3


def test_detail_is_lost_monotonically():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    spreads = [degrade(img, 0.3, g, "area").std() for g in (0.6, 1.2, 2.4, 4.8)]
    assert spreads == sorted(spreads, reverse=True)


def test_invalid_arguments():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        degrade(img, 0.3, 1.0, method="nearest")
    with pytest.raises(ValueError):
        degrade(img.astype(np.float32), 0.3, 1.0)


def test_to_uint8_scaling():
    import numpy as np
    from resolution_sweep.data.spacenet2 import to_uint8
    a = np.array([[[0, 400, 800, 1600]]], dtype=np.uint16)
    out = to_uint8(a, 800)
    assert out.dtype == np.uint8 and abs(int(out[0, 0, 1]) - 128) <= 1
    assert out[0, 0, 0] == 0 and out[0, 0, 2] == 255 and out[0, 0, 3] == 255
    b = np.zeros((1, 1, 1), dtype=np.uint8)
    assert to_uint8(b, None) is b
    import pytest
    with pytest.raises(ValueError):
        to_uint8(a, None)
