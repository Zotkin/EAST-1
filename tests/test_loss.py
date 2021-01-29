import torch
import pytest

from loss import PODFlatLoss, PODSpacialLoss, PODLoss, filter_tensor


@pytest.fixture()
def data():
    a = torch.randn((10, 32, 64, 64))
    b = torch.randn((10, 32, 64, 64))
    return a,b


@pytest.fixture()
def featuremaps(data):
    return [*data, *data, *data], [*data, *data, *data]


@pytest.fixture()
def scoremaps():
    return torch.randn((10, 512, 512)), torch.randn((10, 512, 512))


@pytest.fixture()
def memory_flags():
    return torch.tensor([True if i % 2 == 0 else False for i in range(10)])


def test_filter_tensor(scoremaps, memory_flags):
    result = filter_tensor(scoremaps[0], memory_flags)
    assert result.shape[0] == torch.sum(memory_flags).item()

def test_PODFlatLoss():

    loss_object = PODFlatLoss()
    a = torch.randn((10, 512, 512))
    b = torch.randn((10, 512, 512))

    loss = loss_object(a,b)

    assert type(loss) == torch.Tensor
    assert type(loss.item()) == float


@pytest.mark.parametrize("dim", [1,2])
def test_PODSpacialLoss(dim, data):
    loss_object = PODSpacialLoss(dim)
    a,b = data
    loss = loss_object(a, b)
    assert type(loss) == torch.Tensor
    assert type(loss.item()) == float


def test_PODLoss(featuremaps, scoremaps, memory_flags):
    featuremaps_a, featuremaps_b = featuremaps
    scoremaps_a, scoremaps_b = scoremaps
    loss_object = PODLoss()
    loss = loss_object(featuremaps_a, featuremaps_b, scoremaps_a, scoremaps_b, memory_flags)
    assert len(loss) == 3
    for term in loss:
        assert type(term) == torch.Tensor
        assert type(term.item()) == float

def test_PODLoss_zero_memory_flag(featuremaps, scoremaps):
    memory_flags = torch.tensor([False for _ in range(scoremaps[0].shape[0])])
    featuremaps_a, featuremaps_b = featuremaps
    scoremaps_a, scoremaps_b = scoremaps
    loss_object = PODLoss()
    loss = loss_object(featuremaps_a, featuremaps_b, scoremaps_a, scoremaps_b, memory_flags)
    assert len(loss) == 3
    for term in loss:
        assert type(term) == torch.Tensor
        assert type(term.item()) == float