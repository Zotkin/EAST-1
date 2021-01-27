import torch
import pytest

from loss import PODFlatLoss, PODSpacialLoss, PODLoss

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


def test_PODLoss(featuremaps, scoremaps):
    featuremaps_a, featuremaps_b = featuremaps
    scoremaps_a, scoremaps_b = scoremaps
    loss_object = PODLoss()
    loss = loss_object(featuremaps_a, featuremaps_b, scoremaps_a, scoremaps_b)
    assert type(loss) == torch.Tensor
    assert type(loss.item()) == float