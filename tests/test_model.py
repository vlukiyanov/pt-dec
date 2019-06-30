from ptdec.model import predict, train
import torch
from torch.utils.data import TensorDataset
from unittest.mock import MagicMock, Mock


def test_train_with_prediction():
    model = Mock()
    model.return_value = torch.zeros(100, 100).requires_grad_()
    model.cluster_number = 10
    model.encoder.return_value = torch.zeros(100, 100)
    model.state_dict.return_value = MagicMock()
    optimizer = Mock()
    dataset = TensorDataset(torch.zeros(100, 100), torch.zeros(100, 1))
    train(
        dataset=dataset,
        model=model,
        epochs=1,
        batch_size=100,
        optimizer=optimizer,
        cuda=False
    )
    assert model.call_count == 2


def test_predict():
    autoencoder = Mock()
    autoencoder.return_value = torch.zeros(10, 100)
    dataset = TensorDataset(torch.zeros(100, 100), torch.zeros(100, 1))
    output = predict(dataset, autoencoder, batch_size=10, cuda=False)
    assert autoencoder.call_count == 10
    assert output.shape == (100,)
