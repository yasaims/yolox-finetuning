from yoloxft.v1 import trainer_pb2, trainer_pb2_grpc

RPC_METHODS = {
    "ListDatasets",
    "GetDefaultHyperparameters",
    "StartTraining",
    "GetCurrentJob",
    "StreamTrainingEvents",
    "StopTraining",
    "ListModels",
    "GetModel",
    "DeleteModel",
    "EvaluateModel",
    "Predict",
}


def test_servicer_exposes_all_rpc_methods() -> None:
    service = trainer_pb2.DESCRIPTOR.services_by_name["TrainerService"]
    assert {m.name for m in service.methods} == RPC_METHODS
    for name in RPC_METHODS:
        assert callable(getattr(trainer_pb2_grpc.TrainerServiceServicer, name))
