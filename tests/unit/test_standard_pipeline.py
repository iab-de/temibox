import os
import time
import pytest
from datetime import datetime
from typing import Any, Optional, List

from temibox.traits import Trainable, Transformable, Predictable, Cleanable, Evaluating, Cacheable, Supervising, PipelineStep, Trackable
from temibox.capabilities import CudaCapable, InferenceCapable
from temibox.domain import Document, Triplet, Label, UseCase
from temibox.context import Context, ContextArg
from temibox.pipeline import StandardPipeline
from temibox.prediction import Prediction, RawPrediction
from temibox.tracker import Tracker
from temibox.cache import Cache
from temibox.interfaces import Pipeline

from _support import TestUseCase, TestEvaluator, TestTrainable, TestSupervisor
from _support import TestTrainableTransformable, TestTrainablePredictable, TestTransformableOnly

def test_order():

    t1 = TestTrainable("t1", activities := [])
    t2 = TestTrainableTransformable("t2", activities)
    t3 = TestTrainablePredictable("t3", activities)

    pipeline = StandardPipeline() \
                    .add_usecase(TestUseCase()) \
                    .add_step("t1", t1) \
                    .add_step("t3", t3) \
                    .add_step("t2", t2)

    pipeline.train()
    assert activities == ["t1-train", "t3-train", "t2-train", "t2-transform"], "Train-order does not match"
    activities.clear()

    pipeline.train()
    pipeline.train()
    pipeline.train()

    assert activities == ["t1-train", "t3-train", "t2-train", "t2-transform"] * 3, "Train-order does not match"
    activities.clear()

    pipeline.transform()
    assert activities == ["t2-transform"], "Transform-order does not match"
    activities.clear()

    pipeline.predict()
    assert activities == ["t2-transform", "t3-predict"], "Predict-order does not match"
    activities.clear()

    pipeline.clean()
    assert activities == ["t1-clean"], "Clean-order does not match"
    activities.clear()

    pipeline.evaluate()
    assert activities == [], "Evaluate should be empty"
    activities.clear()

def test_context_arg():
    t1 = TestTrainable("t1", activities:=[])
    t2 = TestTrainableTransformable("t2", activities)

    pipeline = StandardPipeline() \
        .add_usecase(TestUseCase()) \
        .add_step("t1", t1) \
        .add_step("t2", t2)

    out = pipeline.transform(initial_value = (initial_value := 999))

    assert "initial_value" in out, "Initial value not returned"
    assert "transformed_value" in out, "Transformed-value not returned"
    assert isinstance(out["transformed_value"], ContextArg), "Transformed-value should be an instance of ContextArg"
    assert ContextArg.extract(out["transformed_value"], "test-usecase") == initial_value * 2, "Transformed-value incorrect (1/2)"
    assert ContextArg.extract(out["transformed_value"]) == initial_value * 2, "Transformed-value incorrect (2/2)"

def test_invalid_dependencies():

    t1 = TestTrainable("t1", activities := [])
    t2 = TestTrainableTransformable("t2", activities)

    pipeline = StandardPipeline().add_usecase(TestUseCase())

    with pytest.raises(Exception):
        pipeline.add_step("t1", t1, dependencies="t2")

    with pytest.raises(Exception):
        pipeline.add_step("t2", t2, dependencies="t2")

def test_identifier():

    pipeline = StandardPipeline().add_usecase(TestUseCase())

    assert pipeline.identifier == "standard-pipeline"
    pipeline.use_identifier("custom-pipeline")
    assert pipeline.identifier == "custom-pipeline"

def test_adding_usecases():

    pipeline = StandardPipeline().add_usecase(TestUseCase())

    with pytest.raises(Exception):
        pipeline.add_usecase(TestUseCase())

    pipeline.add_usecase(TestUseCase(name="different name"))

def test_get_usecases():
    uc_1 = TestUseCase("usecase 1")
    uc_2 = TestUseCase("usecase 2")

    pipeline = StandardPipeline() \
                .add_usecase(uc_2) \
                .add_usecase(uc_1)

    usecases = pipeline.get_usecases()

    assert usecases[0] == uc_2, "Wrong usecase 0"
    assert usecases[1] == uc_1, "Wrong usecase 1"

def test_add_step():

    uc_1 = TestUseCase("usecase 1")
    uc_2 = TestUseCase("usecase 2")

    t1 = TestTrainable("t1", activities := [])

    pipeline = StandardPipeline()
    with pytest.raises(Exception):
        pipeline.add_step("t1", t1, usecases=[uc_2]) \

    pipeline \
        .add_usecase(uc_2) \
        .add_usecase(uc_1)

    with pytest.raises(Exception):
        new_uc_2 =  TestUseCase("usecase 2")
        pipeline.add_step("t1", t1, usecases=[new_uc_2]) \

    pipeline.add_step("t1", t1, usecases=[uc_2])

    with pytest.raises(Exception):
        pipeline.add_step("t1", t1, usecases=[uc_2])

def test_add_step_supervisor():

    supervised = set()
    ts = TestSupervisor(supervised)

    uc_1 = TestUseCase("usecase 1")
    uc_2 = TestUseCase("usecase 2")
    uc_3 = TestUseCase("usecase 3")

    t1 = TestTrainable("t1", activities := [])
    t2 = TestTrainableTransformable("t2", activities)
    t3 = TestTrainableTransformable("t3", activities)
    t4 = TestTrainable("t4", activities)

    StandardPipeline() \
        .add_usecase(uc_1) \
        .add_usecase(uc_2) \
        .add_usecase(uc_3) \
        .add_step("t1", t1) \
        .add_step("t2", t2) \
        .add_step("t3", t3) \
        .add_step("t4", t4) \
        .add_step("ts", ts, dependencies=["t1", "t4", "t3"])

    assert supervised == {"t1", "t3", "t4"}, "Invalid supervised steps"

def test_add_step_tracker():

    class MyTracker(Tracker):
        pass

    t1 = TestTrainable("t1", activities := [])
    t2 = TestTrainable("t2", activities)

    pipeline = StandardPipeline() \
                .add_usecase(TestUseCase("usecase 1")) \
                .add_step("t1", t1)

    assert not isinstance(pipeline.get_progress_tracker(), MyTracker), "Invalid tracker instance before (pipeline)"
    assert not isinstance(t1.get_progress_tracker(), MyTracker), "Invalid tracker instance before (step)"

    pipeline.use_progress_tracker(MyTracker())
    pipeline.add_step("t2", t2)

    assert isinstance(pipeline.get_progress_tracker(), MyTracker), "Invalid tracker instance after (pipeline)"
    assert isinstance(t1.get_progress_tracker(), MyTracker), "Invalid tracker instance after (step t1)"
    assert isinstance(t2.get_progress_tracker(), MyTracker), "Invalid tracker instance after (step t2)"

def test_get_steps():

    uc_1 = TestUseCase("usecase 1")
    uc_2 = TestUseCase("usecase 2")
    uc_3 = TestUseCase("usecase 3")

    t1 = TestTrainable("t1", activities := [])
    t2 = TestTrainableTransformable("t2", activities)
    t3 = TestTrainablePredictable("t3", activities)
    t4 = TestTrainablePredictable("t4", activities)

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_usecase(uc_2) \
        .add_usecase(uc_3) \
        .add_step("t1", t1) \
        .add_step("t2", t2) \
        .add_step("t3", t3) \
        .add_step("t4", t4)

    steps = pipeline.get_steps()

    assert set(steps.keys()) == {Cleanable, Trainable, Transformable, Predictable}, "Step map keys don't match"
    assert {x[0] for x in steps[Trainable]} == {"t1", "t2", "t3", "t4"}, "Trainable steps incorrect"
    assert {x[0] for x in steps[Transformable]} == {"t2"}, "Transformable steps incorrect"
    assert {x[0] for x in steps[Predictable]} == {"t3", "t4"}, "Predictable steps incorrect"
    assert {x[0] for x in steps[Cleanable]} == {"t1"}, "Cleanable steps incorrect"

def test_get_step_usecases():
    uc_1 = TestUseCase("usecase 1")
    uc_2 = TestUseCase("usecase 2")
    uc_3 = TestUseCase("usecase 3")

    t1 = TestTrainable("t1", activities := [])
    t2 = TestTrainableTransformable("t2", activities)
    t3 = TestTrainableTransformable("t3", activities)
    t4 = TestTrainable("t4", activities)

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_usecase(uc_2) \
        .add_usecase(uc_3) \
        .add_step("t1", t1, usecases=[uc_1, uc_3]) \
        .add_step("t2", t2, usecases=[uc_2]) \
        .add_step("t3", t3)

    assert pipeline.get_step_usecases("t1") == [uc_1, uc_3], "Wrong usecases for t1"
    assert pipeline.get_step_usecases("t2") == [uc_2], "Wrong usecases for t2"
    assert pipeline.get_step_usecases("t3") == [uc_1, uc_2, uc_3], "Wrong usecases for t3"

    with pytest.raises(Exception):
        pipeline.add_step("t4", t4, usecases=[uc_2], dependencies=["t1"])

    with pytest.raises(Exception):
        pipeline.get_step_usecases("t5")

def test_step_dependencies():

    uc_1 = TestUseCase("usecase 1")

    t1 = TestTrainable("t1", activities := [])
    t2 = TestTrainableTransformable("t2", activities)
    t3 = TestTrainablePredictable("t3", activities)
    t4 = TestTrainablePredictable("t4", activities)

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_step("t1", t1) \
        .add_step("t2", t2, dependencies=["t1"]) \
        .add_step("t3", t3, dependencies=["t1"]) \
        .add_step("t4", t4, dependencies=["t2", "t1"])

    assert pipeline.get_step_dependencies("t1") == [], "Invalid dependencies t1"
    assert pipeline.get_step_dependencies("t2") == [t1], "Invalid dependencies t2"
    assert pipeline.get_step_dependencies("t3") == [t1], "Invalid dependencies t3"
    assert pipeline.get_step_dependencies("t4") == [t2, t1], "Invalid dependencies t4"
    assert pipeline.get_step_dependencies("does not exist") == [], "Unknown steps have no dependencies"

def test_get_step():
    uc_1 = TestUseCase("usecase 1")

    t1 = TestTrainable("t1", activities := [])

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_step("t1", t1)

    assert pipeline.get_step("t1") == t1, "Step not returned correctly"
    assert pipeline.get_step("does not exist") is None, "Step should not exist"

def test_get_timestamp():
    uc_1 = TestUseCase("usecase 1")

    t1 = TestTrainable("t1", activities := [], sleep=True)

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_step("t1", t1)

    assert (t := pipeline.init_timestamp) is not None and t > 0, "Init-timestamp not initialized"
    assert pipeline.train_duration_sec == 0, "Train duration should be 0 before training"
    assert pipeline.is_training, "Pipeline should be in training mode"
    pipeline.train()
    assert pipeline.train_duration_sec > 0, "Train duration should be greater 0 after training"
    assert not pipeline.is_training, "Pipeline should not be in training mode"

def test_train():
    uc_1 = TestUseCase("usecase 1")

    t1 = TestTrainable("t1", activities := [])
    t2 = TestTrainableTransformable("t2", activities)
    t3 = TestTrainablePredictable("t3", activities)
    t4 = TestTransformableOnly("t4", activities)

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_step("t1", t1) \
        .add_step("t2", t2) \
        .add_step("t3", t3) \
        .add_step("t4", t4)

    with pytest.raises(Exception):
        pipeline.train(start_at="t4")

    with pytest.raises(Exception):
        pipeline.train(stop_at="t4")

    pipeline.train(start_at="t2")
    pipeline.train(stop_at="t2")
    pipeline.train(start_at="t1", stop_at="t2")

def test_transform_train():
    uc_1 = TestUseCase("usecase 1")
    uc_2 = TestUseCase("usecase 2")
    uc_3 = TestUseCase("usecase 3")

    t1 = TestTrainable("t1", activities := [])
    t2 = TestTrainableTransformable("t2", activities)
    t3 = TestTrainableTransformable("t3", activities)
    t4 = TestTrainable("t4", activities)

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_usecase(uc_3) \
        .add_step("t1", t1, usecases = [uc_3]) \
        .add_step("t2", t2, usecases = [uc_1]) \
        .add_step("t3", t3) \
        .add_step("t4", t4)

    pipeline.transform()
    pipeline.transform(usecases = [uc_1, uc_3])
    pipeline.transform(usecases=[uc_1])
    pipeline.transform(usecases=[uc_3])

    with pytest.raises(Exception):
        pipeline.transform(usecases=[uc_2])

    t2.fail_transform()
    pipeline.transform(usecases=[uc_3])
    with pytest.raises(Exception):
        pipeline.transform(usecases=[uc_1])

def test_predict():
    uc_1 = TestUseCase("usecase 1")
    uc_2 = TestUseCase("usecase 2")
    uc_3 = TestUseCase("usecase 3")

    t1 = TestTrainableTransformable("t1", activities := [])
    t2 = TestTrainablePredictable("t2", activities)
    t3 = TestTrainablePredictable("t3", activities)

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_usecase(uc_2) \
        .add_step("t1", t1) \
        .add_step("t2", t2, usecases=[uc_2], dependencies=[]) \
        .add_step("t3", t3, usecases=[uc_2], dependencies=["t1"])

    with pytest.raises(Exception):
        pipeline.predict(usecases=[uc_3], initial_value = 123)

    out = pipeline.predict(usecases = [uc_1, uc_2], initial_value=(initial_value := 999))

    assert activities == ["t1-transform", "t1-transform", "t2-predict", "t3-predict"], "Incorrect order"
    assert len(out) == 2, "Prediction should contain one item"

    for i, pred in enumerate(out):
        assert pred.model == f"t{i+2}", "Incorrect model"
        assert pred.usecase_name == uc_2.name, "Incorrect usecase"
        assert len(pred.payload_raw) == 3, "Incorrect raw payload"

        for i, (label, label_id, score) in enumerate([["a", 0, 0.25], ["b", 1, 0.50], ["c", 2, 0.75]]):
            assert pred.payload_raw[i].label == label, f"Incorrect label for raw prediction {i+1}"
            assert pred.payload_raw[i].label_id == label_id, f"Incorrect label_id for raw prediction {i + 1}"
            assert abs(pred.payload_raw[i].score - score) < 1e-5, f"Incorrect score for raw prediction {i + 1}"

    assert out[0].payload is None, "Incorrect payload 0"
    assert out[1].payload == initial_value * 2, "Incorrect payload 1"

def test_clean():
    uc_1 = TestUseCase("usecase 1")

    t1 = TestTrainable("t1", activities := [])
    t2 = TestTrainable("t2", activities, fail_clean=True)

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_step("t1", t1) \
        .add_step("t2", t2)

    pipeline.clean()

    assert activities == ["t1-clean", "t2-clean"], "Activities are incorrect"

def test_cache():
    uc_1 = TestUseCase("usecase 1")
    t1 = TestTrainable("t1", activities := [])

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_step("t1", t1)

    p1 = pipeline.cache
    c1 = t1.cache

    assert p1 != c1, "Pipeline and step caches are not the same"

    pipeline.configure_cache(on = True, max_entries=512)
    pipeline.clear_cache()

    c2 = t1.cache
    pipeline.configure_cache(on=True, max_entries=256)

    assert activities == ["t1-configure-cache", "t1-clear-cache", "t1-clear-cache", "t1-configure-cache", "t1-clear-cache"], "Activities are incorrect (1)"
    assert c1 != c2, "Configuring cache should replace the cache object"
    assert pipeline.is_caching, "Pipeline should be caching"
    assert t1.is_caching, "Step should be caching"

    activities.clear()
    pipeline.configure_cache(on=False)
    assert activities == ["t1-configure-cache", "t1-clear-cache"], "Activities are incorrect (2)"
    assert not pipeline.is_caching, "Pipeline should not be caching"
    assert not t1.is_caching, "Step should not be caching"

def test_evaluate():
    uc_1 = TestUseCase("usecase 1")
    uc_2 = TestUseCase("usecase 2")
    uc_3 = TestUseCase("usecase 3")

    t1 = TestTransformableOnly("t1", activities := [])
    t2 = TestEvaluator("t2", activities)

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_usecase(uc_2) \
        .add_step("t1", t1) \
        .add_step("t2", t2, usecases=[uc_2], dependencies=["t1"])

    result = pipeline.evaluate(usecases=[uc_2], initial_value = (initial_value := 123))

    assert "t1" in result and "t2" in result, "Results are per step"
    assert "evaluation_result" in result["t2"], "Evaluation result should have been returned"
    assert isinstance(result["t2"]["evaluation_result"], ContextArg), "Result should be a ContextArg"
    assert ContextArg.extract(result["t2"]["evaluation_result"], uc_1.name)  is None, "First Usecase was not evaluated"
    assert ContextArg.extract(result["t2"]["evaluation_result"], uc_2.name) == initial_value*2*3, "Incorrect evaluation result (1)"
    assert activities == ["t1-evaluate", "t1-transform", "t2-evaluate"], "Activities are incorrect"

    with pytest.raises(Exception):
        pipeline.evaluate(usecases=[uc_3], initial_value=456)

    result_2 = pipeline.evaluate(initial_value=(new_initial_value := 789))
    assert ContextArg.extract(result_2["t2"]["evaluation_result"], uc_2.name) == new_initial_value * 2 * 3, "Incorrect evaluation result (2)"

    t2.fail_evaluations()
    with pytest.raises(Exception):
        pipeline.evaluate(usecases=[uc_2], initial_value=1)

def test_realistic_scenario():

    uc_1 = TestUseCase("usecase 1")
    uc_2 = TestUseCase("usecase 2")
    uc_3 = TestUseCase("usecase 3")

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_usecase(uc_2) \
        .add_usecase(uc_3) \
        .add_step("data",       TestTrainableTransformable("data", activities := [])) \
        .add_step("filter",     TestTransformableOnly("filter", activities)) \
        .add_step("embedder_1", TestTransformableOnly("embedder-1", activities), usecases = [uc_1, uc_2], dependencies = ["data", "filter"]) \
        .add_step("embedder_2", TestTransformableOnly("embedder-2", activities), usecases = [uc_3],       dependencies = ["data", "filter"]) \
        .add_step("model_1",    TestTrainablePredictable("model-1", activities), usecases = [uc_1],       dependencies = ["embedder_1"]) \
        .add_step("model_2",    TestTrainablePredictable("model-2", activities), usecases = [uc_2],       dependencies = ["embedder_1"]) \
        .add_step("model_3",    TestTrainablePredictable("model-3", activities), usecases = [uc_3],       dependencies = ["embedder_2"])

    pipeline.train()
    assert activities == ["data-train",
                          "data-transform", "data-transform", "data-transform",
                          "filter-transform", "filter-transform", "filter-transform",
                          "embedder-1-transform", "embedder-1-transform", "embedder-2-transform",
                          "model-1-train", "model-2-train", "model-3-train"], "Train activities wrong"

    activities.clear()
    out = pipeline.predict(init_value=133)
    assert activities == ["data-transform", "data-transform", "data-transform",
                          "filter-transform", "filter-transform", "filter-transform",
                          "embedder-1-transform", "embedder-1-transform",
                          "embedder-2-transform",
                          "model-1-predict", "model-2-predict", "model-3-predict"], "Predict activities wrong"

    assert len(out) == 3, "Wrong prediction (1)"
    assert out[0].usecase_name == uc_1.name and out[0].model == "model-1", "Wrong order of predictions (1)"
    assert out[1].usecase_name == uc_2.name and out[1].model == "model-2", "Wrong order of predictions (2)"
    assert out[2].usecase_name == uc_3.name and out[2].model == "model-3", "Wrong order of predictions (3)"

    activities.clear()
    pipeline.predict(usecases=[uc_2])
    assert activities == ["data-transform",
                          "filter-transform",
                          "embedder-1-transform",
                          "model-2-predict"], "Predict(uc_2) activities wrong"

def test_cuda_mode():
    uc_1 = TestUseCase("usecase 1")
    t1 = TestTrainable("t1", [])

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_step("t1", t1)

    assert not  t1.is_cuda, "Default cuda mode is wrong"
    t1.set_cuda_mode(on = True)
    assert t1.is_cuda, "Setting cuda mode failed (True)"
    t1.set_cuda_mode(on=False)
    assert not t1.is_cuda, "Setting cuda mode failed (False)"

    assert not  t1.is_cuda, "Default cuda mode is wrong"
    pipeline.set_inference_mode(on = False, cuda = True)
    assert pipeline.is_cuda, "Setting cuda mode failed for pipeline (True)"
    assert t1.is_cuda, "Setting cuda mode failed for step (True)"

    pipeline.set_inference_mode(on=False, cuda=False)
    assert not pipeline.is_cuda, "Setting cuda mode failed for pipeline (False)"
    assert not t1.is_cuda, "Setting cuda mode failed for step (False)"

def test_inference_mode():
    uc_1 = TestUseCase("usecase 1")
    t1 = TestTrainable("t1", [])

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_step("t1", t1)

    assert not t1.is_inference, "Default inference mode is wrong"
    t1.set_inference_mode(on=True)
    assert t1.is_inference, "Setting inference mode failed (True)"
    t1.set_inference_mode(on=False)
    assert not t1.is_inference, "Setting inference mode failed (False)"

    assert not t1.is_inference, "Default inference mode is wrong"
    pipeline.set_inference_mode(on=True, cuda=False)
    assert pipeline.is_inference, "Setting inference mode failed for pipeline (True)"
    assert t1.is_inference, "Setting inference mode failed for step (True)"

    pipeline.set_inference_mode(on=False, cuda=False)
    assert not pipeline.is_inference, "Setting inference mode failed for pipeline (False)"
    assert not t1.is_inference, "Setting inference mode failed for step (False)"

def test_mode_context_manager():
    uc_1 = TestUseCase("usecase 1")
    t1 = TestTrainable("t1", [])

    pipeline = StandardPipeline() \
        .add_usecase(uc_1) \
        .add_step("t1", t1)

    assert not t1.is_inference and not t1.is_cuda, "Default values are wrong (pre-mode)"

    with pipeline.modes(inference = True, cuda = True):
        assert t1.is_inference and t1.is_cuda, "Mode-1 values are wrong"

    assert not t1.is_inference and not t1.is_cuda, "Default values are wrong (post-mode)"

def test_import_export():
    import tempfile
    from io import BytesIO

    for test in [1, 2]:

        uc_1 = TestUseCase("usecase 1")
        t1 = TestTrainableTransformable("t1", activities:=[])

        pipeline = StandardPipeline() \
            .add_usecase(uc_1) \
            .add_step("t1", t1)

        pipeline.train()
        assert activities == ["t1-train", "t1-transform"], "Activities are Wrong"

        if test == 1:
            buffer = BytesIO()
            pipeline.export(buffer=buffer, prune=True)
            pipe: StandardPipeline = Pipeline.load(buffer = buffer)
        else:
            pipeline.export(suffix = "test", folder = tempfile.gettempdir(), prune=True)
            pipe: StandardPipeline = Pipeline.load(suffix = "test", folder = tempfile.gettempdir())
            try:
                os.unlink(os.path.normpath(f"{tempfile.gettempdir()}/pipeline_test.pkl"))
                os.unlink(os.path.normpath(f"{tempfile.gettempdir()}/pipeline_test.json"))
            except:
                pass

        assert pipe != pipeline, "Should not be the same object"

        steps = pipe.get_steps()

        assert len(steps.get(Trainable,     [])) == 1, "Metadata should only contain one trainable step"
        assert len(steps.get(Transformable, [])) == 1, "Metadata should only contain one transformable step"
        assert len(steps.get(Cleanable,     [])) == 0, "Metadata should only contain no cleanable steps"
        assert len(steps.get(Predictable,   [])) == 0, "Metadata should only contain no predictable steps"

        t1_loaded: TestTrainableTransformable | None = pipe.get_step("t1")
        assert t1_loaded is not None, "Loaded step should not be None"
        assert t1_loaded._activities == ["t1-train", "t1-transform"], "Activities for the loaded step are wrong"

        with pytest.raises(Exception):
            pipe.export(folder = None, buffer = None)