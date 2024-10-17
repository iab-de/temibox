import re
import inspect
from datetime import datetime
from typing import Any, Optional, Type, Callable
from contextlib import contextmanager

from ..tracker import Tracker
from ..prediction import Prediction
from ..pipeline.pipeline import Pipeline
from ..traits import PipelineStep, Trainable, Transformable, Cleanable, Cacheable, Predictable, Trackable, Supervising, Evaluating
from ..capabilities import CudaCapable, InferenceCapable
from ..domain import UseCase, Document
from ..context import Context, ContextArg
from ..cache import Cache

from .dag import DAG

class StandardPipeline(Pipeline):
    r"""
    - Training is done for all the added usecases simultaneously (i.e. the steps get to know all the
    usecases they are going to work with).
    - Transformation and Prediction are done on a per-usecase basis. The pipeline transforms/predicts all the usecases
    for a step in a loop, one usecase at a time, before moving on to the next step.

    """

    def __init__(self, tracker: Tracker = None, allow_cuda: bool = True):
        super().__init__()

        self._usecases = []
        self._step_usecases = {}

        self._dag: DAG[PipelineStep] = DAG()

        self._train_start = 0
        self._train_end   = 0

        self._init_timestamp = int(datetime.now().timestamp())
        self._tracker = tracker or Tracker()

        self._allow_cuda = allow_cuda
        self._use_cache = False
        self._cache = Cache(max_entries=1024)

        self._mode_inference = False
        self._mode_cuda  = False
        self._identifier = "standard-pipeline"

        if not allow_cuda:
            print("cuda will not be used!")

    def _get_cname(self, name: str) -> str:
        return name.lower().strip()

    def _validate_dependencies(self, cname, cdepends_on, cusecases):

        if cname in cdepends_on:
            raise Exception(f"Self-dependencies are not allowed")

        if missing := [n for n in cdepends_on if self._dag.get(n) is None]:
            raise Exception(f"Unknown dependencies: '{', '.join(missing)}'")

        cunames = {x.name for x in cusecases}
        for dname in cdepends_on:
            dunames = {x.name for x in self.get_step_usecases(dname)}
            if cunames & dunames != cunames:
                raise Exception(f"Dependency '{dname}' does not satisfy all the usecases: {', '.join(list(cunames - dunames))}")

    def _get_effective_kwargs(self,
                             ctx: Context,
                             cname: str | None,
                             kwargs: dict[str, Any],
                             ctx_kwargs: dict[str, ContextArg],
                             ctx_dep_kwargs: dict[str, dict[str, ContextArg]]) -> dict[str, Any]:

        if cname and (depends_on := self._dag.get_dependencies(cname)):
            extra_kwargs = {k: v for d in depends_on for k, v in ctx_dep_kwargs.get(d, {}).items()}
        else:
            extra_kwargs = ctx_kwargs.copy()

        effective_kwargs = {**extra_kwargs, **kwargs, "ctx": ctx}

        return effective_kwargs

    def _step_with_context(self,
                        ctx: Context,
                        cname: str,
                        method: Callable[[Any], dict[str, Any]],
                        kwargs: dict[str, Any],
                        ctx_kwargs: dict[str, ContextArg],
                        ctx_dep_kwargs: dict[str, dict[str, ContextArg]]):

        relevant_kwargs = self._get_effective_kwargs(ctx, cname, kwargs, ctx_kwargs, ctx_dep_kwargs)
        step_out = method(**relevant_kwargs)

        if step_out is None or not isinstance(step_out, dict):
            return

        for k, v in step_out.items():
            if cname not in ctx_dep_kwargs:
                ctx_dep_kwargs[cname] = {}

            if k not in ctx_dep_kwargs[cname]:
                ctx_dep_kwargs[cname][k] = ContextArg()

            if k not in ctx_kwargs:
                ctx_kwargs[k] = ContextArg()

            ctx_dep_kwargs[cname][k].add(ctx.active_usecase.name, v)
            ctx_kwargs[k].add(ctx.active_usecase.name, v)

    @property
    def identifier(self):
        return self._identifier

    def use_identifier(self, identifier: str) -> None:
        self._identifier = identifier

    def add_usecase(self, usecase: UseCase) -> Pipeline:
        if usecase.name in {u.name for u in self._usecases}:
            raise Exception(f"Usecase '{usecase.name}' already added")

        self._usecases.append(usecase)
        return self

    def get_usecases(self) -> list[UseCase]:
        return self._usecases.copy()

    def get_step_usecases(self, name: str) -> list[UseCase]:
        cname = self._get_cname(name)
        if cname not in self._step_usecases:
            raise Exception(f"Unknown step '{name}'")

        return self._step_usecases[cname]

    def add_step(self,
                 name: str,
                 step: PipelineStep,
                 usecases: list[UseCase] = None,
                 dependencies: list[str] = None) -> Pipeline:

        cname = self._get_cname(name)

        # Validate usecases
        if not len(self._usecases):
            raise Exception("No usecases available. Use `pipeline.add_usecase` to add some")

        if usecases:
            known = {id(x) for x in self._usecases}
            missing = [x.name for x in usecases if id(x) not in known]

            if missing:
                raise Exception(f"Step references unknown usecases: {', '.join(missing)}")
        else:
            usecases = self._usecases

        self._step_usecases[cname] = usecases
        step.register_usecases(usecases)

        if self._dag.get(cname):
            raise Exception(f"Step '{cname}' already exists")

        # Validate dependencies
        if dependencies is not None:
            self._validate_dependencies(cname, dependencies, usecases)
        else:
            dependencies = list(self._dag.nodes.keys())

        # Add to the DAG and register for supervision
        self._dag.add(name, step, depends_on = dependencies)
        if isinstance(step, Supervising) and dependencies:
            for dname in dependencies:
                step.supervise(dname, self._dag.get(dname).value)

        # Add Tracker
        if isinstance(step, Trackable):
            step.use_progress_tracker(self._tracker)

        # Add identifier
        if isinstance(step, Predictable):
            step.use_identifier(name)

        return self

    def replace_step(self, name: str, step: PipelineStep):
        node: DAG | None = self._dag.get(name)
        if node is None:
            raise Exception(f"No such step '{name}'")

        node.set_value(step)

    def get_steps(self) -> dict[Type[PipelineStep], list[tuple[str, PipelineStep]]]:

        step_map = {
            Trainable:     [],
            Transformable: [],
            Cleanable:     [],
            Predictable:   [],
        }

        for name, node in self._dag.nodes.items():
            step = node.value
            for step_type, collection in step_map.items():
                if isinstance(step, step_type):
                    collection.append((name, step))

        return step_map

    def get_step(self, name: str) -> Optional[PipelineStep]:
        if node := self._dag.get(name):
            return node.value

        return None

    def get_step_dependencies(self, name: str) -> list[PipelineStep]:
        if node := self._dag.get(name):
            return [c.value for c in node.parents if c.value]

        return []

    def set_inference_mode(self, on: bool, cuda: bool, **kwargs):
        self._mode_inference = on
        self._mode_cuda      = cuda

        for node in self._dag.nodes.values():

            step = node.value
            if isinstance(step, InferenceCapable):
                step.set_inference_mode(on=on, **kwargs)

            if hasattr(self, "_allow_cuda") and not self._allow_cuda:
                continue

            if isinstance(step, CudaCapable):
                step.set_cuda_mode(on=cuda)

    @property
    def is_cuda(self) -> bool:
        return self._mode_cuda

    @property
    def is_inference(self) -> bool:
        return self._mode_inference

    @contextmanager
    def modes(self, inference: bool, cuda: bool):
        o_inference = self._mode_inference
        o_cuda      = self._mode_cuda

        try:
            self.set_inference_mode(on=inference, cuda=cuda)
            yield self
        finally:
            self.set_inference_mode(on=o_inference, cuda=o_cuda)

    @property
    def init_timestamp(self) -> int:
        return  self._init_timestamp

    @property
    def train_duration_sec(self) -> int:
        return self._train_end - self._train_start

    @property
    def is_training(self) -> bool:
        return self._train_end is None or self._train_end == 0

    def train(self,
              start_at: str = None,
              stop_at:  str = None,
              ctx_kwargs:     dict[str, ContextArg] = None,
              ctx_dep_kwargs: dict[str, dict[str, ContextArg]] = None,
              **kwargs) -> None:

        r"""
        Trains the assembled pipeline

        :param start_at:
        :param stop_at:
        :param ctx_kwargs: all instances of ContextArg created by pipeline transformations (possible overwrites if steps output has the same name)
        :param ctx_dep_kwargs: step-specific instances of ContextArg (used to pass outputs from dependencies; not overwrites, since namespaced per step)
        :param kwargs:
        :return:
        """

        # Check start/stop conditions
        if start_at is not None and ((node := self._dag.get(start_at)) is None or not isinstance(node.value, Trainable)):
            raise Exception(f"Unknown or incompatible start step '{start_at}'")

        if stop_at is not None and ((node := self._dag.get(stop_at)) is None or not isinstance(node.value, Trainable)):
            raise Exception(f"Unknown or incompatible stop step '{stop_at}'")

        if ctx_kwargs is None:
            ctx_kwargs = {}

        if ctx_dep_kwargs is None:
            ctx_dep_kwargs = {}

        self._train_start = int(datetime.now().timestamp()) if not self._train_start else self._train_start
        self._train_end   = None
        started = False
        for name, step in self._dag.walk(predicate = lambda x: isinstance(x, Trainable) or isinstance(x, Transformable)):
            cname = self._get_cname(name)

            if start_at is not None and not started and cname != self._get_cname(start_at):
                continue

            if stop_at is not None and cname == self._get_cname(stop_at):
                print(f"Stopping at '{stop_at}' as requested")
                return

            started = True

            step_usecases = self.get_step_usecases(name).copy()
            ctx = Context(pipeline = self, usecases = step_usecases, active_usecase = None, active_step_name = name)
            relevant_kwargs = self._get_effective_kwargs(ctx, cname, kwargs, ctx_kwargs, ctx_dep_kwargs)

            if isinstance(step, Trainable):
                step.train(**relevant_kwargs)

            if isinstance(step, Transformable):
                for usecase in step_usecases:
                    ctx = Context(pipeline=self, usecases=step_usecases, active_usecase=usecase, active_step_name = name)
                    self._step_with_context(ctx, cname, step.transform, kwargs, ctx_kwargs, ctx_dep_kwargs)

        self._train_end = int(datetime.now().timestamp())

    def _get_valid_usecases(self, usecases: list[UseCase]) -> list[UseCase]:

        if usecases is not None and len(usecases):
            available = {u.name for u in self._usecases}
            usecases = [u for u in usecases if u.name in available]
        else:
            usecases = self._usecases.copy()

        return usecases

    def transform(self,
                  usecases:       list[UseCase] = None,
                  ctx_kwargs:     dict[str, ContextArg] = None,
                  ctx_dep_kwargs: dict[str, dict[str, ContextArg]] = None,
                  **kwargs) -> dict[str, Any]:

        usecases = self._get_valid_usecases(usecases)

        if not len(usecases):
            raise Exception("No valid usecases provided")

        if ctx_kwargs is None:
            ctx_kwargs = {}

        if ctx_dep_kwargs is None:
            ctx_dep_kwargs = {}

        for name, step in self._dag.walk(predicate=lambda x: isinstance(x, Transformable)):
            cname = self._get_cname(name)

            self._tracker.progress(0, 0, f"[{name}] Transforming")
            for usecase in usecases:
                step_usecases = self._step_usecases.get(cname, [])
                if usecase.name not in {u.name for u in step_usecases}:
                    continue

                try:
                    ctx = Context(pipeline = self, usecases = step_usecases, active_usecase = usecase, active_step_name = name)
                    self._step_with_context(ctx, cname, step.transform, kwargs, ctx_kwargs, ctx_dep_kwargs)
                except Exception as e:
                    print(f"Failed transforming step '{name}'")
                    raise e

        ctx = Context(pipeline = self, usecases = self._usecases)
        return self._get_effective_kwargs(ctx, None, kwargs, ctx_kwargs, ctx_dep_kwargs)

    def predict(self,
                usecases: list[UseCase] = None,
                **kwargs) -> list[Prediction]:

        usecases = self._get_valid_usecases(usecases)

        if not len(usecases):
            raise Exception("No valid usecases provided")

        ctx_kwargs = {}
        ctx_dep_kwargs = {}
        collection: list[Prediction] = []

        with self.modes(inference=True, cuda=True):

            self._tracker.progress(0, 0, "Starting transformations")

            self.transform(**{**kwargs,
                              **{"ctx_kwargs": ctx_kwargs,
                                 "ctx_dep_kwargs": ctx_dep_kwargs,
                                 "usecases": usecases}
                              })

            for usecase in usecases:

                self._tracker.progress(0, 0, f"[{usecase.name}] Starting predictions")

                for name, step in self._dag.walk(predicate=lambda x: isinstance(x, Predictable)):
                    cname = self._get_cname(name)

                    step_usecases = self._step_usecases.get(cname, [])
                    if usecase.name not in {u.name for u in step_usecases}:
                        continue

                    ctx = Context(pipeline=self, usecases=step_usecases, active_usecase=usecase, active_step_name=name)
                    relevant_kwargs = self._get_effective_kwargs(ctx, cname, kwargs, ctx_kwargs, ctx_dep_kwargs)
                    collection += step.predict(**relevant_kwargs)

                self._tracker.progress(0, 0, f"Finished usecase '{usecase.name}'")

        return collection

    def clean(self) -> None:

        for name, step in self._dag.walk(predicate=lambda x: isinstance(x, Cleanable)):
            try:
                step.clean()
            except Exception as e:
                print(f"Could not prune {name}: {str(e)}")

    def configure_cache(self, on: bool, max_entries: int = 1024):
        self._cache.configure_cache(on = on, max_entries = max_entries)

        for name, step in self._dag.walk(predicate=lambda x: isinstance(x, Cacheable)):
            step.configure_cache(on = on, max_entries = max_entries)

        self._use_cache = on

    def clear_cache(self):
        self._cache.clear_cache()

        for name, step in self._dag.walk(predicate=lambda x: isinstance(x, Cacheable)):
            step.clear_cache()

    @property
    def cache(self) -> Cache:
        return self._cache

    @property
    def is_caching(self) -> bool:
        return self._use_cache

    def use_progress_tracker(self, tracker: Tracker) -> None:

        self._tracker = tracker or Tracker()

        for step in self._dag.values:
            if isinstance(step, Trackable):
                step.use_progress_tracker(self._tracker)

    def get_progress_tracker(self) -> Tracker:
        return self._tracker

    def evaluate(self,
                 usecases: list[UseCase] = None,
                 **kwargs):

        usecases = self._get_valid_usecases(usecases)

        if not len(usecases):
            raise Exception("No valid usecases provided")

        ctx_kwargs = {}
        ctx_dep_kwargs = {}

        with self.modes(inference=True, cuda=True):

            for usecase in usecases:

                for name, step in self._dag.walk(predicate=lambda x: isinstance(x, Evaluating)):
                    cname = self._get_cname(name)

                    step_usecases = self._step_usecases.get(cname, [])
                    if usecase.name not in {u.name for u in step_usecases}:
                        continue

                    try:
                        ctx = Context(pipeline=self, usecases=step_usecases, active_usecase=usecase, active_step_name = name)
                        self._step_with_context(ctx, cname, step.evaluate, kwargs, ctx_kwargs, ctx_dep_kwargs)
                    except Exception as e:
                        print(f"Failed transforming step '{name}'")
                        raise e

        return ctx_dep_kwargs

    def _fill_method_params(self,
                           method: Callable[[Any], Any],
                           signatures: dict[str, set[str]],
                           step_name: str | None) -> set[str]:

        key = method.__name__
        if key not in signatures:
            signatures[key] = set() if step_name is None else dict()

        if step_name:
            signatures[key][step_name] = set()

        substitutions = [("temibox.[a-z_\.]+", ""),
                         ("Type\[ForwardRef\('Document'\)\]", "Document")]
        params = set()
        for param, value in inspect.signature(method).parameters.items():
            if param in ["self", "ctx", "kwargs"]:
                continue

            val = str(value)
            for pat, sub in substitutions:
                val = re.sub(pat, sub, val)

            params.add(val)

        if len(params):
            if step_name:
                signatures[key][step_name] |= params
            else:
                signatures[key] |= params

        return params

    def get_signature(self,
                      step_type: Type[PipelineStep],
                      per_step: bool = False,
                      as_string: bool = False, **kwargs) -> dict[str, set[str]] | str:
        r"""
        Liefert die über alle Pipeline-Schritte definierte Signature der Pipeline-Methoden
        "train", "transform", "predict" und "evaluate"

        Diese Signatur listet die Gesamtmenge der Parameter *aller* im Berechnungspfad involvierter Komponenten.

        :param step_type: Pipeline-Fähigkeit aus temibox.traits (mögliche Ausprägungen: Trainable, Transformable, Predictable, Evaluating)
        :param per_step:  Ausgabe per Komponente
        :param as_string: Ausgabe als formatierte Zeichenkette

        :return: dict oder str mit setzbaren Parametern
        """
        if step_type not in [Trainable, Transformable, Predictable, Evaluating]:
            raise Exception("Unsupported step type")

        relevant_methods = {Trainable:     ["train"],
                            Transformable: ["transform"],
                            Predictable:   ["predict"],
                            Evaluating:    ["evaluate"]}

        relevant_methods_all = [xi for x in relevant_methods.values() for xi in x]

        signatures = {}
        relevant_traits = {Transformable} | {step_type} if step_type != Trainable else {step_type}
        name_order = {}
        for name, step in self._dag.walk(predicate=lambda x: any([isinstance(x,y) for y in relevant_traits])):

            if name not in name_order:
                name_order[name] = len(name_order)

            for trait in {x for x in relevant_traits if isinstance(step, x)}:
                trait_methods = {x[0] for x in inspect.getmembers(trait, predicate = inspect.isfunction) if x[0] in relevant_methods_all}

                for method_name, method in inspect.getmembers(step, predicate = inspect.ismethod):
                    if method_name[0] == "_" or method_name not in trait_methods:
                        continue

                    self._fill_method_params(method, signatures, name if per_step else None)

        if len(relevant_traits) > 1:
            for k in signatures.keys():
                if per_step:
                    for t in signatures["transform"].keys():
                        if t not in signatures[k]:
                            signatures[k][t] = signatures["transform"][t]
                else:
                    signatures[k] |= signatures["transform"]

        signatures = {k:v for k,v in signatures.items() if k in relevant_methods[step_type]}

        if as_string:
            sig_str = ""
            for method_name, params in signatures.items():
                sig_str += f"{method_name.upper()}:\n"

                params = {"": params} if not per_step else params
                if per_step:
                    params = {x: params[x] for x in sorted(params, key = lambda key: name_order.get(key, 0))}

                for step_name, step_params in params.items():

                    if step_name != "":
                        sig_str += f"\n> {step_name}\n"
                        tab  = "\t\t"
                    else:
                        tab = "\t"

                    for p in sorted(step_params):
                        p_name, p_type = [x.strip() for x in p.split(":")]
                        sig_str += f"{tab}{p_name}" + " "*(20 - len(p_name)) + ": " + p_type + "\n"

                    sig_str += "\n"

            sig_str += "\n"

            return sig_str.strip()

        return signatures

    def show_workflow(self, workflow: str):

        if (wf := workflow.strip().lower()) not in ["train", "predict", "evaluate"]:
            raise Exception("Unknown workflow. Supported workflows: 'train', 'predict', 'evaluate'")

        if len(self._dag.nodes) == 0:
            raise Exception("Pipeline is empty")

        predicate_trainable     = lambda x: isinstance(x, Trainable)
        predicate_transformable = lambda x: isinstance(x, Transformable)
        predicate_predictable   = lambda x: isinstance(x, Predictable)
        predicate_evaluating    = lambda x: isinstance(x, Evaluating)

        if wf == "train":
            return self._dag.display("Workflow: 'train'", [predicate_trainable])

        elif wf == "predict":
            self._dag.display("Workflow: 'predict'", [predicate_transformable, predicate_predictable])

        elif wf == "evaluate":
            self._dag.display("Workflow: 'evaluate'", [predicate_predictable, predicate_evaluating])


