import os
import re
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Any, Optional

from .traits import Predictable
from .capabilities import CudaCapable, InferenceCapable


@dataclass
class StepMetadata:
    r"""
    Step metadata used in pipeline export

    Contains information on the classname, cuda and inference modes
    as well as a list of class hierarchies (mro)
    """
    classname: str
    cuda:      bool
    inference: bool
    mro:       list[str]


@dataclass
class RepositoryMetadata:
    r"""
    git-repository metadata used in pipeline export

    Contains information on the path, origin, active branch and commit
    as well as dirty-flag.
    """
    path:    str
    origin:  str
    branch:  str
    commit:  str
    message: str
    dirty:   bool


@dataclass
class MetaData:
    f"""
    Pipeline metadata used in pipeline export
    
    Contains metadata for each step, as well as the git-repository,
    export data, pipeline identifier and training duration
    """
    steps:              dict[str, StepMetadata]
    repository:         Optional[RepositoryMetadata]
    export_date:        str
    identifier:         str
    train_duration_sec: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineMetadata:
    r"""
    Collection of static methods for pipeline metadata generation
    """

    @staticmethod
    def _run_git_cmd(cmd) -> str:
        return subprocess.check_output(cmd.split()).strip().decode()

    @staticmethod
    def _get_git_metadata() -> Optional[RepositoryMetadata]:

        pm = PipelineMetadata

        try:
            return RepositoryMetadata(
                        path    = os.getcwd(),
                        origin  = pm._run_git_cmd("git config --get remote.origin.url"),
                        branch  = pm._run_git_cmd("git rev-parse --abbrev-ref HEAD"),
                        commit  = pm._run_git_cmd("git describe --always"),
                        message = pm._run_git_cmd("git log -1 --pretty=%B"),
                        dirty   = len(pm._run_git_cmd("git diff --stat")) > 0)
        except:
            return None

    @staticmethod
    def _clean_classname(dirtyname) -> str:
        return re.sub(r"<class '(.*)'>", r"\1", str(dirtyname))

    @staticmethod
    def _get_identifier(pipeline: 'Pipeline') -> str:
        predictors = pipeline.get_steps().get(Predictable, [])

        pred = "???"
        if len(predictors):
            pred = " / ".join([f"{p.__module__}.{p.__class__.__name__}" for _, p in predictors])

        return f"{pred} [{pipeline.init_timestamp}]"

    @staticmethod
    def get_metadata(pipeline: 'Pipeline') -> MetaData:
        r"""
        Returns pipeline metadata for a given pipeline

        :param pipeline: Pipeline

        :return: pipeline metadata
        """
        pm = PipelineMetadata

        step_meta = {}
        for typename, steplist in pipeline.get_steps().items():
            tname = typename.__name__

            for sname, step in steplist:
                cuda      = step.is_cuda if isinstance(step, CudaCapable) else False
                inference = step.is_inference if isinstance(step, InferenceCapable) else False
                mro       = [pm._clean_classname(str(c)) for c in list(type(step).__mro__)]

                if tname not in step_meta:
                    step_meta[tname] = {}

                step_meta[tname][sname] = StepMetadata(classname = pm._clean_classname(type(step)),
                                                       cuda      = cuda,
                                                       inference = inference,
                                                       mro       = mro)

        meta = MetaData(steps = step_meta,
                        repository         = pm._get_git_metadata(),
                        export_date        = str(datetime.now()),
                        identifier         = pm._get_identifier(pipeline),
                        train_duration_sec = pipeline.train_duration_sec)


        return meta