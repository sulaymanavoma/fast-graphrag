import json
import os
import shutil
import time
from typing import Any, Callable, List, Optional

from fast_graphrag._exceptions import InvalidStorageError
from fast_graphrag._utils import logger


class Workspace:
    INDEX_FILE = "index.json"

    @staticmethod
    def new(working_dir: str, checkpoint: Optional[str] = None, keep_n: int = 3) -> "Workspace":
        return Workspace(working_dir, checkpoint, keep_n)

    def __init__(self, working_dir: str, checkpoint: Optional[str] = None, keep_n: int = 3):
        self.working_dir: str = working_dir
        self.keep_n: int = keep_n
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
            self.checkpoints: List[str] = []

            with open(os.path.join(working_dir, self.INDEX_FILE), "w") as f:
                json.dump(self.checkpoints, f)
        else:
            with open(os.path.join(working_dir, self.INDEX_FILE), "r") as f:
                self.checkpoints: List[str] = json.load(f)

        # self.checkpoints = sorted(
        #     (int(x.name) for x in os.scandir(self.working_dir) if x.is_dir() and not x.name.startswith("0__err_")),
        #     reverse=True,
        # )
        self.current_load_checkpoint: Optional[str] = checkpoint or (
            self.checkpoints[-1] if len(self.checkpoints) else None
        )
        self.save_checkpoint: Optional[str] = None
        self.failed_checkpoints: List[str] = []

    def __del__(self):
        # TODO: delete all files that starts with a failed checkpoint
        for checkpoint in self.failed_checkpoints:
            old_path = os.path.join(self.working_dir, checkpoint)
            new_path = os.path.join(self.working_dir, f"0__err_{checkpoint}")
            os.rename(old_path, new_path)

        if self.keep_n > 0:
            checkpoints = sorted((x.name for x in os.scandir(self.working_dir) if x.is_dir()), reverse=True)
            for checkpoint in checkpoints[self.keep_n :]:
                shutil.rmtree(os.path.join(self.working_dir, str(checkpoint)))

    def make_for(self, namespace: str) -> "Namespace":
        return Namespace(self, namespace)

    def get_load_path(self, resource_name: str) -> Optional[str]:
        if self.current_load_checkpoint is None:
            return None
        return os.path.join(self.working_dir, self.current_load_checkpoint + f"_{resource_name}")

    def get_save_path(self, resource_name: str) -> str:
        if self.save_checkpoint is None:
            self.save_checkpoint = str(time.time())
        return os.path.join(self.working_dir, self.save_checkpoint + f"_{resource_name}")

    def _rollback(self) -> bool:
        if self.current_load_checkpoint is None:
            return False
        # List all directories in the working directory and select the one
        # with the smallest number greater then the current load checkpoint.
        try:
            self.current_load_checkpoint = next(x for x in self.checkpoints if x < self.current_load_checkpoint)
            logger.warning("Rolling back to checkpoint: %s", self.current_load_checkpoint)
        except (StopIteration, ValueError):
            self.current_load_checkpoint = None
            logger.warning("No checkpoints to rollback to. Last checkpoint tried: %s", self.current_load_checkpoint)

        return True

    async def with_checkpoints(self, fn: Callable[[], Any]) -> Any:
        while True:
            try:
                return await fn()
            except Exception as e:
                logger.warning("Error occurred loading checkpoint: %s", e)
                if self.current_load_checkpoint is not None:
                    self.failed_checkpoints.append(str(self.current_load_checkpoint))
                if self._rollback() is False:
                    break
        raise InvalidStorageError("No valid checkpoints to load or default storages cannot be created.")


class Namespace:
    def __init__(self, workspace: Workspace, namespace: Optional[str] = None):
        self.namespace = namespace
        self.workspace = workspace

    def get_load_path(self, resource_name: str) -> Optional[str]:
        assert self.namespace is not None, "Namespace must be set to get resource load path."
        load_path = self.workspace.get_load_path()
        if load_path is None:
            return None
        return os.path.join(load_path, f"{self.namespace}_{resource_name}")

    def get_save_path(self, resource_name: str) -> str:
        assert self.namespace is not None, "Namespace must be set to get resource save path."
        return os.path.join(self.workspace.get_save_path(), f"{self.namespace}_{resource_name}")
