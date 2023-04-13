import warnings

from packaging import version
from typing import Callable
import optuna
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage
from optuna.integration import TorchDistributedTrial

# Define key names of `Trial.system_attrs`.
_EPOCH_KEY = "ddp_pl:epoch"
_INTERMEDIATE_VALUE = "ddp_pl:intermediate_value"
_PRUNED_KEY = "ddp_pl:pruned"

__version__ = "0.0.1"

with optuna._imports.try_import() as _imports:
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # type: ignore[assignment, misc]  # NOQA[F811]
    LightningModule = object  # type: ignore[assignment, misc]  # NOQA[F811]
    Trainer = object  # type: ignore[assignment, misc]  # NOQA[F811]


class DDPPyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    .. note::
        For the distributed data parallel training, the version of PyTorchLightning needs to be
        higher than or equal to v1.6.0. In addition, :class:`~optuna.study.Study` should be
        instantiated with RDB storage.
    .. note::
        If you would like to use PyTorchLightningPruningCallback in a distributed training
        environment, you need to evoke `PyTorchLightningPruningCallback.check_pruned()`
        manually so that :class:`~optuna.exceptions.TrialPruned` is properly handled.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        _imports.check()
        super().__init__()
        self._trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        self._trial = self._trial if trainer.is_global_zero else None
        self._trial = TorchDistributedTrial(self._trial)

    def on_fit_start(self, trainer: Trainer, pl_module: "pl.LightningModule") -> None:
        self.is_ddp_backend = trainer._accelerator_connector.is_distributed
        if self.is_ddp_backend:
            self.check_exception(self.validate_setting, trainer, pl_module)
            # It is necessary to store intermediate values directly in the backend storage because
            # they are not properly propagated to main process due to cached storage.
            # TODO(Shinichi) Remove intermediate_values from system_attr after PR #4431 is merged.
            self._trial.set_system_attr(_INTERMEDIATE_VALUE, dict())
    
    def check_exception(self, func: Callable, trainer: Trainer, pl_module: "pl.LightningModule"):
        if not self.is_ddp_backend:
            return
        err = None
        if trainer.is_global_zero:
            try:
                func(trainer, pl_module)
            except Exception as e:
                err = e
        err = self._trial._broadcast(err)
        if err != None:
            raise err
    
    def validate_setting(self, trainer: Trainer, pl_module: "pl.LightningModule"):
        if not self.is_ddp_backend:
            return None
        if version.parse(pl.__version__) < version.parse(  # type: ignore[attr-defined]
                "1.6.0"
            ):
                raise ValueError("PyTorch Lightning>=1.6.0 is required in DDP.")
        if trainer.is_global_zero:
            # If it were not for this block, fitting is started even if unsupported storage
            # is used. Note that the ValueError is transformed into ProcessRaisedException inside
            # torch.
            if not (
                isinstance(self._trial._delegate.study._storage, _CachedStorage)
                and isinstance(self._trial._delegate.study._storage._backend, RDBStorage)
            ):
                raise ValueError(
                    "optuna.integration.PyTorchLightningPruningCallback"
                    " supports only optuna.storages.RDBStorage in DDP."
                )
        return None
    
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Trainer calls `on_validation_end` for sanity check. Therefore, it is necessary to avoid
        # calling `trial.report` multiple times at epoch 0. For more details, see
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                f"The metric '{self.monitor}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name."
            )
            warnings.warn(message)
            return

        epoch = pl_module.current_epoch

        # Determine if the trial should be terminated in a single process.
        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=epoch)
            if not self._trial.should_prune():
                return
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")

        # Determine if the trial should be terminated in a DDP.
        self._trial.report(current_score.item(), step=epoch)
        should_stop = self._trial.should_prune()

        intermediate_values=None
        if trainer.is_global_zero:
            delegate_trial = self._trial._delegate
            # Update intermediate value in the storage.
            _trial_id = delegate_trial._trial_id
            _study = delegate_trial.study
            _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
            intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)
            intermediate_values[epoch] = current_score.item()  # type: ignore[index]
        intermediate_values = self._trial._broadcast(intermediate_values)
        self._trial.set_system_attr(_INTERMEDIATE_VALUE, intermediate_values)

        # Terminate every process if any world process decides to stop.
        trainer.should_stop = trainer.should_stop or should_stop
        if not should_stop:
            return
        self._trial.set_system_attr(_PRUNED_KEY, True)
        self._trial.set_system_attr(_EPOCH_KEY, epoch)

    def check_pruned(self) -> None:
        """Raise :class:`optuna.TrialPruned` manually if pruned.
        Currently, ``intermediate_values`` are not properly propagated between processes due to
        storage cache. Therefore, necessary information is kept in trial_system_attrs when the
        trial runs in a distributed situation. Please call this method right after calling
        ``pytorch_lightning.Trainer.fit()``.
        If a callback doesn't have any backend storage for DDP, this method does nothing.
        """

        # Redirect to previous implementation if self._trial isn't TorchDistributedTrial
        if not isinstance(self._trial, TorchDistributedTrial):
            self._check_pruned_not_ddp()
            return
        
        is_global_zero = self._trial._delegate != None
        err = None
        is_pruned = False
        intermediate_values = None

        if is_global_zero:
            try:
                _trial_system_attrs, is_pruned, intermediate_values = self._main_node_final_report()
            except Exception as e:
                err = e
        
        err = self._trial._broadcast(err)
        if err != None:
            raise err
        
        intermediate_values = self._trial._broadcast(intermediate_values)
        for epoch, score in intermediate_values.items():
            self._trial.report(score, step=int(epoch))
        
        is_pruned = self._trial._broadcast(is_pruned)
        if is_pruned:
            epoch = _trial_system_attrs.get(_EPOCH_KEY) if is_global_zero else None
            epoch = self._trial._broadcast(epoch)
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")

    def _main_node_final_report(self):
        assert self._trial._delegate != None, "Final report called in non-main node. " \
                                              "Something went horribly wrong!"
        trial = self._trial._delegate
        assert isinstance(trial.study._storage, _CachedStorage), "GLOBAL RANK 0: " \
                    "study storage is not Cached Storage! Something went horribly wrong!"
        _trial_id = trial._trial_id
        _study = trial.study

        _trial_system_attrs = _study._storage._backend.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(_PRUNED_KEY)
        intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)

        # Confirm if DDP backend is used in case this method is called from a non-DDP situation by
        # mistake.
        assert intermediate_values != None, "GLOBAL RANK 0: intermediate values are None! " \
                                                            "Something went horribly wrong!"
        
        return _trial_system_attrs, is_pruned, intermediate_values
    
    def _check_pruned_not_ddp(self):
        """Raise :class:`optuna.TrialPruned` manually if pruned.
        Currently, ``intermediate_values`` are not properly propagated between processes due to
        storage cache. Therefore, necessary information is kept in trial_system_attrs when the
        trial runs in a distributed situation. Please call this method right after calling
        ``pytorch_lightning.Trainer.fit()``.
        If a callback doesn't have any backend storage for DDP, this method does nothing.
        """
        _trial_id = self._trial._trial_id
        _study = self._trial.study
        # Confirm if storage is not InMemory in case this method is called in a non-distributed
        # situation by mistake.
        if not isinstance(_study._storage, _CachedStorage):
            return

        _trial_system_attrs = _study._storage._backend.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(_PRUNED_KEY)
        intermediate_values = _trial_system_attrs.get(_INTERMEDIATE_VALUE)

        # Confirm if DDP backend is used in case this method is called from a non-DDP situation by
        # mistake.
        if intermediate_values is None:
            return
        for epoch, score in intermediate_values.items():
            self._trial.report(score, step=int(epoch))
        if is_pruned:
            epoch = _trial_system_attrs.get(_EPOCH_KEY)
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")