# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
from time import time

from ultralytics.hub.utils import PREFIX, traces
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params


def on_pretrain_routine_end(trainer):
    """Logs info before starting timer for upload rate limit."""
    session = getattr(trainer, 'hub_session', None)
    if session:
        # Start timer for upload rate limit
        LOGGER.info(f'{PREFIX}View model at https://hub.ultralytics.com/models/{session.model_id} 🚀')
        session.timers = {'metrics': time(), 'ckpt': time()}  # start timer on session.rate_limit


def on_fit_epoch_end(trainer):
    """Uploads training progress metrics at the end of each epoch."""
    session = getattr(trainer, 'hub_session', None)
    if session:
        # Upload metrics after val end
        all_plots = {**trainer.label_loss_items(trainer.tloss, prefix='train'), **trainer.metrics}
        if trainer.epoch == 0:
            model_info = {
                'model/parameters': get_num_params(trainer.model),
                'model/GFLOPs': round(get_flops(trainer.model), 3),
                'model/speed(ms)': round(trainer.validator.speed['inference'], 3)}
            all_plots = {**all_plots, **model_info}
        session.metrics_queue[trainer.epoch] = json.dumps(all_plots)
        if time() - session.timers['metrics'] > session.rate_limits['metrics']:
            session.upload_metrics()
            session.timers['metrics'] = time()  # reset timer
            session.metrics_queue = {}  # reset queue


def on_model_save(trainer):
    """Saves checkpoints to Ultralytics HUB with rate limiting."""
    session = getattr(trainer, 'hub_session', None)
    if session:
        # Upload checkpoints with rate limiting
        is_best = trainer.best_fitness == trainer.fitness
        if time() - session.timers['ckpt'] > session.rate_limits['ckpt']:
            LOGGER.info(f'{PREFIX}Uploading checkpoint https://hub.ultralytics.com/models/{session.model_id}')
            session.upload_model(trainer.epoch, trainer.last, is_best)
            session.timers['ckpt'] = time()  # reset timer


def on_train_end(trainer):
    """Upload final model and metrics to Ultralytics HUB at the end of training."""
    session = getattr(trainer, 'hub_session', None)
    if session:
        # Upload final model and metrics with exponential standoff
        LOGGER.info(f'{PREFIX}Syncing final model...')
        session.upload_model(trainer.epoch, trainer.best, map=trainer.metrics.get('metrics/mAP50-95(B)', 0), final=True)
        session.alive = False  # stop heartbeats
        LOGGER.info(f'{PREFIX}Done ✅\n'
                    f'{PREFIX}View model at https://hub.ultralytics.com/models/{session.model_id} 🚀')


def on_train_start(trainer):
    """Run traces on train start."""
    traces(trainer.args, traces_sample_rate=1.0)


def on_val_start(validator):
    """Runs traces on validation start."""
    traces(validator.args, traces_sample_rate=1.0)


def on_predict_start(predictor):
    """Run traces on predict start."""
    traces(predictor.args, traces_sample_rate=1.0)


def on_export_start(exporter):
    """Run traces on export start."""
    traces(exporter.args, traces_sample_rate=1.0)


callbacks = {
    'on_pretrain_routine_end': on_pretrain_routine_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_model_save': on_model_save,
    'on_train_end': on_train_end,
    'on_train_start': on_train_start,
    'on_val_start': on_val_start,
    'on_predict_start': on_predict_start,
    'on_export_start': on_export_start}
