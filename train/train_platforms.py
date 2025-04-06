import os
import glob

class TrainPlatform:
    def __init__(self, save_dir, *args, **kwargs):
        self.path, file = os.path.split(save_dir)
        self.name = kwargs.get('name', file)

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_media(self, title, series, iteration, local_path):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='mdm-sandbox',
                              task_name=name)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_media(self, title, series, iteration, local_path):
        self.logger.report_media(title=title, series=series, iteration=iteration, local_path=local_path)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name=None):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass

class WandBPlatform(TrainPlatform):
    import wandb
    def __init__(self, save_dir, config=None, *args, **kwargs):
        super().__init__(save_dir, args, kwargs)
        self.wandb.init(
            project='mdm-sandbox',
            name=self.name,
            save_code=True,
            config=config)  # config can also be sent via report_args()

    def report_scalar(self, name, value, iteration, group_name=None):
        self.wandb.log({name: value}, step=iteration)

    def report_media(self, title, series, iteration, local_path):
        files = glob.glob(f'{local_path}/*.mp4')
        self.wandb.log({series: [self.wandb.Video(file, format='mp4', fps=20) for file in files]}, step=iteration)

    def report_args(self, args, name):
        self.wandb.config.update(args)

    def watch_model(self, *args, **kwargs):
        self.wandb.watch(args, kwargs)

    def close(self):
        self.wandb.finish()


