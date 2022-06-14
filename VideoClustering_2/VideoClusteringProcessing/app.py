from utils.utils import setup_logging
setup_logging()


if __name__ == "__main__":
    from pipeline.pipeline import PipeLine
    from config.config import Config
    import logging
    config = Config()
    logger = logging.getLogger(config.main_logger_name)
    pipeline = PipeLine()
    logger.info("*" * 50 + f'\nStart Program with config: {config}\n' + "*" * 50)
    pipeline.start()
    pipeline.join()
