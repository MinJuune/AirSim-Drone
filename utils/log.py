from config.config import LOG_LEVEL


def log(level, message):
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    if levels.index(level) >= levels.index(LOG_LEVEL):
        print(f"[{level}] {message}")