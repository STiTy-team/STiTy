class STiTyError(Exception):
    pass


class ConfigError(STiTyError):
    pass


class DataError(STiTyError):
    pass


class AudioError(DataError):
    pass
