from ..registry import Component


class Filter(Component):

    def filter(self, records: list) -> list:
        raise NotImplementedError
