from ..registry import Component


class Labeler(Component):

    def label(self, records: list) -> list:
        raise NotImplementedError
