from ..registry import Component


class Enhancer(Component):

    async def enhance(self, audio: bytes) -> bytes:
        raise NotImplementedError
