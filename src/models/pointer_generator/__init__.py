"""Pointer-Generator Network 模块."""
from .pg_decoder import PointerGeneratorDecoder
from .pg_model import PointerGeneratorSeq2Seq

__all__ = ['PointerGeneratorSeq2Seq', 'PointerGeneratorDecoder']