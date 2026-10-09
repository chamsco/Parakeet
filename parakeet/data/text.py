"""Text normalisation, the shared tag vocabulary, and the character tokeniser.

Two design points matter for the teacher mixture:

* **Tags.** Orpheus conditions on inline paralinguistic tags (``<laugh>``, ``<sigh>`` ...) and
  MiniMax exposes emotion controls.  Parakeet maps both onto one *shared* tag vocabulary, so a
  single student learns both teachers' controllability.
* **Character-level text.** Following SupertonicTTS we train on characters and let
  cross-attention learn alignment, which removes the G2P dependency (and therefore a whole
  class of language-specific bugs).  ``phoneme`` mode exists for the single-voice Tiny model,
  where a deterministic G2P is an acceptable extra dependency.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<s>", "</s>"

#: characters we keep (lower-case ASCII + common punctuation + a little IPA for phoneme mode)
_BASE_CHARS = list(" abcdefghijklmnopqrstuvwxyz0123456789.,!?;:'\"-()[]")
_IPA_CHARS = list("əɪʊɛæɑɔʌɜɹðθʃʒŋʤʧɚɝˈˌ")

#: shared paralinguistic / emotion tag vocabulary (Orpheus tags + MiniMax emotion controls)
TAGS: List[str] = [
    "<laugh>", "<chuckle>", "<giggle>", "<sigh>", "<cough>", "<sniffle>", "<groan>",
    "<yawn>", "<gasp>", "<hum>", "<breath>", "<whisper>", "<scream>", "<cry>",
    "<angry>", "<sad>", "<happy>", "<excited>", "<surprised>", "<fearful>", "<disgusted>",
    "<neutral>", "<calm>", "<serious>",
]

_ABBREV = {
    "mr.": "mister", "mrs.": "missus", "dr.": "doctor", "st.": "saint", "no.": "number",
    "&": " and ", "%": " percent ", "+": " plus ", "=": " equals ",
}

_NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
    "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen",
    "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty",
    "30": "thirty", "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
    "80": "eighty", "90": "ninety", "100": "hundred", "1000": "thousand",
}


def normalize_text(text: str, keep_tags: bool = True) -> str:
    """Light, deterministic normalisation.  Deliberately conservative: heavier normalisation
    (dates, currencies, ordinals) belongs in the data pipeline where it can be validated.

    Tags are handled by *segmentation* rather than by placeholder substitution: each non-tag
    segment is normalised independently and the tags are re-inserted in place.  (Placeholders
    silently break here, because the number-to-words pass rewrites the digits inside them.)
    """
    text = unicodedata.normalize("NFKC", text)
    if keep_tags:
        parts = re.split(r"(<[a-zA-Z_ ]+>)", text)
        out: List[str] = []
        for part in parts:
            if re.fullmatch(r"<[a-zA-Z_ ]+>", part or ""):
                out.append(part.lower())
            else:
                out.append(normalize_text(part, keep_tags=False))
        return re.sub(r"\s+", " ", " ".join(p for p in out if p)).strip()

    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")
    for k, v in _ABBREV.items():
        text = text.replace(k, v)
    text = text.lower()
    text = re.sub(r"\b(\d+)\b", lambda m: " ".join(_NUM_WORDS.get(c, c) for c in m.group(1)), text)
    text = re.sub(r"[^a-z0-9'\-.,!?;:()\[\] ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class CharVocab:
    mode: str = "char"
    tags: Sequence[str] = field(default_factory=lambda: TAGS)
    chars: Optional[List[str]] = None

    def __post_init__(self) -> None:
        base = list(_BASE_CHARS) + (list(_IPA_CHARS) if self.mode == "phoneme" else [])
        symbols = [PAD, UNK, BOS, EOS] + list(self.tags) + base
        self.itos: List[str] = symbols
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(symbols)}
        self.tag_ids: Dict[str, int] = {t: self.stoi[t] for t in self.tags if t in self.stoi}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids = [self.stoi[BOS]] if add_special else []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            elif ch.isspace():
                continue
            else:
                ids.append(self.stoi[UNK])
        if add_special:
            ids.append(self.stoi[EOS])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        specials = {self.stoi[PAD], self.stoi[BOS], self.stoi[EOS]}
        return "".join(self.itos[i] for i in ids if i not in specials and i < len(self.itos))

    def split_tags(self, text: str) -> tuple[str, List[str]]:
        """Separate inline tags from the spoken text (tags drive style, not pronunciation)."""
        found = re.findall(r"<[a-zA-Z_ ]+>", text)
        found = [t.lower() for t in found if t.lower() in self.tag_ids]
        clean = re.sub(r"<[a-zA-Z_ ]+>", " ", text)
        return normalize_text(clean, keep_tags=False), found


class TextTokenizer:
    """Tokeniser used by training and inference."""

    def __init__(self, mode: str = "char", tags: Optional[Sequence[str]] = None) -> None:
        self.vocab = CharVocab(mode=mode, tags=list(tags) if tags else TAGS)

    @property
    def vocab_size(self) -> int:
        return self.vocab.vocab_size

    def encode(self, text: str, max_len: int = 512, add_special: bool = True) -> torch.Tensor:
        ids = self.vocab.encode(text, add_special=add_special)[:max_len]
        return torch.tensor(ids, dtype=torch.long)

    def batch(
        self, texts: Sequence[str], max_len: int = 512, add_special: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch; ``max_len`` caps each sequence.  The returned width is the longest
        encoded sequence in the batch (not ``max_len``)."""
        seqs = [self.encode(t, max_len, add_special) for t in texts]
        lengths = torch.tensor([s.numel() for s in seqs], dtype=torch.long)
        width = int(lengths.max()) if seqs else 0
        out = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : s.numel()] = s
        mask = torch.arange(width)[None, :] < lengths[:, None]
        return out, mask

    def style_tags(self, text: str) -> List[str]:
        _, tags = self.vocab.split_tags(text)
        return tags
