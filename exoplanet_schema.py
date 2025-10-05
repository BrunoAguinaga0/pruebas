import json
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

NORMALIZE_PATTERN = re.compile(r'[^a-z0-9]+')

def normalize(name: str) -> str:
    return NORMALIZE_PATTERN.sub('_', name.strip().lower()).strip('_')

def load_schema(path: str = "schema_mapping.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_reverse_index(mapping: dict) -> Dict[str, str]:
    reverse = {}
    for canon, syns in mapping["synonyms"].items():
        for cand in [canon] + syns:
            reverse[normalize(cand)] = canon
    return reverse

def auto_map_columns(columns: List[str], mapping: dict, fuzzy: bool = True, fuzzy_threshold: int = 82) -> Tuple[Dict[str, str], List[str], Dict[str, str]]:
    """
    Devuelve:
      exact_map: dict columna_original -> canon
      unmapped: lista de columnas sin mapping
      fuzzy_suggestions: dict columna_original -> canon_sugerido (solo si fuzzy)
    """
    reverse = build_reverse_index(mapping)
    exact_map = {}
    unmapped = []

    for col in columns:
        key = normalize(col)
        if key in reverse:
            exact_map[col] = reverse[key]
        else:
            unmapped.append(col)

    fuzzy_suggestions = {}
    if fuzzy and HAS_RAPIDFUZZ and unmapped:
        # Prepara lista de claves reverse (normalizadas)
        canon_keys = list(reverse.keys())
        for col in unmapped:
            col_norm = normalize(col)
            match = process.extractOne(col_norm, canon_keys, scorer=fuzz.WRatio)
            if match:
                candidate_norm, score, _ = match
                if score >= fuzzy_threshold:
                    fuzzy_suggestions[col] = reverse[candidate_norm]

    return exact_map, unmapped, fuzzy_suggestions

def apply_mapping(df, mapping_dict: Dict[str, str]):
    return df.rename(columns=mapping_dict)

def columns_needed(mapping: dict) -> List[str]:
    return mapping["canonical_features"]