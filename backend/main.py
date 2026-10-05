import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag import (
    search_knowledge,
    get_expeditions_for_station,
)

from backend.temperature_ml import (
    predict_temperature,
    station_display_name,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="POLAR AI API",
    description=(
        "Semantic RAG + Temperature ML + Outreach Content API "
        "for the POLAR Science Portal"
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BASIC TEXT HELPERS
# ============================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_values(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            clean_text(item)
            for item in value
            if clean_text(item)
        ]

    text = clean_text(value)

    if not text:
        return []

    parts = re.split(r",|;|\|", text)

    return [
        clean_text(part)
        for part in parts
        if clean_text(part)
    ]


def join_naturally(items: List[str]) -> str:
    items = [
        clean_text(item)
        for item in items
        if clean_text(item)
    ]

    if not items:
        return ""

    if len(items) == 1:
        return items[0]

    if len(items) == 2:
        return f"{items[0]} and {items[1]}"

    return ", ".join(items[:-1]) + f", and {items[-1]}"


def first_non_empty(
    metadata: Dict[str, Any],
    *keys: str,
) -> str:
    for key in keys:
        value = clean_text(metadata.get(key))

        if value:
            return value

    return ""


# ============================================================
# PROFESSIONAL CONTENT CLEANER
# ============================================================

def clean_professional_content(text: str) -> str:
    """
    Final safety layer for generated outreach content.

    Removes:
    - emojis
    - hashtags
    - decorative symbols

    Keeps normal scientific punctuation and formatting.
    """

    if not text:
        return ""

    # Remove emoji and pictographic Unicode ranges.
    text = re.sub(
        r"[\U0001F300-\U0001FAFF"
        r"\U00002700-\U000027BF"
        r"\U00002600-\U000026FF"
        r"\U00002300-\U000023FF]",
        "",
        text,
    )

    # Remove hashtag words.
    text = re.sub(
        r"(?<!\w)#[A-Za-z0-9_]+",
        "",
        text,
    )

    # Remove decorative bullet characters.
    text = re.sub(
        r"[•●○▪◦◆◇★☆✓✔️]",
        "",
        text,
    )

    # Remove repeated spaces.
    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    # Clean excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# METADATA NORMALIZATION
# ============================================================

def normalize_metadata(
    metadata: Optional[Dict[str, Any]]
) -> Dict[str, Any]:

    if not metadata:
        return {}

    normalized = dict(metadata)

    normalized["source"] = clean_text(
        normalized.get("source")
    ).lower()

    normalized["title"] = first_non_empty(
        normalized,
        "title",
        "name",
        "document_title",
        "expedition_title",
        "station_name",
    )

    normalized["name"] = first_non_empty(
        normalized,
        "name",
        "title",
        "station_name",
        "expedition_title",
        "document_title",
    )

    normalized["year"] = first_non_empty(
        normalized,
        "year",
        "expedition_year",
        "publication_year",
    )

    normalized["region"] = first_non_empty(
        normalized,
        "region",
        "polar_region",
        "location",
    )

    normalized["location"] = first_non_empty(
        normalized,
        "location",
        "site",
    )

    normalized["station_name"] = first_non_empty(
        normalized,
        "station_name",
        "station",
    )

    normalized["station_id"] = first_non_empty(
        normalized,
        "station_id",
    )

    normalized["expedition_id"] = first_non_empty(
        normalized,
        "expedition_id",
    )

    normalized["document_id"] = first_non_empty(
        normalized,
        "document_id",
    )

    normalized["research_focus"] = split_values(
        normalized.get("research_focus")
    )

    normalized["research_areas"] = split_values(
        normalized.get("research_areas")
    )

    normalized["institutions"] = split_values(
        normalized.get("institutions")
    )

    normalized["scientists"] = split_values(
        normalized.get("scientists")
    )

    normalized["overview"] = clean_text(
        normalized.get("overview")
    )

    normalized["coordinates"] = clean_text(
        normalized.get("coordinates")
    )

    normalized["established_year"] = clean_text(
        normalized.get("established_year")
    )

    normalized["objectives"] = clean_text(
        normalized.get("objectives")
    )

    normalized["description"] = clean_text(
        normalized.get("description")
    )

    normalized["abstract"] = clean_text(
        normalized.get("abstract")
    )

    normalized["content"] = clean_text(
        normalized.get("content")
    )

    return normalized


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def normalize_result(
    result: Dict[str, Any]
) -> Dict[str, Any]:

    metadata = normalize_metadata(
        result.get("metadata", {})
    )

    return {
        "score": float(
            result.get("score", 0)
        ),
        "content": clean_text(
            result.get("content")
        ),
        "metadata": metadata,
    }


def normalize_results(
    results: Any
) -> List[Dict[str, Any]]:

    if not results:
        return []

    normalized = []

    for result in results:
        if not isinstance(result, dict):
            continue

        normalized.append(
            normalize_result(result)
        )

    normalized.sort(
        key=lambda item: item.get("score", 0),
        reverse=True,
    )

    return normalized


# ============================================================
# SOURCE TYPE
# ============================================================

def get_source_type(
    result: Dict[str, Any]
) -> str:

    metadata = result.get(
        "metadata",
        {},
    )

    source = clean_text(
        metadata.get("source")
    ).lower()

    aliases = {
        "station": "stations",
        "stations": "stations",
        "expedition": "expeditions",
        "expeditions": "expeditions",
        "document": "documents",
        "documents": "documents",
        "media": "media",
        "story": "stories",
        "stories": "stories",
        "ai_content": "ai_content",
    }

    return aliases.get(
        source,
        source,
    )


# ============================================================
# ENTITY SELECTION
# ============================================================

def get_entity_candidates(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    if not results:
        return []

    candidates = []

    for result in results:

        source_type = get_source_type(result)

        if source_type in {
            "stations",
            "expeditions",
            "documents",
        }:
            candidates.append(result)

    return candidates


def get_primary_entity(
    results: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:

    candidates = get_entity_candidates(results)

    if not candidates:
        return None

    return candidates[0]


# ============================================================
# QUERY INTENT
# ============================================================

def is_expedition_query(
    query: str
) -> bool:

    query_lower = clean_text(query).lower()

    expedition_terms = [
        "expedition",
        "expeditions",
        "indian arctic expedition",
        "indian antarctic expedition",
        "antarctic expedition",
        "arctic expedition",
    ]

    return any(
        term in query_lower
        for term in expedition_terms
    )


def is_expedition_objective_query(
    query: str
) -> bool:

    query_lower = clean_text(query).lower()

    objective_terms = [
        "objective",
        "objectives",
        "main objective",
        "main objectives",
        "purpose",
        "aim",
        "aims",
        "goal",
        "goals",
        "what was the expedition about",
        "what were they studying",
        "what did the expedition study",
    ]

    return any(
        term in query_lower
        for term in objective_terms
    )


def is_relationship_query(
    query: str,
    results: List[Dict[str, Any]],
) -> bool:

    query_lower = clean_text(query).lower()

    relationship_terms = {
        "associated",
        "association",
        "associated with",
        "related",
        "related to",
        "linked",
        "linked to",
        "connected",
        "connected to",
        "belong",
        "belongs to",
        "part of",
        "connection",
        "relationship",
        "relations",
        "which expedition",
        "which expeditions",
        "which station",
        "which stations",
    }

    return any(
        term in query_lower
        for term in relationship_terms
    )


# ============================================================
# TEMPERATURE ML
# ============================================================

def is_temperature_prediction_query(
    query: str
) -> bool:

    q = clean_text(query).lower()

    has_station = bool(
        re.search(
            r"\b(?:maitri|bharati|himadri)\b",
            q,
        )
    )

    has_temperature = bool(
        re.search(
            r"\b(?:temperature|temp|thermal)\b",
            q,
        )
    )

    has_prediction_intent = bool(
        re.search(
            r"\b(?:"
            r"predict(?:ed|ing|ion)?|"
            r"forecast(?:s|ed|ing)?|"
            r"estim(?:ate|ated|ating|ation)|"
            r"expect(?:ed|ing)?|"
            r"what\s+(?:will|would)"
            r")\b",
            q,
        )
    )

    return (
        has_station
        and has_temperature
        and has_prediction_intent
    )


def extract_temperature_station(
    query: str
) -> Optional[str]:

    q = clean_text(query).lower()

    match = re.search(
        r"\b(maitri|bharati|himadri)\b",
        q,
    )

    return (
        match.group(1)
        if match
        else None
    )


def extract_number(
    query: str,
    patterns: List[str],
) -> Optional[float]:

    for pattern in patterns:

        match = re.search(
            pattern,
            query,
            re.IGNORECASE,
        )

        if match:

            try:
                return float(
                    match.group(1)
                )

            except ValueError:
                return None

    return None


def parse_temperature_prediction(
    query: str
) -> Dict[str, Any]:

    station = extract_temperature_station(
        query
    )

    ap = extract_number(
        query,
        [
            r"(?:pressure|ap)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"
        ],
    )

    rh = extract_number(
        query,
        [
            r"(?:humidity|rh)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"
        ],
    )

    ws = extract_number(
        query,
        [
            r"(?:wind\s*speed|ws)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"
        ],
    )

    wd = extract_number(
        query,
        [
            r"(?:wind\s*direction|direction|wd)\s*(?:is|=|:)?\s*(-?\d+(?:\.\d+)?)"
        ],
    )

    date_time: Optional[datetime] = None

    date_time_match = re.search(
        r"((?:"
        r"\d{4}-\d{1,2}-\d{1,2}|"
        r"\d{1,2}\s+(?:january|february|march|april|may|june|july|"
        r"august|september|october|november|december)\s+\d{4}"
        r")"
        r"(?:[t\s]+)"
        r"(?:at\s+)?"
        r"(?:"
        r"\d{1,2}:\d{2}(?:\s*(?:am|pm))?|"
        r"\d{1,2}\s*(?:am|pm)"
        r"))",
        query,
        re.IGNORECASE,
    )

    if date_time_match:

        date_text = date_time_match.group(1)

        try:

            date_time = (
                pd.to_datetime(
                    date_text,
                    errors="raise",
                ).to_pydatetime()
            )

        except (TypeError, ValueError):

            date_time = None

    return {
        "station": station,
        "date_time": date_time,
        "ap": ap,
        "rh": rh,
        "ws": ws,
        "wd": wd,
    }


def generate_temperature_answer(
    query: str
) -> Dict[str, Any]:

    parsed = parse_temperature_prediction(
        query
    )

    station = parsed["station"]
    date_time = parsed["date_time"]
    ap = parsed["ap"]
    rh = parsed["rh"]
    ws = parsed["ws"]
    wd = parsed["wd"]

    if not station:

        return {
            "answer": (
                "Temperature prediction is available for "
                "Maitri, Bharati, and Himadri stations."
            ),
            "citations": [],
        }

    missing = []

    if date_time is None:
        missing.append("date and time")

    if ap is None:
        missing.append("pressure (hPa)")

    if rh is None:
        missing.append("humidity (%)")

    if ws is None:
        missing.append("wind speed (m/s)")

    if wd is None:
        missing.append("wind direction (degrees)")

    if missing:

        missing_text = join_naturally(
            missing
        )

        return {
            "answer": (
                f"To predict the temperature at "
                f"**{station_display_name(station)}**, "
                f"I need the following inputs: "
                f"**{missing_text}**.\n\n"
                "Example:\n\n"
                "> Predict the temperature at Bharati "
                "on 15 January 2026 at 12 PM, "
                "pressure 980 hPa, humidity 70%, "
                "wind speed 10 m/s, wind direction 180°."
            ),
            "citations": [],
        }

    try:

        prediction = predict_temperature(
            station=station,
            date_time=date_time,
            ap=ap,
            rh=rh,
            ws=ws,
            wd=wd,
        )

    except ValueError as exc:

        return {
            "answer": str(exc),
            "citations": [],
        }

    except Exception as exc:

        print(
            "TEMPERATURE ML ERROR:",
            repr(exc),
        )

        raise

    station_name = station_display_name(
        station
    )

    formatted_date = date_time.strftime(
        "%d %B %Y"
    )

    formatted_time = date_time.strftime(
        "%I:%M %p"
    )

    answer = (
        f"### Temperature Prediction — {station_name}\n\n"
        f"**Date:** {formatted_date}\n\n"
        f"**Time:** {formatted_time}\n\n"
        f"**Predicted Temperature:** "
        f"**{prediction:.2f} °C**\n\n"
        f"**Input Conditions:**\n"
        f"- Pressure: {ap:.1f} hPa\n"
        f"- Humidity: {rh:.1f}%\n"
        f"- Wind Speed: {ws:.1f} m/s\n"
        f"- Wind Direction: {wd:.1f}°"
    )

    return {
        "answer": answer,
        "citations": [],
    }


# ============================================================
# EXPEDITION HELPERS
# ============================================================

def find_best_expedition(
    results: List[Dict[str, Any]],
    query: str,
) -> Optional[Dict[str, Any]]:

    expedition_results = [
        result
        for result in results
        if get_source_type(result) == "expeditions"
    ]

    if not expedition_results:
        return None

    query_lower = clean_text(query).lower()

    for result in expedition_results:

        metadata = result.get(
            "metadata",
            {},
        )

        title = first_non_empty(
            metadata,
            "title",
            "name",
        ).lower()

        slug = clean_text(
            metadata.get("slug")
        ).lower()

        if title and title in query_lower:
            return result

        if slug and slug in query_lower:
            return result

    return expedition_results[0]


# ============================================================
# STATION ANSWER
# ============================================================

def format_station_answer(
    result: Dict[str, Any]
) -> str:

    metadata = result["metadata"]

    name = first_non_empty(
        metadata,
        "name",
        "title",
        "station_name",
    )

    location = first_non_empty(
        metadata,
        "location",
        "site",
    )

    region = clean_text(
        metadata.get("region")
    )

    overview = first_non_empty(
        metadata,
        "overview",
        "description",
    )

    research_focus = metadata.get(
        "research_focus",
        [],
    )

    sections = []

    if name:
        sections.append(
            f"### {name}"
        )

    if location:
        sections.append(
            f"**Location:** {location}"
        )

    if region:
        sections.append(
            f"**Region:** {region}"
        )

    if research_focus:
        sections.append(
            "**Research Focus:** "
            + join_naturally(
                research_focus
            )
        )

    if overview:
        sections.append(
            f"**Overview:** {overview}"
        )

    if not sections:

        content = result.get(
            "content",
            "",
        )

        if content:
            sections.append(
                content
            )

    return "\n\n".join(
        sections
    )


# ============================================================
# EXPEDITION ANSWER
# ============================================================

def format_expedition_answer(
    result: Dict[str, Any]
) -> str:

    metadata = result["metadata"]

    title = first_non_empty(
        metadata,
        "title",
        "name",
    )

    year = metadata.get(
        "year",
        "",
    )

    region = metadata.get(
        "region",
        "",
    )

    station_name = metadata.get(
        "station_name",
        "",
    )

    objectives = metadata.get(
        "objectives",
        "",
    )

    description = metadata.get(
        "description",
        "",
    )

    research_areas = metadata.get(
        "research_areas",
        [],
    )

    institutions = metadata.get(
        "institutions",
        [],
    )

    scientists = metadata.get(
        "scientists",
        [],
    )

    sections = []

    if title:

        heading = title

        if year:
            heading += f" ({year})"

        sections.append(
            f"### {heading}"
        )

    if region:
        sections.append(
            f"**Region:** {region}"
        )

    if station_name:
        sections.append(
            f"**Station:** {station_name}"
        )

    if objectives:
        sections.append(
            f"**Objectives:** {objectives}"
        )

    if description:
        sections.append(
            f"**Description:** {description}"
        )

    if research_areas:
        sections.append(
            "**Research Areas:** "
            + join_naturally(
                research_areas
            )
        )

    if institutions:
        sections.append(
            "**Institutions:** "
            + join_naturally(
                institutions
            )
        )

    if scientists:
        sections.append(
            "**Scientists:** "
            + join_naturally(
                scientists
            )
        )

    if not sections:

        content = result.get(
            "content",
            "",
        )

        if content:
            sections.append(
                content
            )

    return "\n\n".join(
        sections
    )


# ============================================================
# DOCUMENT ANSWER
# ============================================================

def format_document_answer(
    result: Dict[str, Any]
) -> str:

    metadata = result["metadata"]

    title = first_non_empty(
        metadata,
        "title",
        "name",
    )

    year = metadata.get(
        "year",
        "",
    )

    region = metadata.get(
        "region",
        "",
    )

    description = metadata.get(
        "description",
        "",
    )

    document_type = first_non_empty(
        metadata,
        "document_type",
        "type",
        "category",
    )

    institution = first_non_empty(
        metadata,
        "institution",
        "organization",
        "publisher",
    )

    abstract = metadata.get(
        "abstract",
        "",
    )

    content = result.get(
        "content",
        "",
    )

    sections = []

    if title:

        heading = title

        if year:
            heading += f" ({year})"

        sections.append(
            f"### {heading}"
        )

    if document_type:
        sections.append(
            f"**Type:** {document_type}"
        )

    if region:
        sections.append(
            f"**Region:** {region}"
        )

    if institution:
        sections.append(
            f"**Institution:** {institution}"
        )

    if abstract:
        sections.append(
            f"**Abstract:** {abstract}"
        )

    elif description:
        sections.append(
            f"**About:** {description}"
        )

    elif content:
        sections.append(
            f"**About:** {content}"
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# GENERIC ANSWER
# ============================================================

def format_primary_answer(
    result: Dict[str, Any]
) -> str:

    source_type = get_source_type(
        result
    )

    if source_type == "stations":
        return format_station_answer(result)

    if source_type == "expeditions":
        return format_expedition_answer(result)

    if source_type == "documents":
        return format_document_answer(result)

    content = result.get(
        "content",
        "",
    )

    if content:
        return content

    metadata = result.get(
        "metadata",
        {},
    )

    title = first_non_empty(
        metadata,
        "title",
        "name",
    )

    if title:
        return f"### {title}"

    return ""


# ============================================================
# STATION TO EXPEDITION RELATIONSHIP
# ============================================================

def generate_station_relationship_answer(
    station_result: Dict[str, Any]
) -> Dict[str, Any]:

    metadata = station_result.get(
        "metadata",
        {},
    )

    station_name = first_non_empty(
        metadata,
        "name",
        "title",
        "station_name",
    )

    station_id = first_non_empty(
        metadata,
        "station_id",
    )

    expeditions = get_expeditions_for_station(
        station_id=station_id or None,
        station_name=station_name or None,
    )

    answer_parts = []

    station_text = format_station_answer(
        station_result
    )

    if station_text:
        answer_parts.append(
            station_text
        )

    if expeditions:

        answer_parts.append(
            "### Related Expeditions"
        )

        for expedition in expeditions:

            if not isinstance(
                expedition,
                dict,
            ):
                continue

            expedition_metadata = normalize_metadata(
                expedition.get(
                    "metadata",
                    expedition,
                )
            )

            title = first_non_empty(
                expedition_metadata,
                "title",
                "name",
            )

            year = expedition_metadata.get(
                "year",
                "",
            )

            region = expedition_metadata.get(
                "region",
                "",
            )

            line = title

            if year:
                line += f" ({year})"

            if region:
                line += f" — {region}"

            if line:
                answer_parts.append(
                    f"- {line}"
                )

    return {
        "answer": "\n\n".join(
            answer_parts
        ),
        "related_results": expeditions,
    }


# ============================================================
# CITATIONS
# ============================================================

def make_citation(
    result: Dict[str, Any]
) -> Dict[str, Any]:

    metadata = result.get(
        "metadata",
        {},
    )

    source_type = get_source_type(
        result
    )

    title = first_non_empty(
        metadata,
        "title",
        "name",
    )

    source_id = first_non_empty(
        metadata,
        "id",
        "document_id",
        "expedition_id",
        "station_id",
    )

    slug = first_non_empty(
        metadata,
        "slug",
    )

    route = ""

    if source_type == "stations":

        if slug:
            route = f"/explore/{slug}"

        elif source_id:
            route = f"/explore/{source_id}"

    elif source_type == "expeditions":

        if slug:
            route = f"/expeditions/{slug}"

        elif source_id:
            route = f"/expeditions/{source_id}"

    elif source_type == "documents":

        if slug:
            route = f"/knowledge/{slug}"

        elif source_id:
            route = f"/knowledge/{source_id}"

    return {
        "title": title or "POLAR Knowledge",
        "source": source_type,
        "route": route,
        "score": round(
            float(
                result.get(
                    "score",
                    0,
                )
            ),
            4,
        ),
    }


def build_entity_citations(
    primary: Dict[str, Any]
) -> List[Dict[str, Any]]:

    return [
        make_citation(primary)
    ]


def build_relationship_citations(
    station_result: Dict[str, Any],
    related_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    citations = [
        make_citation(station_result)
    ]

    seen = {
        (
            citations[0]["source"],
            citations[0]["title"],
            citations[0]["route"],
        )
    }

    for item in related_results:

        if not isinstance(
            item,
            dict,
        ):
            continue

        if "metadata" in item:

            normalized = normalize_result(
                item
            )

        else:

            normalized = {
                "score": 0,
                "content": "",
                "metadata": normalize_metadata(
                    item
                ),
            }

        citation = make_citation(
            normalized
        )

        key = (
            citation["source"],
            citation["title"],
            citation["route"],
        )

        if key in seen:
            continue

        seen.add(key)

        citations.append(
            citation
        )

    return citations


# ============================================================
# EXPEDITION-FOCUSED ANSWER
# ============================================================

def generate_expedition_focused_answer(
    query: str,
    results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    if not is_expedition_query(query):
        return None

    expedition = find_best_expedition(
        results,
        query,
    )

    if expedition is None:
        return None

    if is_expedition_objective_query(query):

        metadata = expedition.get(
            "metadata",
            {},
        )

        objectives = clean_text(
            metadata.get("objectives")
        )

        if objectives:

            return {
                "answer": format_expedition_answer(
                    expedition
                ),
                "citations": [
                    make_citation(
                        expedition
                    )
                ],
            }

    return {
        "answer": format_expedition_answer(
            expedition
        ),
        "citations": [
            make_citation(
                expedition
            )
        ],
    }


# ============================================================
# GENERIC ANSWER GENERATION
# ============================================================

def generate_answer(
    query: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:

    results = normalize_results(
        results
    )

    if not results:

        return {
            "answer": (
                "I couldn't find relevant information "
                "in the POLAR knowledge repository."
            ),
            "citations": [],
        }

    expedition_answer = (
        generate_expedition_focused_answer(
            query,
            results,
        )
    )

    if expedition_answer is not None:
        return expedition_answer

    primary = get_primary_entity(
        results
    )

    if primary is None:
        primary = results[0]

    primary_source = get_source_type(
        primary
    )

    if (
        primary_source == "stations"
        and is_relationship_query(
            query,
            results,
        )
    ):

        relationship = (
            generate_station_relationship_answer(
                primary
            )
        )

        return {
            "answer": relationship["answer"],
            "citations": build_relationship_citations(
                primary,
                relationship[
                    "related_results"
                ],
            ),
        }

    answer = format_primary_answer(
        primary
    )

    if not answer:

        answer = (
            "I found relevant POLAR records, "
            "but there isn't enough structured information "
            "to generate a detailed answer."
        )

    return {
        "answer": answer,
        "citations": build_entity_citations(
            primary
        ),
    }


# ============================================================
# OUTREACH PLATFORM NORMALIZATION
# ============================================================

def normalize_platform(
    platform: str
) -> str:

    value = clean_text(
        platform
    ).lower()

    value = re.sub(
        r"[_\-]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    aliases = {

        "website":
            "website",

        "website article":
            "website",

        "web article":
            "website",

        "instagram":
            "instagram",

        "instagram caption":
            "instagram",

        "instagram post":
            "instagram",

        "social media":
            "social_media",

        "social media post":
            "social_media",

        "linkedin":
            "social_media",

        "linkedin post":
            "social_media",

        "youtube":
            "youtube",

        "youtube description":
            "youtube",

        "youtube video":
            "youtube",

        "short video":
            "short_video",

        "short video script":
            "short_video",

        "video script":
            "short_video",

        "reel":
            "short_video",

        "reels":
            "short_video",

        "outreach":
            "outreach",

        "outreach article":
            "outreach",

        "public article":
            "outreach",
    }

    return aliases.get(
        value,
        value,
    )


# ============================================================
# OUTREACH FACT EXTRACTION
# ============================================================

def build_outreach_facts(
    result: Dict[str, Any]
) -> Dict[str, Any]:

    metadata = result.get(
        "metadata",
        {},
    )

    source_type = get_source_type(
        result
    )

    title = first_non_empty(
        metadata,
        "title",
        "name",
    )

    year = clean_text(
        metadata.get("year")
    )

    region = clean_text(
        metadata.get("region")
    )

    location = first_non_empty(
        metadata,
        "location",
        "site",
    )

    description = first_non_empty(
        metadata,
        "description",
        "overview",
        "abstract",
        "content",
    )

    objectives = clean_text(
        metadata.get("objectives")
    )

    research_areas = metadata.get(
        "research_areas",
        [],
    )

    research_focus = metadata.get(
        "research_focus",
        [],
    )

    institution = first_non_empty(
        metadata,
        "institution",
        "organization",
        "publisher",
    )

    document_type = first_non_empty(
        metadata,
        "document_type",
        "type",
        "category",
    )

    scientists = metadata.get(
        "scientists",
        [],
    )

    institutions = metadata.get(
        "institutions",
        [],
    )

    if not description:

        description = clean_text(
            result.get("content")
        )

    if not description:

        description = (
            "This resource is part of the "
            "POLAR Knowledge Repository."
        )

    if source_type == "expeditions":

        if title and year:

            subject_sentence = (
                f"The {title} ({year})"
            )

        elif title:

            subject_sentence = (
                f"The {title}"
            )

        else:

            subject_sentence = (
                "The selected polar expedition"
            )

    elif source_type == "stations":

        subject_sentence = (
            f"{title or 'The selected polar research station'}"
        )

    else:

        subject_sentence = (
            title
            or "The selected polar science resource"
        )

    return {

        "source_type":
            source_type,

        "title":
            title,

        "year":
            year,

        "region":
            region,

        "location":
            location,

        "description":
            description,

        "objectives":
            objectives,

        "research_areas":
            research_areas,

        "research_focus":
            research_focus,

        "institution":
            institution,

        "document_type":
            document_type,

        "scientists":
            scientists,

        "institutions":
            institutions,

        "subject_sentence":
            subject_sentence,
    }


# ============================================================
# OUTREACH CONTENT GENERATOR
# ============================================================

def generate_outreach_content(
    query: str,
    platform: str,
    result: Dict[str, Any],
) -> str:

    platform_key = normalize_platform(
        platform
    )

    supported_platforms = {
        "website",
        "instagram",
        "social_media",
        "youtube",
        "short_video",
        "outreach",
    }

    if platform_key not in supported_platforms:

        raise ValueError(
            "Unsupported platform. Choose Website Article, "
            "Instagram Caption, Social Media Post, "
            "YouTube Description, Short Video Script, "
            "or Outreach Article."
        )

    facts = build_outreach_facts(
        result
    )

    title = facts["title"]
    year = facts["year"]
    region = facts["region"]
    location = facts["location"]
    description = facts["description"]
    objectives = facts["objectives"]
    research_areas = facts["research_areas"]
    research_focus = facts["research_focus"]
    institution = facts["institution"]

    display_title = (
        title
        or "Polar Science"
    )

    # ========================================================
    # 1. WEBSITE ARTICLE
    # ========================================================

    if platform_key == "website":

        sections = []

        sections.append(
            f"Title: {display_title}"
        )

        sections.append(
            "Introduction:\n\n"
            f"{description}"
        )

        if year:
            sections.append(
                "Research Period:\n\n"
                f"{year}"
            )

        if region or location:

            location_parts = []

            if region:
                location_parts.append(
                    region
                )

            if location:
                location_parts.append(
                    location
                )

            sections.append(
                "Research Location:\n\n"
                "The documented work is associated with "
                + join_naturally(
                    location_parts
                )
                + "."
            )

        if objectives:

            sections.append(
                "Research Objectives:\n\n"
                + objectives
            )

        if research_areas:

            sections.append(
                "Research Areas:\n\n"
                "The documented research includes "
                + join_naturally(
                    research_areas
                )
                + "."
            )

        if research_focus:

            sections.append(
                "Scientific Focus:\n\n"
                "The recorded scientific focus includes "
                + join_naturally(
                    research_focus
                )
                + "."
            )

        if institution:

            sections.append(
                "Research Organisation:\n\n"
                f"{institution}"
            )

        sections.append(
            "Significance:\n\n"
            "The information documented in the POLAR "
            "Knowledge Repository provides a structured "
            "record of this polar science activity."
        )

        sections.append(
            "Conclusion:\n\n"
            f"{display_title} represents documented "
            "scientific work within the POLAR knowledge "
            "repository."
        )

        sections.append(
            "Source:\n\n"
            "POLAR Knowledge Repository"
        )

        return clean_professional_content(
            "\n\n".join(sections)
        )

    # ========================================================
    # 2. INSTAGRAM CAPTION
    # ========================================================

    if platform_key == "instagram":

        parts = []

        parts.append(
            display_title
        )

        parts.append(
            description
        )

        if objectives:

            parts.append(
                "Research objective: "
                + objectives
            )

        elif research_areas:

            parts.append(
                "Research areas: "
                + join_naturally(
                    research_areas
                )
                + "."
            )

        if region:

            parts.append(
                "Research region: "
                + region
                + "."
            )

        if location:

            parts.append(
                "Research location: "
                + location
                + "."
            )

        if research_focus:

            parts.append(
                "Scientific focus: "
                + join_naturally(
                    research_focus
                )
                + "."
            )

        parts.append(
            "Further verified information is available "
            "through the POLAR Science Portal."
        )

        return clean_professional_content(
            "\n\n".join(parts)
        )

    # ========================================================
    # 3. SOCIAL MEDIA POST
    # ========================================================

    if platform_key == "social_media":

        sections = []

        sections.append(
            display_title
        )

        if year:

            sections.append(
                f"Research period: {year}."
            )

        sections.append(
            "Research overview:\n\n"
            + description
        )

        if objectives:

            sections.append(
                "Research objective:\n\n"
                + objectives
            )

        if research_areas:

            sections.append(
                "Research areas:\n\n"
                + join_naturally(
                    research_areas
                )
                + "."
            )

        if research_focus:

            sections.append(
                "Scientific focus:\n\n"
                + join_naturally(
                    research_focus
                )
                + "."
            )

        if region:

            sections.append(
                "Region:\n\n"
                + region
            )

        if location:

            sections.append(
                "Location:\n\n"
                + location
            )

        sections.append(
            "Source:\n\n"
            "POLAR Knowledge Repository"
        )

        return clean_professional_content(
            "\n\n".join(sections)
        )

    # ========================================================
    # 4. YOUTUBE DESCRIPTION
    # ========================================================

    if platform_key == "youtube":

        video_title = (
            f"{display_title} | Polar Science"
        )

        sections = []

        sections.append(
            f"Title: {video_title}"
        )

        sections.append(
            "Overview:\n\n"
            f"This video presents documented information "
            f"about {display_title}.\n\n"
            f"{description}"
        )

        if year:

            sections.append(
                "Research Period:\n\n"
                f"{year}"
            )

        if region or location:

            locations = []

            if region:
                locations.append(
                    region
                )

            if location:
                locations.append(
                    location
                )

            sections.append(
                "Research Location:\n\n"
                + join_naturally(
                    locations
                )
                + "."
            )

        if objectives:

            sections.append(
                "Research Objective:\n\n"
                + objectives
            )

        if research_areas:

            sections.append(
                "Research Areas:\n\n"
                + join_naturally(
                    research_areas
                )
                + "."
            )

        if research_focus:

            sections.append(
                "Scientific Focus:\n\n"
                + join_naturally(
                    research_focus
                )
                + "."
            )

        if institution:

            sections.append(
                "Research Organisation:\n\n"
                + institution
            )

        sections.append(
            "Sources:\n\n"
            "POLAR Knowledge Repository"
        )

        return clean_professional_content(
            "\n\n".join(sections)
        )

    # ========================================================
    # 5. SHORT VIDEO SCRIPT
    # ========================================================

    if platform_key == "short_video":

        scenes = []

        scenes.append(
            "Scene 1: Introduction\n\n"
            "Visual: Opening view of the relevant polar "
            "research environment or station.\n\n"
            "Narration: "
            f"{display_title} is a documented subject "
            "of polar science research."
        )

        if region or location:

            setting = []

            if region:
                setting.append(
                    region
                )

            if location:
                setting.append(
                    location
                )

            scenes.append(
                "Scene 2: Research Setting\n\n"
                "Visual: Map or footage showing the "
                "research location.\n\n"
                "Narration: "
                "The documented research is associated "
                "with "
                + join_naturally(setting)
                + "."
            )

        scenes.append(
            "Scene 3: Research Overview\n\n"
            "Visual: Researchers, instruments, field "
            "observations or relevant scientific data.\n\n"
            "Narration: "
            + description
        )

        if objectives:

            scenes.append(
                "Scene 4: Research Objectives\n\n"
                "Visual: Scientific observations, "
                "measurements or research activities.\n\n"
                "Narration: "
                + objectives
            )

        elif research_areas:

            scenes.append(
                "Scene 4: Research Areas\n\n"
                "Visual: Relevant scientific fieldwork "
                "and research activities.\n\n"
                "Narration: The documented research "
                "includes "
                + join_naturally(
                    research_areas
                )
                + "."
            )

        if research_focus:

            scenes.append(
                "Scene 5: Scientific Focus\n\n"
                "Visual: Research data or scientific "
                "observations.\n\n"
                "Narration: The scientific focus includes "
                + join_naturally(
                    research_focus
                )
                + "."
            )

        else:

            scenes.append(
                "Scene 5: Research Significance\n\n"
                "Visual: Polar environment and research "
                "documentation.\n\n"
                "Narration: This documented work forms "
                "part of the scientific information "
                "available through the POLAR Knowledge "
                "Repository."
            )

        scenes.append(
            "Scene 6: Closing\n\n"
            "Visual: POLAR Science Portal interface.\n\n"
            "Narration: Explore verified polar science "
            "information through the POLAR Science Portal."
        )

        return clean_professional_content(
            "\n\n".join(scenes)
        )

    # ========================================================
    # 6. OUTREACH ARTICLE
    # ========================================================

    if platform_key == "outreach":

        sections = []

        sections.append(
            f"Title: {display_title}"
        )

        sections.append(
            "Overview:\n\n"
            + description
        )

        if year:

            sections.append(
                "Research Period:\n\n"
                + year
            )

        if region:

            sections.append(
                "Research Region:\n\n"
                + region
            )

        if location:

            sections.append(
                "Research Location:\n\n"
                + location
            )

        if objectives:

            sections.append(
                "Research Objectives:\n\n"
                + objectives
            )

        if research_areas:

            sections.append(
                "Scientific Areas:\n\n"
                "The documented work includes "
                + join_naturally(
                    research_areas
                )
                + "."
            )

        if research_focus:

            sections.append(
                "Scientific Focus:\n\n"
                + join_naturally(
                    research_focus
                )
                + "."
            )

        if institution:

            sections.append(
                "Research Organisation:\n\n"
                + institution
            )

        sections.append(
            "Importance:\n\n"
            "The POLAR Knowledge Repository provides "
            "structured access to verified information "
            "about polar research, observations and "
            "scientific activities."
        )

        sections.append(
            "Source:\n\n"
            "POLAR Knowledge Repository"
        )

        return clean_professional_content(
            "\n\n".join(sections)
        )

    raise ValueError(
        "Unsupported platform."
    )


# ============================================================
# REQUEST MODELS
# ============================================================

class AskRequest(BaseModel):
    query: str


class ContentGenerationRequest(BaseModel):
    query: str
    platform: str = "Website Article"


# ============================================================
# ROOT + HEALTH
# ============================================================

@app.get("/")
def root():

    return {
        "name": "POLAR AI",
        "status": "running",
        "version": "4.0.0",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
    }


# ============================================================
# EXISTING RAG API
# ============================================================

@app.post("/api/ask")
def ask_polar_ai(
    request: AskRequest
):

    query = clean_text(
        request.query
    )

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:

        # ----------------------------------------------------
        # TEMPERATURE ML
        # ----------------------------------------------------

        if is_temperature_prediction_query(
            query
        ):

            response = generate_temperature_answer(
                query
            )

            return {
                "query": query,
                "answer": response["answer"],
                "citations": response["citations"],
            }

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        results = search_knowledge(
            query,
            top_k=10,
        )

        # ----------------------------------------------------
        # RELEVANCE GATE
        # ----------------------------------------------------

        if (
            not results
            or results[0].get("score", 0) < 0.40
        ):

            return {
                "query": query,
                "answer": (
                    "I couldn't find anything related to your question "
                    "in the POLAR knowledge base."
                ),
                "citations": [],
            }

        # ----------------------------------------------------
        # GROUNDED ANSWER
        # ----------------------------------------------------

        response = generate_answer(
            query,
            results,
        )

        return {
            "query": query,
            "answer": response["answer"],
            "citations": response["citations"],
        }

    except Exception as exc:

        print(
            "POLAR AI ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "POLAR AI failed while processing "
                "the request."
            ),
        )


# ============================================================
# OUTREACH CONTENT GENERATION API
# ============================================================

@app.post("/api/generate-content")
def generate_content(
    request: ContentGenerationRequest
):

    query = clean_text(
        request.query
    )

    platform = clean_text(
        request.platform
    ) or "Website Article"

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:

        # ----------------------------------------------------
        # NORMALIZE PLATFORM
        # ----------------------------------------------------

        normalized_platform = normalize_platform(
            platform
        )

        print(
            "CONTENT REQUEST:",
            {
                "query": query,
                "platform": platform,
                "normalized_platform": normalized_platform,
            },
        )

        # ----------------------------------------------------
        # VALIDATE PLATFORM
        # ----------------------------------------------------

        supported_platforms = {
            "website",
            "instagram",
            "social_media",
            "youtube",
            "short_video",
            "outreach",
        }

        if normalized_platform not in supported_platforms:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported platform. Choose Website Article, "
                    "Instagram Caption, Social Media Post, "
                    "YouTube Description, Short Video Script, "
                    "or Outreach Article."
                ),
            )

        # ----------------------------------------------------
        # SEARCH POLAR KNOWLEDGE
        # ----------------------------------------------------

        results = search_knowledge(
            query,
            top_k=10,
        )

        # ----------------------------------------------------
        # RELEVANCE GATE
        # ----------------------------------------------------

        if (
            not results
            or results[0].get("score", 0) < 0.40
        ):

            return {
                "query": query,
                "platform": platform,
                "content": (
                    "I couldn't find anything related to your "
                    "request in the POLAR knowledge base."
                ),
                "citations": [],
            }

        # ----------------------------------------------------
        # NORMALIZE RESULTS
        # ----------------------------------------------------

        normalized_results = normalize_results(
            results
        )

        # ----------------------------------------------------
        # PRIMARY ENTITY
        # ----------------------------------------------------

        primary = get_primary_entity(
            normalized_results
        )

        if primary is None:
            primary = normalized_results[0]

        # ----------------------------------------------------
        # GENERATE FORMAT-SPECIFIC CONTENT
        # ----------------------------------------------------

        content = generate_outreach_content(
            query=query,
            platform=normalized_platform,
            result=primary,
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {
            "query": query,
            "platform": platform,
            "content": content,
            "citations": [
                make_citation(
                    primary
                )
            ],
        }

    except HTTPException:
        raise

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        print(
            "CONTENT GENERATION ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "POLAR content generation failed "
                "while processing the request."
            ),
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )