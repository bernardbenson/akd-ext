"""Common data structures and enums for akd_ext."""

from enum import StrEnum


class NASASMDDivision(StrEnum):
    """NASA Science Mission Directorate (SMD) divisions."""

    ASTROPHYSICS = "Astrophysics"
    HELIOPHYSICS = "Heliophysics"
    EARTH_SCIENCE = "Earth Science"
    BIOLOGICAL_PHYSICAL_SCIENCES = "Biological and Physical Sciences"
    PLANETARY_SCIENCE = "Planetary Science"
    OTHER = "Other"


class SDEIndexedDocumentType(StrEnum):
    """Document types available in the SDE."""

    DATA = "Data"
    IMAGES = "Images"
    DOCUMENTATION = "Documentation"
    SOFTWARE_TOOLS = "Software and Tools"
    MISSIONS_INSTRUMENTS = "Missions and Instruments"


class SDESearchEndpoint(StrEnum):
    """SDE Search API endpoints selectable by the caller.

    ``generic`` targets the cross-source ``/api/search`` endpoint; every other
    value targets a source-specific ``/api/<source>/search`` endpoint.
    """

    GENERIC = "generic"
    WEB = "web"
    CMR = "cmr"
    PDS3 = "pds3"
    PDS4 = "pds4"
    SPASE = "spase"
    GCN = "gcn"
    HEK = "hek"
    NAVO = "navo"
    OSDR = "osdr"
    CODE = "code"


class SDESearchType(StrEnum):
    """Retrieval strategy used by the SDE Search API."""

    HYBRID = "hybrid"
    KEYWORD = "keyword"
    VECTOR = "vector"


class SDECitationType(StrEnum):
    """Identifier type selected by the citation-normalization hierarchy."""

    DOI = "doi"
    PDS_LID = "pds_lid"
    IVO_ID = "ivo_id"
    BPS_OSDR_ID = "bps_osdr_id"
    URL = "url"
    MISSING = "missing"


class SDECitationStatus(StrEnum):
    """Availability and quality of the selected citation."""

    COMPLETE = "complete"
    FALLBACK = "fallback"
    MISSING = "missing"


class SDEErrorType(StrEnum):
    """Structured error categories returned in the failure contract."""

    VALIDATION_ERROR = "validation_error"
    UPSTREAM_ERROR = "upstream_error"
    EMPTY_RESULTS = "empty_results"
    RETRYABLE_ERROR = "retryable_error"
