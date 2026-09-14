"""Domain-specific JSON schemas and prompts for structured extraction."""

# Shared page metadata fields template
_PAGE_META_FIELDS = """  "summary": "<2-4 sentences describing this specific drawing in real detail - what it shows, its apparent purpose, and its key visual/layout characteristics. Not a one-liner.>",
  "page_type": "<short classification of what this page actually is, e.g. Site Plan, Floor Plan, Structural Framing Plan, Schedule, Detail, Elevation, Section, Mechanical Layout, Cover Sheet, etc.>",
  "insights": ["<3-6 specific, detailed observations about this page beyond the element list - e.g. approximate counts and layout pattern if elements were too numerous to list individually, notable dimensions/callouts/annotations, drawing conventions or grid system used, title block/revision info, or why no elements of this domain appear. Each insight should be a full sentence with real detail, not a fragment.>"],
  "confidence": "<your own honest self-assessment: high|medium|low>",
  "confidence_notes": "<a full sentence explaining that confidence rating with specifics - e.g. what exactly was hard to read, or what made this page clear>",
"""

# ============================================
# 1. CIVIL ENGINEERING DOMAIN
# ============================================

civil_json_schema = (
    "{\n"
    + _PAGE_META_FIELDS
    + '  "elements": [{"id": "<label as drawn, e.g. B1, or null if unreadable>", "type": "<Beam|Column|Slab|Wall|Footing|Pile|Rebar|Connection|...>", "material": "<Concrete|Steel|Timber|Masonry|...>", "...": "<any other real property you can determine, e.g. width, height, length, quantity, grid_location, span, section - add as extra keys, omit if not applicable>"}]\n'
    + "}\n"
)

civil_prompt = (
    "Rapidly identify and extract all actual structural elements visible in this civil/structural engineering drawing. "
    "Focus on: beams, columns, slabs, walls, footings, piles, connections, reinforcement details, and any labeled dimensions or specifications. "
    "Output in JSON format with element IDs as they appear in the drawing."
)

# ============================================
# 2. ELECTRICAL SCHEMATIC DOMAIN
# ============================================

electrical_json_schema = (
    "{\n"
    + _PAGE_META_FIELDS
    + '  "components": [{"id": "<designator as drawn, e.g. R1, C1, U1, or null if unreadable>", "type": "<Resistor|Capacitor|Inductor|Diode|Transistor|IC|Relay|Switch|Connector|Transformer|...>", "value": "<value/rating as drawn, e.g. 10kΩ, 100μF, 5V, or null>", "...": "<any other real property you can determine, e.g. package, pin_count, tolerance, voltage_rating - add as extra keys, omit if not applicable>"}]\n'
    + "}\n"
)

electrical_prompt = (
    "Rapidly extract all actual electrical components and their connections visible in this schematic. "
    "Identify: resistors, capacitors, inductors, diodes, transistors, ICs, relays, switches, connectors, and any other discrete or integrated components. "
    "Include the designator (reference designator) and value/rating as marked on the schematic. "
    "Output in JSON format."
)

# ============================================
# 3. CHIP / MICROELECTRONICS DOMAIN
# ============================================

chip_json_schema = (
    "{\n"
    + _PAGE_META_FIELDS
    + '  "components": [{"id": "<label as drawn, e.g. U1, BLOCK_A, M1, or null if unreadable>", "type": "<NAND|NOR|AND|OR|NOT|Transistor|Cell|Block|Memory|Cache|ALU|Register|Multiplexer|Decoder|...>", "location": "<quadrant/grid location in this layout, e.g. top-left, center, P3, Q4>", "...": "<any other real property you can determine, e.g. size, gate_count, dimensions, layer, connection_type - add as extra keys, omit if not applicable>"}]\n'
    + "}\n"
)

chip_prompt = (
    "Rapidly extract all actual chip blocks, logic gates, memory cells, and layout elements visible in this microelectronics/IC layout diagram. "
    "Identify: transistor arrangements, logic gate arrays, memory blocks, functional blocks, connections, and any labeled regions or hierarchies. "
    "Include the block/cell ID and its location within the chip layout. "
    "Output in JSON format."
)


# ============================================
# Export all schemas and prompts
# ============================================

__all__ = [
    "civil_prompt",
    "civil_json_schema",
    "electrical_prompt",
    "electrical_json_schema",
    "chip_prompt",
    "chip_json_schema",
]
