"""
Prompt templates for VLM-based pedestrian crossing compliance and jaywalking classification.
"""

# Canonical zero-shot rule prompt (baseline verified across experiments)
CANONICAL_PROMPT = (
    "Analyze this video frame of a pedestrian crossing a road. "
    "Rules: GREEN light + crossing = COMPLIANT. "
    "Crossing sign + cars yielding = COMPLIANT. "
    "Zebra markings + pedestrian on them = COMPLIANT. "
    "No light + no sign + no crosswalk + on road = JAYWALKING. "
    "RED light + crossing = JAYWALKING. "
    "Classify as JAYWALKING or COMPLIANT. Reply with only one word."
)

# Detailed prompt for right-of-way awareness (for experimental testing)
RIGHT_OF_WAY_PROMPT = (
    "Analyze this dashcam video frame of a pedestrian crossing a road. "
    "Traffic Rules: "
    "1. COMPLIANT: Pedestrian is crossing on painted crosswalk (zebra markings), OR has a GREEN pedestrian signal, "
    "OR is crossing at an intersection where oncoming vehicles are clearly stopped/yielding to them. "
    "2. JAYWALKING: Pedestrian is crossing against a RED pedestrian/traffic signal, OR crossing on an active multi-lane "
    "roadway away from any intersection/crosswalk without vehicles yielding. "
    "Classify as JAYWALKING or COMPLIANT. Reply with only the single classification word."
)


# Prompt V2: Context-aware prompt addressing sidewalk/curb, yielding, parking lots, and unmarked crossings
PROMPT_V2 = (
    "Analyze this dashcam video frame of a pedestrian near or crossing a roadway.\n"
    "Rules for Classification:\n"
    "1. COMPLIANT:\n"
    "   - Pedestrian is on the sidewalk, walking along the road edge, or waiting at the curb.\n"
    "   - Pedestrian is crossing with a GREEN signal, OR on marked crosswalk/zebra markings.\n"
    "   - Pedestrian is in a parking lot, driveway, plaza, or shared pedestrian zone.\n"
    "   - Pedestrian is crossing at an intersection or street where approaching vehicles are stopped/yielding.\n"
    "   - NOTE: Absence of painted zebra markings alone is NOT sufficient evidence of jaywalking if vehicles are yielding or the pedestrian is in a parking lot/on a sidewalk.\n"
    "2. JAYWALKING:\n"
    "   - Pedestrian is actively crossing against a RED traffic/pedestrian signal.\n"
    "   - Pedestrian is stepping/crossing into active traffic on a roadway away from an intersection without vehicles yielding.\n"
    "Classify as JAYWALKING or COMPLIANT. Reply with only one word: JAYWALKING or COMPLIANT."
)


# Temporal multi-frame prompt (Experiment 3)
TEMPORAL_PROMPT = (
    "These 3 images show consecutive observations of the SAME pedestrian event in chronological temporal order.\n"
    "Reason about the changes across frames:\n"
    "- Did the pedestrian actually enter the active roadway, or remain on the sidewalk/curb?\n"
    "- Is the pedestrian moving across the road?\n"
    "- Did the oncoming vehicle stop or yield, or is there visible evidence of right-of-way?\n"
    "Classify as JAYWALKING or COMPLIANT. Reply with only one word: JAYWALKING or COMPLIANT."
)

# Temporal multi-frame prompt with pedestrian motion context (Experiment 4A)
TEMPORAL_MOTION_PROMPT = (
    "These 3 images show consecutive observations of the SAME pedestrian event in chronological temporal order.\n\n"
    "{pedestrian_motion}\n\n"
    "Reason about the changes across frames and pedestrian motion context:\n"
    "- Did the pedestrian actually enter the active roadway, or remain on the sidewalk/curb?\n"
    "- Is the pedestrian moving across the road?\n"
    "- Did the oncoming vehicle stop or yield, or is there visible evidence of right-of-way?\n"
    "Classify as JAYWALKING or COMPLIANT. Reply with only one word: JAYWALKING or COMPLIANT."
)

# Temporal multi-frame prompt with pedestrian motion and vehicle interaction context (Experiment 4B)
TEMPORAL_VEHICLE_MOTION_PROMPT = (
    "These 3 images show consecutive observations of the SAME pedestrian event in chronological temporal order.\n\n"
    "{pedestrian_motion}\n\n"
    "{vehicle_interaction}\n\n"
    "Reason about the changes across frames and the structured interaction context:\n"
    "- Did the pedestrian actually enter the active roadway, or remain on the sidewalk/curb?\n"
    "- Did oncoming or ego vehicles stop/yield to the pedestrian, or is there visible evidence of right-of-way?\n"
    "- Is the pedestrian crossing in front of active moving traffic without vehicles yielding?\n"
    "Classify as JAYWALKING or COMPLIANT. Reply with only one word: JAYWALKING or COMPLIANT."
)


def get_prompt(prompt_name: str = "canonical") -> str:
    """Returns the requested prompt template string."""
    prompts = {
        "canonical": CANONICAL_PROMPT,
        "right_of_way": RIGHT_OF_WAY_PROMPT,
        "v2": PROMPT_V2,
        "temporal": TEMPORAL_PROMPT,
        "temporal_motion": TEMPORAL_MOTION_PROMPT,
        "temporal_vehicle_motion": TEMPORAL_VEHICLE_MOTION_PROMPT,
        "v4b": TEMPORAL_VEHICLE_MOTION_PROMPT,
    }
    return prompts.get(prompt_name, CANONICAL_PROMPT)




