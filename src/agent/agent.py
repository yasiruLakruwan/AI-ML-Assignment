from src.agent.prompt import *
from src.logger import get_logger

logger = get_logger(__name__)

class TemporalAgent:
    """
    Agentic temporal reasoning layer for elderly-person
    video analysis.

    The agent:
    1. Analyzes the current frame.
    2. Decides whether more context is required.
    3. Requests previous/following temporal context.
    4. Combines the evidence.
    5. Produces the final event decision.
    """

    def __init__(self, vlm):

        self.vlm = vlm

    # =========================================================
    # STEP 1: ANALYZE CURRENT FRAME
    # =========================================================

    def analyze_current(self, frame):

        print(
        "\n[AGENT] Analyzing current frame..."
        )

        result = self.vlm.analyze_frame(
            frame,
            CURRENT_OBSERVATION_PROMPT
        )
        
        return result

    # =========================================================
    # STEP 2: DECIDE WHAT TO DO NEXT
    # =========================================================

    def decide_action(self, observation):
        logger.info("=========================Deciding the action====================")
        # Check whether Gemini successfully returned a result
        if "error" in observation:

            print(
                "[AGENT] Gemini response error."
            )

            return "UNKNOWN"

        # -----------------------------------------------------
        # Current frame is sufficient
        # -----------------------------------------------------

        if observation.get("sufficient") is True:

            print(
                "[AGENT] Current frame is sufficient."
            )

            return "NONE"

        # -----------------------------------------------------
        # Current frame is NOT sufficient
        # -----------------------------------------------------

        action = observation.get(
            "action",
            "ANALYZE_PREVIOUS_AND_FOLLOWING"
        )

        valid_actions = {
            "ANALYZE_PREVIOUS",
            "ANALYZE_FOLLOWING",
            "ANALYZE_PREVIOUS_AND_FOLLOWING",
            "CHECK_BED_REGION",
            "NONE"
        }

        if action not in valid_actions:

            print(
                f"[AGENT] Unknown action '{action}'. "
                f"Using temporal analysis."
            )

            action = "ANALYZE_PREVIOUS_AND_FOLLOWING"

        print(
            f"[AGENT] Action selected: {action}"
        )

        print(
            f"[AGENT] Reason: "
            f"{observation.get('reason', 'No reason provided.')}"
        )
        
        return action

    # =========================================================
    # STEP 3: ANALYZE PREVIOUS TEMPORAL CONTEXT
    # =========================================================

    def analyze_previous(self, frame_paths):

        if not frame_paths:

            print(
                "[AGENT] No previous frames available."
            )

            return {
                "finding": "NO_PREVIOUS_CONTEXT",
                "evidence": "No previous frames were available.",
                "confidence": 0.0
            }

        print(
            "\n[AGENT] Analyzing previous temporal segment..."
        )

        print(
            f"[AGENT] Number of previous frames: "
            f"{len(frame_paths)}"
        )

        result = self.vlm.analyze_frames(
            frame_paths,
            PREVIOUS_CONTEXT_PROMPT
        )

        print(
            "[AGENT] Previous finding:"
        )

        print(result)

        return result

    # =========================================================
    # STEP 4: ANALYZE FOLLOWING TEMPORAL CONTEXT
    # =========================================================

    def analyze_following(self, frame_paths):

        if not frame_paths:

            print(
                "[AGENT] No following frames available."
            )

            return {
                "finding": "NO_FOLLOWING_CONTEXT",
                "evidence": "No following frames were available.",
                "confidence": 0.0
            }

        print(
            "\n[AGENT] Analyzing following temporal segment..."
        )

        print(
            f"[AGENT] Number of following frames: "
            f"{len(frame_paths)}"
        )

        result = self.vlm.analyze_frames(
            frame_paths,
            FOLLOWING_CONTEXT_PROMPT
        )

        print(
            "[AGENT] Following finding:"
        )

        print(result)
        logger.info(f"Analyzing results: {result}")
        return result

    # =========================================================
    # STEP 5: CHECK BED REGION
    # =========================================================

    def check_bed_region(self, frame):

        prompt = """
You are analyzing an elderly-care video frame.

The person appears to be lying horizontally.

Determine whether the person is:

1. Lying on the bed
2. Lying on the floor
3. Sitting/standing
4. Unknown

Pay particular attention to the spatial relationship
between the person's body and the bed.

Return JSON only:

{
    "finding": "ON_BED",
    "evidence": "...",
    "confidence": 0.0
}

Possible findings:

ON_BED
ON_FLOOR
UNKNOWN
"""

        print(
            "\n[AGENT] Checking bed/person spatial relationship..."
        )

        result = self.vlm.analyze_frame(
            frame,
            prompt
        )

        print(
            "[AGENT] Bed region result:"
        )
        logger.info(f"Results of the analyzing : {result}")
        print(result)

        return result

    # =========================================================
    # STEP 6: FINAL TEMPORAL DECISION
    # =========================================================

    def final_decision(
        self,
        current,
        previous=None,
        following=None
    ):

        prompt = FINAL_DECISION_PROMPT

        prompt += "\n\nCURRENT OBSERVATION:\n"
        prompt += str(current)

        prompt += "\n\nPREVIOUS EVIDENCE:\n"
        prompt += str(previous)

        prompt += "\n\nFOLLOWING EVIDENCE:\n"
        prompt += str(following)

        print(
            "\n[AGENT] Making final temporal decision..."
        )

        # IMPORTANT:
        # This is text reasoning, so use analyze_text()
        result = self.vlm.analyze_text(prompt)

        print(
            "[AGENT] Final decision:"
        )

        logger.info(f"Agent's final decission : {result}")
        
        print(result)

        return result