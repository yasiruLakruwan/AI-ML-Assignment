from src.video import VideoReader
from src.vision import PersonTracker
from src.bed import BedRegion
from src.state_classifier import StateClassifier
from src.state_machine import StateMachine
from src.config import BED_REGION
import os
from src.evaluatoer import Evaluator
from src.agent.gemini_vlm import GeminiVLM
from src.agent.agent import TemporalAgent
from dotenv import load_dotenv

load_dotenv()

def main():

    video_path = "data/raw/elder_video.mp4"

    evaluator = Evaluator()
    # =========================================================
    # INITIALIZE EXISTING VIDEO PIPELINE
    # =========================================================

    reader = VideoReader(video_path)

    tracker = PersonTracker()

    bed = BedRegion(BED_REGION)

    classifier = StateClassifier(bed)

    state_machine = StateMachine()

    # =========================================================
    # INITIALIZE GEMINI AGENT
    # =========================================================

    gemini_model = os.getenv("GEMINI_MODEL")

    vlm = GeminiVLM(
        model=gemini_model
    )

    agent = TemporalAgent(vlm)

    # =========================================================
    # PROCESS VIDEO
    # =========================================================

    for timestamp, frame in reader.frames(
        sample_fps=2
    ):

        # -----------------------------------------------------
        # EXISTING COMPUTER VISION PIPELINE
        # -----------------------------------------------------

        observation = tracker.process(frame)

        if observation is None:

            state = "UNKNOWN"

        else:

            result = classifier.classify(
                observation,
                timestamp
            )

            state = result["state"]

        print(
            f"\n[{timestamp:.2f}s] "
            f"Current state: {state}"
        )

        # =====================================================
        # AGENTIC ANALYSIS
        # =====================================================

        if state in [
            "BESIDE_BED",
            "ON_FLOOR",
            "UNKNOWN"
        ]:

            print(
                "[AGENT] Potentially ambiguous observation."
            )

            # -------------------------------------------------
            # STEP 1
            # Analyze current frame
            # -------------------------------------------------

            current_result = agent.analyze_current(
                frame
            )

            # -------------------------------------------------
            # STEP 2
            # Ask agent what to do next
            # -------------------------------------------------

            action = agent.decide_action(
                current_result
            )

            print(
                f"[AGENT] Decision: {action}"
            )

            # =================================================
            # CURRENT FRAME IS SUFFICIENT
            # =================================================

            if action == "NONE":

                final_result = current_result

                print(
                    "[AGENT] Current frame is sufficient."
                )

            # =================================================
            # CHECK BED REGION
            # =================================================

            elif action == "CHECK_BED_REGION":

                bed_result = agent.check_bed_region(
                    frame
                )

                # ---------------------------------------------
                # Add bed-region evidence to final reasoning
                # ---------------------------------------------

                final_result = agent.final_decision(
                    current=current_result,
                    previous=bed_result,
                    following=None
                )

            # =================================================
            # TEMPORAL ANALYSIS
            # =================================================

            else:

                print(
                    "[AGENT] Temporal context required."
                )

                # -------------------------------------------------
                # Request temporal frames from VideoReader
                # -------------------------------------------------

                context = reader.get_temporal_context(
                    current_timestamp=timestamp,
                    previous_seconds=8,
                    following_seconds=5,
                    sample_count=4
                )

                previous_result = None
                following_result = None

                # =================================================
                # PREVIOUS CONTEXT
                # =================================================

                if action in [
                    "ANALYZE_PREVIOUS",
                    "ANALYZE_PREVIOUS_AND_FOLLOWING"
                ]:

                    print(
                        "[AGENT] Previous context requested."
                    )

                    previous_result = (
                        agent.analyze_previous(
                            context["previous"]
                        )
                    )

                # =================================================
                # FOLLOWING CONTEXT
                # =================================================

                if action in [
                    "ANALYZE_FOLLOWING",
                    "ANALYZE_PREVIOUS_AND_FOLLOWING"
                ]:

                    print(
                        "[AGENT] Following context requested."
                    )

                    following_result = (
                        agent.analyze_following(
                            context["following"]
                        )
                    )

                # =================================================
                # FINAL DECISION
                # =================================================

                final_result = agent.final_decision(
                    current=current_result,
                    previous=previous_result,
                    following=following_result
                )

        # =====================================================
        # NORMAL EXISTING CV STATE
        # =====================================================

        else:

            final_result = {
                "event": state,
                "reason": "Existing state classifier.",
                "confidence": None
            }

        # =====================================================
        # CONVERT AGENT RESULT TO STATE
        # =====================================================

        final_state = convert_agent_event_to_state(
            final_result,
            state
        )

        evaluator.add_result(
            timestamp=timestamp,
            prediction=final_state
        )

        print(
            f"[FINAL] State: {final_state}"
        )

        # =====================================================
        # UPDATE STATE MACHINE
        # =====================================================

        transition = state_machine.update(
            final_state,
            timestamp
        )

        if transition:

            print(
                f"{timestamp:.2f}: "
                f"{transition['previous_state']} "
                f"-> "
                f"{transition['current_state']}"
            )

    evaluator.save_predictions(
        "data/evaluation/predictions.csv"
    )

    # =========================================================
    # FINALIZE STATE MACHINE
    # =========================================================

    state_machine.finalize(
        reader.duration
    )

    # =========================================================
    # PRINT TIMELINE
    # =========================================================

    print("\nTimeline:")

    for segment in state_machine.timeline:

        print(segment)


# =============================================================
# CONVERT GEMINI EVENT TO EXISTING STATE
# =============================================================

def convert_agent_event_to_state(
    result,
    fallback_state
):

    # Gemini failed
    if not isinstance(result, dict):

        return fallback_state

    if "error" in result:

        return fallback_state

    event = result.get(
        "event"
    )

    # ---------------------------------------------------------
    # Agent detected bed exit
    # ---------------------------------------------------------

    if event == "BED_EXIT":

        return "BED_EXIT"

    # ---------------------------------------------------------
    # Agent detected fall
    # ---------------------------------------------------------

    if event == "FALL":

        return "FALL"

    # ---------------------------------------------------------
    # Agent says normal
    # ---------------------------------------------------------

    if event == "NORMAL":

        return "NORMAL"

    # ---------------------------------------------------------
    # Unknown result
    # ---------------------------------------------------------


    return fallback_state

    
# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()