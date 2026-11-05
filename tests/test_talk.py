from unittest.mock import patch

from agentops.talk import (
    TALK_TURN_ID_HEADER,
    echo_talk_turn_id,
    get_talk_turn_id,
    talk_turn_metadata,
)


class TestGetTalkTurnId:
    def test_extracts_case_insensitively(self):
        assert get_talk_turn_id({"X-AOS-Talk-Turn-Id": "turn-123"}) == "turn-123"

    def test_matches_the_canonical_header_name(self):
        assert get_talk_turn_id({TALK_TURN_ID_HEADER: "turn-456"}) == "turn-456"

    def test_passes_through_a_raw_turn_id_string(self):
        assert get_talk_turn_id("turn-789") == "turn-789"

    def test_returns_none_when_header_absent(self):
        assert get_talk_turn_id({"content-type": "application/json"}) is None

    def test_returns_none_for_empty_dict_or_none(self):
        assert get_talk_turn_id({}) is None
        assert get_talk_turn_id(None) is None

    def test_returns_none_for_blank_value(self):
        assert get_talk_turn_id({TALK_TURN_ID_HEADER: ""}) is None

    def test_returns_none_for_non_mapping_non_string_input(self):
        assert get_talk_turn_id(12345) is None


class TestTalkTurnMetadata:
    def test_wraps_the_turn_id_under_the_expected_key(self):
        assert talk_turn_metadata({TALK_TURN_ID_HEADER: "turn-abc"}) == {
            "talkTurnId": "turn-abc"
        }

    def test_empty_dict_when_absent_not_none(self):
        # Must compose safely with `{**metadata, **talk_turn_metadata(...)}`
        # without the caller having to guard against None first.
        assert talk_turn_metadata({}) == {}
        assert talk_turn_metadata(None) == {}


class TestEchoTalkTurnId:
    def test_stamps_metadata_on_the_active_trace_when_header_present(self):
        with patch("langfuse.decorators.langfuse_context") as mock_ctx:
            result = echo_talk_turn_id({TALK_TURN_ID_HEADER: "turn-xyz"})

        assert result == "turn-xyz"
        mock_ctx.update_current_trace.assert_called_once_with(
            metadata={"talkTurnId": "turn-xyz"}
        )

    def test_accepts_a_raw_turn_id_string(self):
        with patch("langfuse.decorators.langfuse_context") as mock_ctx:
            result = echo_talk_turn_id("turn-raw")

        assert result == "turn-raw"
        mock_ctx.update_current_trace.assert_called_once_with(
            metadata={"talkTurnId": "turn-raw"}
        )

    def test_no_ops_when_header_missing(self):
        with patch("langfuse.decorators.langfuse_context") as mock_ctx:
            result = echo_talk_turn_id({})

        assert result is None
        mock_ctx.update_current_trace.assert_not_called()

    def test_no_ops_when_given_none(self):
        with patch("langfuse.decorators.langfuse_context") as mock_ctx:
            result = echo_talk_turn_id(None)

        assert result is None
        mock_ctx.update_current_trace.assert_not_called()
