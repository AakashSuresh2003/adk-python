# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

from google.adk.agents.invocation_context import InvocationContext
from google.adk.artifacts.base_artifact_service import ArtifactVersion
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.genai import Client
from google.genai import types
import pytest


class TestSaveFilesAsArtifactsPlugin:
  """Test suite for SaveFilesAsArtifactsPlugin."""

  def setup_method(self):
    """Set up test fixtures."""
    self.plugin = SaveFilesAsArtifactsPlugin()

    # Mock invocation context
    self.mock_context = Mock(spec=InvocationContext)
    self.mock_context.app_name = "test_app"
    self.mock_context.user_id = "test_user"
    self.mock_context.invocation_id = "test_invocation_123"
    self.mock_context.session = Mock()
    self.mock_context.session.id = "test_session"

    artifact_service = Mock()
    artifact_service.save_artifact = AsyncMock(return_value=0)

    async def _mock_get_artifact_version(**kwargs):
      filename = kwargs.get("filename", "unknown_file")
      version = kwargs.get("version", 0)
      return ArtifactVersion(
          version=version,
          canonical_uri=f"gs://mock-bucket/{filename}/versions/{version}",
          mime_type="application/pdf",
      )

    artifact_service.get_artifact_version = AsyncMock(
        side_effect=_mock_get_artifact_version
    )
    self.mock_context.artifact_service = artifact_service

  @pytest.mark.asyncio
  async def test_save_files_with_display_name(self):
    """Test saving files when inline_data has display_name."""
    inline_data = types.Blob(
        display_name="test_document.pdf",
        data=b"test data",
        mime_type="application/pdf",
    )

    original_part = types.Part(inline_data=inline_data)
    user_message = types.Content(parts=[original_part])

    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    self.mock_context.artifact_service.save_artifact.assert_called_once_with(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename="test_document.pdf",
        artifact=original_part,
    )

    assert result
    assert len(result.parts) == 2
    assert result.parts[0].text == '[Uploaded Artifact: "test_document.pdf"]'
    assert result.parts[1].file_data
    assert (
        result.parts[1].file_data.file_uri
        == "gs://mock-bucket/test_document.pdf/versions/0"
    )
    assert result.parts[1].file_data.display_name == "test_document.pdf"
    assert result.parts[1].file_data.mime_type == "application/pdf"

  @pytest.mark.asyncio
  async def test_save_files_without_display_name(self):
    """Test saving files when inline_data has no display_name."""
    inline_data = types.Blob(
        display_name=None, data=b"test data", mime_type="application/pdf"
    )

    original_part = types.Part(inline_data=inline_data)
    user_message = types.Content(parts=[original_part])

    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    expected_filename = "artifact_test_invocation_123_0"
    self.mock_context.artifact_service.save_artifact.assert_called_once_with(
        app_name="test_app",
        user_id="test_user",
        session_id="test_session",
        filename=expected_filename,
        artifact=original_part,
    )

    assert result
    assert len(result.parts) == 2
    assert result.parts[0].text == f'[Uploaded Artifact: "{expected_filename}"]'
    assert result.parts[1].file_data
    assert (
        result.parts[1].file_data.file_uri
        == "gs://mock-bucket/artifact_test_invocation_123_0/versions/0"
    )
    assert result.parts[1].file_data.display_name == expected_filename

  @pytest.mark.asyncio
  async def test_multiple_files_in_message(self):
    """Test handling multiple files in a single message."""
    inline_data1 = types.Blob(
        display_name="file1.txt", data=b"file1 content", mime_type="text/plain"
    )
    inline_data2 = types.Blob(
        display_name="file2.jpg", data=b"file2 content", mime_type="image/jpeg"
    )

    user_message = types.Content(
        parts=[
            types.Part(inline_data=inline_data1),
            types.Part(text="Some text between files"),
            types.Part(inline_data=inline_data2),
        ]
    )

    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    assert self.mock_context.artifact_service.save_artifact.call_count == 2
    first_call = (
        self.mock_context.artifact_service.save_artifact.call_args_list[0]
    )
    second_call = (
        self.mock_context.artifact_service.save_artifact.call_args_list[1]
    )
    assert first_call[1]["filename"] == "file1.txt"
    assert second_call[1]["filename"] == "file2.jpg"

    assert result
    assert len(result.parts) == 5
    assert result.parts[0].text == '[Uploaded Artifact: "file1.txt"]'
    assert result.parts[1].file_data
    assert (
        result.parts[1].file_data.file_uri
        == "gs://mock-bucket/file1.txt/versions/0"
    )
    assert result.parts[1].file_data.display_name == "file1.txt"
    assert result.parts[2].text == "Some text between files"
    assert result.parts[3].text == '[Uploaded Artifact: "file2.jpg"]'
    assert result.parts[4].file_data
    assert (
        result.parts[4].file_data.file_uri
        == "gs://mock-bucket/file2.jpg/versions/0"
    )
    assert result.parts[4].file_data.display_name == "file2.jpg"

  @pytest.mark.asyncio
  async def test_no_artifact_service(self):
    """Test behavior when artifact service is not available."""
    self.mock_context.artifact_service = None

    inline_data = types.Blob(
        display_name="test.pdf", data=b"test data", mime_type="application/pdf"
    )
    user_message = types.Content(parts=[types.Part(inline_data=inline_data)])

    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    assert result == user_message
    assert result.parts[0].inline_data == inline_data

  @pytest.mark.asyncio
  async def test_no_parts_in_message(self):
    """Test behavior when message has no parts."""
    user_message = types.Content(parts=[])

    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    assert result is None
    self.mock_context.artifact_service.save_artifact.assert_not_called()

  @pytest.mark.asyncio
  async def test_parts_without_inline_data(self):
    """Test behavior with parts that don't have inline_data."""
    user_message = types.Content(
        parts=[types.Part(text="Hello world"), types.Part(text="No files here")]
    )

    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    assert result is None
    self.mock_context.artifact_service.save_artifact.assert_not_called()

  @pytest.mark.asyncio
  async def test_save_artifact_failure(self):
    """Test behavior when saving artifact fails."""
    self.mock_context.artifact_service.save_artifact.side_effect = Exception(
        "Storage error"
    )

    inline_data = types.Blob(
        display_name="test.pdf", data=b"test data", mime_type="application/pdf"
    )
    user_message = types.Content(parts=[types.Part(inline_data=inline_data)])

    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    assert result is None

  @pytest.mark.asyncio
  async def test_mixed_success_and_failure(self):
    """Test behavior when some files save successfully and others fail."""
    save_calls = 0

    async def _save_side_effect(*_args, **_kwargs):
      nonlocal save_calls
      save_calls += 1
      if save_calls == 2:
        raise Exception("Storage error on second file")
      return 0

    self.mock_context.artifact_service.save_artifact.side_effect = (
        _save_side_effect
    )

    inline_data1 = types.Blob(
        display_name="success.pdf",
        data=b"success data",
        mime_type="application/pdf",
    )
    inline_data2 = types.Blob(
        display_name="failure.pdf",
        data=b"failure data",
        mime_type="application/pdf",
    )

    original_part2 = types.Part(inline_data=inline_data2)
    user_message = types.Content(
        parts=[types.Part(inline_data=inline_data1), original_part2]
    )

    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    assert result
    assert len(result.parts) == 3
    assert result.parts[0].text == '[Uploaded Artifact: "success.pdf"]'
    assert result.parts[1].file_data
    assert result.parts[2] == original_part2
    assert result.parts[2].inline_data == inline_data2

  @pytest.mark.asyncio
  async def test_placeholder_text_format(self):
    """Test that placeholder text is formatted correctly."""
    inline_data = types.Blob(
        display_name="test file with spaces.docx",
        data=b"document data",
        mime_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
    )

    user_message = types.Content(parts=[types.Part(inline_data=inline_data)])

    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    expected_text = '[Uploaded Artifact: "test file with spaces.docx"]'
    assert result.parts[0].text == expected_text
    assert result.parts[1].file_data

  def test_plugin_name_default(self):
    """Test that plugin has correct default name."""
    plugin = SaveFilesAsArtifactsPlugin()
    assert plugin.name == "save_files_as_artifacts_plugin"

  @pytest.mark.asyncio
  async def test_file_size_exceeds_limit(self):
    """Test that files exceeding 20MB limit are uploaded via Files API."""
    # Create a file larger than 20MB (20 * 1024 * 1024 bytes)
    large_file_data = b"x" * (21 * 1024 * 1024)  # 21 MB
    inline_data = types.Blob(
        display_name="large_file.pdf",
        data=large_file_data,
        mime_type="application/pdf",
    )

    user_message = types.Content(parts=[types.Part(inline_data=inline_data)])

    # Mock the Files API upload
    with patch(
        "google.adk.plugins.save_files_as_artifacts_plugin.Client"
    ) as mock_client:
      # Mock uploaded file response
      mock_uploaded_file = MagicMock()
      mock_uploaded_file.uri = (
          "https://generativelanguage.googleapis.com/v1beta/files/test-file-id"
      )
      mock_client.return_value.files.upload.return_value = mock_uploaded_file

      result = await self.plugin.on_user_message_callback(
          invocation_context=self.mock_context, user_message=user_message
      )

      # Should upload via Files API
      mock_client.return_value.files.upload.assert_called_once()

      # Should save the artifact with file_data
      self.mock_context.artifact_service.save_artifact.assert_called_once()

      # Should return success message with placeholder and file_data
      assert result is not None
      assert len(result.parts) == 2
      assert '[Uploaded Artifact: "large_file.pdf"]' in result.parts[0].text
      assert result.parts[1].file_data is not None
      assert result.parts[1].file_data.file_uri == mock_uploaded_file.uri

  @pytest.mark.asyncio
  async def test_file_size_at_limit(self):
    """Test that files exactly at 20MB limit are processed successfully."""
    # Create a file exactly 20MB (20 * 1024 * 1024 bytes)
    file_data = b"x" * (20 * 1024 * 1024)  # Exactly 20 MB
    inline_data = types.Blob(
        display_name="max_size_file.pdf",
        data=file_data,
        mime_type="application/pdf",
    )

    user_message = types.Content(parts=[types.Part(inline_data=inline_data)])

    result = await self.plugin.on_user_message_callback(
        invocation_context=self.mock_context, user_message=user_message
    )

    # Should save the artifact since it's at the limit
    self.mock_context.artifact_service.save_artifact.assert_called_once()
    assert result is not None
    assert len(result.parts) == 2
    assert result.parts[0].text == '[Uploaded Artifact: "max_size_file.pdf"]'
    assert result.parts[1].file_data is not None

  @pytest.mark.asyncio
  async def test_file_size_just_over_limit(self):
    """Test that files just over 20MB limit are uploaded via Files API."""
    # Create a file just over 20MB
    large_file_data = b"x" * (20 * 1024 * 1024 + 1)  # 20 MB + 1 byte
    inline_data = types.Blob(
        display_name="slightly_too_large.pdf",
        data=large_file_data,
        mime_type="application/pdf",
    )

    user_message = types.Content(parts=[types.Part(inline_data=inline_data)])

    # Mock the Files API upload
    with patch.object(Client, "files") as mock_files:
      mock_uploaded_file = MagicMock()
      mock_uploaded_file.uri = (
          "https://generativelanguage.googleapis.com/v1beta/files/test-file-id"
      )
      mock_files.upload.return_value = mock_uploaded_file

      result = await self.plugin.on_user_message_callback(
          invocation_context=self.mock_context, user_message=user_message
      )

      # Should upload via Files API
      mock_files.upload.assert_called_once()
      self.mock_context.artifact_service.save_artifact.assert_called_once()

      # Should return success
      assert result is not None
      assert len(result.parts) == 2
      assert "[Uploaded Artifact:" in result.parts[0].text

  @pytest.mark.asyncio
  async def test_mixed_file_sizes(self):
    """Test processing multiple files with mixed sizes."""
    # Small file (should succeed with inline_data)
    small_file_data = b"x" * (5 * 1024 * 1024)  # 5 MB
    small_inline_data = types.Blob(
        display_name="small.pdf",
        data=small_file_data,
        mime_type="application/pdf",
    )

    # Large file (should succeed with Files API)
    large_file_data = b"x" * (25 * 1024 * 1024)  # 25 MB
    large_inline_data = types.Blob(
        display_name="large.pdf",
        data=large_file_data,
        mime_type="application/pdf",
    )

    user_message = types.Content(
        parts=[
            types.Part(inline_data=small_inline_data),
            types.Part(inline_data=large_inline_data),
        ]
    )

    # Mock the Files API upload for large file
    with patch.object(Client, "files") as mock_files:
      mock_uploaded_file = MagicMock()
      mock_uploaded_file.uri = (
          "https://generativelanguage.googleapis.com/v1beta/files/test-file-id"
      )
      mock_files.upload.return_value = mock_uploaded_file

      result = await self.plugin.on_user_message_callback(
          invocation_context=self.mock_context, user_message=user_message
      )

      # Should save both files
      assert self.mock_context.artifact_service.save_artifact.call_count == 2

      # Should upload large file via Files API
      mock_files.upload.assert_called_once()

      # Should return success messages for both files
      assert result is not None
      assert (
          len(result.parts) == 4
      )  # [small placeholder, small file_data, large placeholder, large file_data]
      assert '[Uploaded Artifact: "small.pdf"]' in result.parts[0].text
      assert result.parts[1].file_data is not None
      assert '[Uploaded Artifact: "large.pdf"]' in result.parts[2].text
      assert result.parts[3].file_data is not None

  @pytest.mark.asyncio
  async def test_files_api_upload_failure(self):
    """Test that Files API upload failures are handled gracefully."""
    # Create a file larger than 20MB
    large_file_data = b"x" * (30 * 1024 * 1024)  # 30 MB
    inline_data = types.Blob(
        display_name="huge_file.pdf",
        data=large_file_data,
        mime_type="application/pdf",
    )

    user_message = types.Content(parts=[types.Part(inline_data=inline_data)])

    # Mock the Files API to raise an exception
    with patch.object(Client, "files") as mock_files:
      mock_files.upload.side_effect = Exception("API quota exceeded")

      result = await self.plugin.on_user_message_callback(
          invocation_context=self.mock_context, user_message=user_message
      )

      # Should attempt Files API upload
      mock_files.upload.assert_called_once()

      # Should not save artifact on upload failure
      self.mock_context.artifact_service.save_artifact.assert_not_called()

      # Should return error message
      assert result is not None
      assert len(result.parts) == 1
      assert "[Upload Error:" in result.parts[0].text
      assert "huge_file.pdf" in result.parts[0].text
      assert "API quota exceeded" in result.parts[0].text

  @pytest.mark.asyncio
  async def test_file_exceeds_files_api_limit(self):
    """Test that files exceeding 2GB limit are rejected with clear error."""
    # Use a small file for the test
    large_data = b"x" * 1000
    inline_data = types.Blob(
        display_name="huge_video.mp4",
        data=large_data,
        mime_type="video/mp4",
    )
    user_message = types.Content(parts=[types.Part(inline_data=inline_data)])

    # Patch the size limit to be smaller than the file data
    with patch(
        "google.adk.plugins.save_files_as_artifacts_plugin._MAX_FILES_API_SIZE_BYTES",
        500,
    ):
      result = await self.plugin.on_user_message_callback(
          invocation_context=self.mock_context, user_message=user_message
      )

    # Should not attempt any upload
    self.mock_context.artifact_service.save_artifact.assert_not_called()

    # Should return error message about the limit
    assert result is not None
    assert len(result.parts) == 1
    assert "[Upload Error:" in result.parts[0].text
    assert "huge_video.mp4" in result.parts[0].text
    # Note: This assertion will depend on fixing the hardcoded "2GB" in the error message.
    assert "exceeds the maximum supported size" in result.parts[0].text
