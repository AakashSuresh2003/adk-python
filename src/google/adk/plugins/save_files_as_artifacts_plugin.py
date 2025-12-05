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

import copy
import logging
import os
import tempfile
from typing import Optional
import urllib.parse

from google.genai import Client
from google.genai import types

from ..agents.invocation_context import InvocationContext
from .base_plugin import BasePlugin

logger = logging.getLogger('google_adk.' + __name__)

# Schemes supported by our current LLM connectors. Vertex exposes `gs://` while
# hosted endpoints use HTTPS. Expand this list when BaseLlm surfaces provider
# capabilities.
_MODEL_ACCESSIBLE_URI_SCHEMES = {'gs', 'https', 'http'}

# Maximum file size for inline_data (20MB as per Gemini API documentation)
# Maximum file size for Files API (2GB as per Gemini API documentation)
# https://ai.google.dev/gemini-api/docs/files
_MAX_INLINE_DATA_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
_MAX_FILES_API_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


class SaveFilesAsArtifactsPlugin(BasePlugin):
  """A plugin that saves files embedded in user messages as artifacts.

  This is useful to allow users to upload files in the chat experience and have
  those files available to the agent within the current session.

  We use Blob.display_name to determine the file name. By default, artifacts are
  session-scoped. For cross-session persistence, prefix the filename with
  "user:".
  Artifacts with the same name will be overwritten. A placeholder with the
  artifact name will be put in place of the embedded file in the user message
  so the model knows where to find the file. You may want to add load_artifacts
  tool to the agent, or load the artifacts in your own tool to use the files.
  """

  def __init__(self, name: str = 'save_files_as_artifacts_plugin'):
    """Initialize the save files as artifacts plugin.

    Args:
      name: The name of the plugin instance.
    """
    super().__init__(name)

  async def on_user_message_callback(
      self,
      *,
      invocation_context: InvocationContext,
      user_message: types.Content,
  ) -> Optional[types.Content]:
    """Process user message and save any attached files as artifacts."""
    if not invocation_context.artifact_service:
      logger.warning(
          'Artifact service is not set. SaveFilesAsArtifactsPlugin'
          ' will not be enabled.'
      )
      return user_message

    if not user_message.parts:
      return None

    new_parts = []
    modified = False

    for i, part in enumerate(user_message.parts):
      if part.inline_data is None:
        new_parts.append(part)
        continue

      try:
        # Check file size before processing
        inline_data = part.inline_data
        file_size = len(inline_data.data) if inline_data.data else 0

        # Use display_name if available, otherwise generate a filename
        file_name = inline_data.display_name
        if not file_name:
          file_name = f'artifact_{invocation_context.invocation_id}_{i}'
          logger.info(
              f'No display_name found, using generated filename: {file_name}'
          )

        # Store original filename for display to user/placeholder
        display_name = file_name

        # Check if file exceeds Files API limit (2GB)
        if file_size > _MAX_FILES_API_SIZE_BYTES:
          file_size_gb = file_size / (1024 * 1024 * 1024)
          error_message = (
              f'File {display_name} ({file_size_gb:.2f} GB) exceeds the'
              ' maximum supported size of 2GB. Please upload a smaller file.'
          )
          logger.warning(error_message)
          new_parts.append(types.Part(text=f'[Upload Error: {error_message}]'))
          modified = True
          continue

        # For files larger than 20MB, use Files API
        if file_size > _MAX_INLINE_DATA_SIZE_BYTES:
          file_size_mb = file_size / (1024 * 1024)
          logger.info(
              f'File {display_name} ({file_size_mb:.2f} MB) exceeds'
              ' inline_data limit. Uploading via Files API...'
          )

          # Upload to Files API and convert to file_data
          try:
            file_part = await self._upload_to_files_api(
                inline_data=inline_data,
                file_name=file_name,
            )

            # Save the file_data artifact
            version = await invocation_context.artifact_service.save_artifact(
                app_name=invocation_context.app_name,
                user_id=invocation_context.user_id,
                session_id=invocation_context.session.id,
                filename=file_name,
                artifact=copy.copy(file_part),
            )

            placeholder_part = types.Part(
                text=f'[Uploaded Artifact: "{display_name}"]'
            )
            new_parts.append(placeholder_part)
            new_parts.append(file_part)
            modified = True
            logger.info(f'Successfully uploaded {display_name} via Files API')
          except Exception as e:
            error_message = (
                f'Failed to upload file {display_name} ({file_size_mb:.2f} MB)'
                f' via Files API: {str(e)}'
            )
            logger.error(error_message)
            new_parts.append(
                types.Part(text=f'[Upload Error: {error_message}]')
            )
            modified = True
          continue

        # For files <= 20MB, use inline_data (existing behavior)
        # Create a copy to stop mutation of the saved artifact if the original part is modified
        version = await invocation_context.artifact_service.save_artifact(
            app_name=invocation_context.app_name,
            user_id=invocation_context.user_id,
            session_id=invocation_context.session.id,
            filename=file_name,
            artifact=copy.copy(part),
        )

        placeholder_part = types.Part(
            text=f'[Uploaded Artifact: "{display_name}"]'
        )
        new_parts.append(placeholder_part)

        file_part = await self._build_file_reference_part(
            invocation_context=invocation_context,
            filename=file_name,
            version=version,
            mime_type=inline_data.mime_type,
            display_name=display_name,
        )
        if file_part:
          new_parts.append(file_part)

        modified = True
        logger.info(f'Successfully saved artifact: {file_name}')

      except Exception as e:
        logger.error(f'Failed to save artifact for part {i}: {e}')
        # Keep the original part if saving fails
        new_parts.append(part)
        continue

    if modified:
      return types.Content(role=user_message.role, parts=new_parts)
    else:
      return None

  async def _upload_to_files_api(
      self,
      *,
      inline_data: types.Blob,
      file_name: str,
  ) -> types.Part:

    # Create a temporary file with the inline data
    temp_file_path = None
    try:
      # Determine file extension from display_name or mime_type
      file_extension = ''
      if inline_data.display_name and '.' in inline_data.display_name:
        file_extension = os.path.splitext(inline_data.display_name)[1]
      elif inline_data.mime_type:
        # Simple mime type to extension mapping
        mime_to_ext = {
            'application/pdf': '.pdf',
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/gif': '.gif',
            'text/plain': '.txt',
            'application/json': '.json',
        }
        file_extension = mime_to_ext.get(inline_data.mime_type, '')

      # Create temporary file
      with tempfile.NamedTemporaryFile(
          mode='wb',
          suffix=file_extension,
          delete=False,
      ) as temp_file:
        temp_file.write(inline_data.data)
        temp_file_path = temp_file.name

      # Upload to Files API
      client = Client()
      uploaded_file = client.files.upload(file=temp_file_path)

      # Create file_data Part
      return types.Part(
          file_data=types.FileData(
              file_uri=uploaded_file.uri,
              mime_type=inline_data.mime_type,
              display_name=inline_data.display_name or file_name,
          )
      )
    finally:
      # Clean up temporary file
      if temp_file_path and os.path.exists(temp_file_path):
        try:
          os.unlink(temp_file_path)
        except Exception as cleanup_error:
          logger.warning(
              f'Failed to cleanup temp file {temp_file_path}: {cleanup_error}'
          )

  async def _build_file_reference_part(
      self,
      *,
      invocation_context: InvocationContext,
      filename: str,
      version: int,
      mime_type: Optional[str],
      display_name: str,
  ) -> Optional[types.Part]:
    """Constructs a file reference part if the artifact URI is model-accessible."""

    artifact_service = invocation_context.artifact_service
    if not artifact_service:
      return None

    try:
      artifact_version = await artifact_service.get_artifact_version(
          app_name=invocation_context.app_name,
          user_id=invocation_context.user_id,
          session_id=invocation_context.session.id,
          filename=filename,
          version=version,
      )
    except Exception as exc:  # pylint: disable=broad-except
      logger.warning(
          'Failed to resolve artifact version for %s: %s', filename, exc
      )
      return None

    if (
        not artifact_version
        or not artifact_version.canonical_uri
        or not _is_model_accessible_uri(artifact_version.canonical_uri)
    ):
      return None

    file_data = types.FileData(
        file_uri=artifact_version.canonical_uri,
        mime_type=mime_type or artifact_version.mime_type,
        display_name=display_name,
    )
    return types.Part(file_data=file_data)


def _is_model_accessible_uri(uri: str) -> bool:
  try:
    parsed = urllib.parse.urlparse(uri)
  except ValueError:
    return False

  if not parsed.scheme:
    return False

  return parsed.scheme.lower() in _MODEL_ACCESSIBLE_URI_SCHEMES
