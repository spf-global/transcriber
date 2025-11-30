# app/services/api_clients/openai_gpt4o_diarize.py

import os
import logging
import time
from typing import Tuple, Optional, Callable, List, Dict, Any
from openai import OpenAI, OpenAIError, APIError, APIConnectionError, RateLimitError
from app.services import file_service
from app.config import Config

# Define a type hint for the progress callback
ProgressCallback = Optional[Callable[[str, bool], None]]  # Message, IsError

# Maximum chunk duration for gpt-4o-transcribe-diarize (1400 seconds)
DIARIZE_MAX_CHUNK_SECONDS = 1400
DIARIZE_CHUNK_LENGTH_MS = DIARIZE_MAX_CHUNK_SECONDS * 1000


class OpenAIGPT4oDiarizeTranscriptionAPI:
    """
    Integration with OpenAI GPT-4o Transcribe Diarize using synchronous requests.
    Provides speaker diarization (identification of different speakers).
    Handles large file splitting. Reports progress via callback.
    """
    MODEL_NAME = "gpt-4o-transcribe-diarize"
    API_NAME = "OpenAI_GPT4o_Diarize"

    def __init__(self, api_key: str) -> None:
        """Initializes the OpenAI GPT-4o Diarize API client."""
        if not api_key:
            logging.error(f"[{self.API_NAME}] API key is required but not provided.")
            raise ValueError("OpenAI API key is required.")
        self.api_key = api_key
        try:
            self.client = OpenAI(api_key=self.api_key)
            logging.info(f"[{self.API_NAME}] Client initialized successfully for model {self.MODEL_NAME}.")
        except OpenAIError as e:
            logging.error(f"[{self.API_NAME}] Failed to initialize OpenAI client: {e}")
            raise ValueError(f"OpenAI client initialization failed: {e}") from e

    def transcribe(self, audio_file_path: str, language_code: str,
                   progress_callback: ProgressCallback = None,
                   context_prompt: str = "",
                   original_filename: Optional[str] = None
                   ) -> Tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]]]:
        """
        Transcribes the audio file using OpenAI GPT-4o Transcribe Diarize.
        Provides speaker diarization with segments.

        Returns:
            A tuple containing (transcription_text, detected_language, speaker_segments)
            or (None, None, None) on failure.
            - speaker_segments: List of dicts with keys: speaker, text, start, end
        """
        requested_language = language_code
        display_filename = original_filename or os.path.basename(audio_file_path)
        log_prefix = f"[{self.API_NAME}:{display_filename}]"

        transcription_text = None
        speaker_segments = None
        final_language_used = None

        try:
            if not os.path.exists(audio_file_path):
                msg = f"ERROR: Audio file not found at path: {audio_file_path}"
                if progress_callback:
                    progress_callback(msg, True)
                logging.error(f"{log_prefix} {msg}")
                return None, None, None

            file_size = os.path.getsize(audio_file_path)

            # Check if splitting is needed based on file size
            # Note: gpt-4o-transcribe-diarize has a 1400 second limit per chunk
            if file_size > file_service.OPENAI_MAX_FILE_SIZE:
                logging.info(f"{log_prefix} File size ({file_size / 1024 / 1024:.2f}MB) exceeds limit. Starting chunked transcription.")
                return self._split_and_transcribe(audio_file_path, requested_language, progress_callback, display_filename)
            else:
                logging.info(f"{log_prefix} File size ({file_size / 1024 / 1024:.2f}MB) within limit. Processing as single file.")
                abs_path = os.path.abspath(audio_file_path)
                temp_dir = os.path.dirname(abs_path)
                if not file_service.validate_file_path(abs_path, temp_dir):
                    msg = f"ERROR: Audio file path is not allowed or outside expected directory: {abs_path}"
                    if progress_callback:
                        progress_callback(msg, True)
                    logging.error(f"{log_prefix} {msg}")
                    raise ValueError(msg)

                with open(abs_path, "rb") as audio_file:
                    api_params = {
                        "model": self.MODEL_NAME,
                        "file": audio_file,
                        "response_format": "diarized_json",
                    }
                    # Note: gpt-4o-transcribe-diarize does not support prompt parameter

                    log_params = {k: v for k, v in api_params.items() if k != 'file'}
                    lang_note = " (Language: implicit detection by model)"
                    if progress_callback:
                        progress_callback("Language detection: automatic (implicit by model).", False)
                    logging.info(f"{log_prefix} Calling API with parameters: {log_params}{lang_note}")

                    if progress_callback:
                        progress_callback(f"Transcribing with OpenAI {self.MODEL_NAME} (with speaker diarization)...", False)

                    start_time = time.time()
                    logging.info(f"{log_prefix} Calling OpenAI API...")
                    transcript_response = self.client.audio.transcriptions.create(**api_params)
                    duration = time.time() - start_time
                    logging.info(f"{log_prefix} OpenAI API call successful. Duration: {duration:.2f}s")

                    # Parse diarized_json response
                    transcription_text, speaker_segments = self._parse_diarized_response(transcript_response, log_prefix)

            # Language Detection Note
            final_language_used = 'auto'  # Model detects implicitly
            ui_lang_msg = f"OpenAI {self.MODEL_NAME} transcription finished. Language detected implicitly by model."

            if speaker_segments:
                unique_speakers = set(seg.get('speaker', 'Unknown') for seg in speaker_segments)
                ui_lang_msg += f" Identified {len(unique_speakers)} speaker(s)."

            logging.info(f"{log_prefix} {ui_lang_msg}")
            if progress_callback:
                progress_callback(ui_lang_msg, False)
                progress_callback("Transcription with diarization completed.", False)

            return transcription_text, final_language_used, speaker_segments

        except FileNotFoundError as fnf_error:
            error_msg = f"ERROR: Audio file disappeared: {fnf_error}"
            if progress_callback:
                progress_callback(error_msg, True)
            logging.error(f"{log_prefix} {error_msg}")
            return None, None, None
        except RateLimitError as rle:
            error_msg = f"ERROR: OpenAI API rate limit exceeded: {rle}. Please try again later."
            if progress_callback:
                progress_callback(error_msg, True)
            logging.warning(f"{log_prefix} {error_msg}")
            return None, None, None
        except APIConnectionError as ace:
            error_msg = f"ERROR: OpenAI API connection error: {ace}. Check network connectivity."
            if progress_callback:
                progress_callback(error_msg, True)
            logging.error(f"{log_prefix} {error_msg}")
            return None, None, None
        except APIError as apie:
            error_msg = f"ERROR: OpenAI API returned an error: {apie}"
            if progress_callback:
                progress_callback(error_msg, True)
            logging.error(f"{log_prefix} {error_msg}")
            return None, None, None
        except OpenAIError as oae:
            error_msg = f"ERROR: OpenAI SDK Error: {oae}"
            if progress_callback:
                progress_callback(error_msg, True)
            logging.error(f"{log_prefix} {error_msg}")
            return None, None, None
        except ValueError as ve:
            error_msg = f"ERROR: Input Error: {ve}"
            if progress_callback:
                progress_callback(error_msg, True)
            logging.error(f"{log_prefix} {error_msg}")
            return None, None, None
        except Exception as e:
            error_msg = f"ERROR: Unexpected error during {self.API_NAME} transcription: {e}"
            if progress_callback:
                progress_callback(error_msg, True)
            logging.exception(f"{log_prefix} Unexpected error detail:")
            return None, None, None

    def _parse_diarized_response(self, response: Any, log_prefix: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse the diarized_json response from OpenAI.

        Returns:
            Tuple of (full_text, speaker_segments)
        """
        speaker_segments = []
        full_text_parts = []

        try:
            # The response should have a 'segments' attribute with diarization info
            if hasattr(response, 'segments') and response.segments:
                for segment in response.segments:
                    seg_dict = {
                        'speaker': getattr(segment, 'speaker', 'Unknown'),
                        'text': getattr(segment, 'text', '').strip(),
                        'start': getattr(segment, 'start', 0.0),
                        'end': getattr(segment, 'end', 0.0),
                    }
                    speaker_segments.append(seg_dict)
                    if seg_dict['text']:
                        full_text_parts.append(seg_dict['text'])

                logging.info(f"{log_prefix} Parsed {len(speaker_segments)} diarized segments.")
            elif hasattr(response, 'text'):
                # Fallback if no segments but text is available
                full_text_parts.append(response.text)
                logging.warning(f"{log_prefix} No segments in response, using text only.")
            else:
                logging.warning(f"{log_prefix} Unexpected response format: {type(response)}")
                # Try to convert response to string
                full_text_parts.append(str(response))

        except Exception as e:
            logging.error(f"{log_prefix} Error parsing diarized response: {e}")
            # Attempt to get any text from response
            if hasattr(response, 'text'):
                full_text_parts.append(response.text)

        full_text = " ".join(full_text_parts)
        return full_text, speaker_segments

    def _split_and_transcribe(self, audio_file_path: str, language_code: str,
                              progress_callback: ProgressCallback = None,
                              display_filename: Optional[str] = None
                              ) -> Tuple[Optional[str], Optional[str], Optional[List[Dict[str, Any]]]]:
        """Handles splitting large files and transcribing chunks with diarization."""
        log_prefix = f"[{self.API_NAME}:{display_filename or os.path.basename(audio_file_path)}]"

        temp_dir = os.path.dirname(audio_file_path)
        chunk_files = []
        all_segments = []
        all_texts = []

        try:
            # Use file_service to split audio
            chunk_files = file_service.split_audio_file(audio_file_path, temp_dir, progress_callback)
            if not chunk_files:
                raise Exception("Audio splitting failed or resulted in no chunks.")

            total_chunks = len(chunk_files)
            logging.info(f"{log_prefix} Starting transcription of {total_chunks} chunks...")

            # Track time offset for segment timestamps
            time_offset = 0.0

            for idx, chunk_path in enumerate(chunk_files):
                chunk_num = idx + 1
                chunk_log_prefix = f"{log_prefix}:Chunk{chunk_num}"

                chunk_text, chunk_segments = self._transcribe_single_chunk_with_retry(
                    chunk_path, chunk_num, total_chunks,
                    progress_callback, chunk_log_prefix
                )

                if chunk_text is None:
                    raise Exception(f"Failed to transcribe chunk {chunk_num}. Aborting.")

                all_texts.append(chunk_text)

                # Adjust segment timestamps with offset and add to all_segments
                if chunk_segments:
                    chunk_duration = 0.0
                    for seg in chunk_segments:
                        adjusted_seg = seg.copy()
                        adjusted_seg['start'] = seg['start'] + time_offset
                        adjusted_seg['end'] = seg['end'] + time_offset
                        all_segments.append(adjusted_seg)
                        chunk_duration = max(chunk_duration, seg['end'])
                    time_offset += chunk_duration

                logging.info(f"{chunk_log_prefix} Transcription successful.")

            full_transcription = " ".join(filter(None, all_texts))
            final_language_used = 'auto'

            unique_speakers = set(seg.get('speaker', 'Unknown') for seg in all_segments)
            ui_lang_msg = f"Aggregated {total_chunks} chunk transcriptions. Identified {len(unique_speakers)} speaker(s)."

            logging.info(f"{log_prefix} {ui_lang_msg}")
            if progress_callback:
                progress_callback(ui_lang_msg, False)
                progress_callback("Transcription with diarization completed.", False)

            return full_transcription, final_language_used, all_segments

        except Exception as e:
            error_msg = f"ERROR: Error during split and transcribe process: {e}"
            if progress_callback:
                progress_callback(error_msg, True)
            logging.exception(f"{log_prefix} Error detail in _split_and_transcribe:")
            return None, None, None
        finally:
            if chunk_files:
                if progress_callback:
                    progress_callback("Cleaning up temporary chunk files...", False)
                removed_count = file_service.remove_files(chunk_files)
                logging.info(f"{log_prefix} Cleaned up {removed_count} temporary chunk file(s).")
                if progress_callback:
                    progress_callback(f"Cleaned up {removed_count} temporary chunk file(s).", False)

    def _transcribe_single_chunk_with_retry(self, chunk_path: str, idx: int, total_chunks: int,
                                            progress_callback: ProgressCallback = None,
                                            log_prefix: str = "", max_retries: int = 3
                                            ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
        """
        Transcribes a single chunk with retry logic using GPT-4o Diarize.

        Returns: Tuple of (transcription_text, speaker_segments) or (None, None) on failure.
        """
        last_error = None
        chunk_base_name = os.path.basename(chunk_path)
        effective_log_prefix = log_prefix or f"[{self.API_NAME}:Chunk{idx}]"

        for attempt in range(max_retries):
            if progress_callback:
                progress_callback(f"Transcribing chunk {idx}/{total_chunks} (with diarization)", False)

            try:
                abs_chunk_path = os.path.abspath(chunk_path)
                temp_dir = os.path.dirname(abs_chunk_path)
                if not file_service.validate_file_path(abs_chunk_path, temp_dir):
                    msg = f"Chunk file path is not allowed: {abs_chunk_path}"
                    logging.error(f"{effective_log_prefix} {msg}")
                    raise ValueError(msg)

                with open(abs_chunk_path, "rb") as audio_file:
                    api_params = {
                        "model": self.MODEL_NAME,
                        "file": audio_file,
                        "response_format": "diarized_json",
                    }

                    log_params = {k: v for k, v in api_params.items() if k != 'file'}
                    logging.info(f"{effective_log_prefix} Attempt {attempt + 1}: Calling API with parameters: {log_params}")

                    start_time = time.time()
                    logging.info(f"{effective_log_prefix} Attempt {attempt + 1}: Calling OpenAI API...")
                    response = self.client.audio.transcriptions.create(**api_params)
                    duration = time.time() - start_time
                    logging.info(f"{effective_log_prefix} Attempt {attempt + 1}: API call successful. Duration: {duration:.2f}s")

                    text, segments = self._parse_diarized_response(response, effective_log_prefix)
                    return text.strip() if text else "", segments

            except RateLimitError as rle:
                last_error = rle
                wait_time = 2 ** attempt
                error_detail = f"Rate limit hit on chunk {idx}, attempt {attempt + 1}. Retrying in {wait_time}s..."
                if progress_callback:
                    progress_callback(error_detail, False)
                logging.warning(f"{effective_log_prefix} Rate limit hit, attempt {attempt + 1}. Retrying in {wait_time}s... ({rle})")
                time.sleep(wait_time)
            except (APIConnectionError, APIError) as e:
                last_error = e
                wait_time = 2 ** attempt
                error_detail = f"API error on chunk {idx} (Attempt {attempt + 1}). Retrying in {wait_time}s..."
                if progress_callback:
                    progress_callback(error_detail, False)
                logging.error(f"{effective_log_prefix} API error on chunk {idx}, attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            except OpenAIError as oae:
                last_error = oae
                error_detail = f"ERROR: OpenAI SDK error on chunk {idx}: {oae}"
                if progress_callback:
                    progress_callback(error_detail, True)
                logging.error(f"{effective_log_prefix} OpenAI SDK error on chunk {idx}, attempt {attempt + 1}: {oae}")
                break
            except ValueError as ve:
                last_error = ve
                error_detail = f"ERROR: Input error processing chunk {idx}: {ve}"
                if progress_callback:
                    progress_callback(error_detail, True)
                logging.error(f"{effective_log_prefix} {error_detail}")
                break
            except FileNotFoundError as fnf_error:
                last_error = fnf_error
                error_detail = f"ERROR: Chunk file not found: {chunk_base_name}. Error: {fnf_error}"
                if progress_callback:
                    progress_callback(error_detail, True)
                logging.error(f"{effective_log_prefix} Chunk file not found on attempt {attempt + 1}: {chunk_base_name}. Error: {fnf_error}")
                break
            except Exception as e:
                last_error = e
                error_detail = f"ERROR: Unexpected error transcribing chunk {idx}: {e}"
                if progress_callback:
                    progress_callback(error_detail, True)
                logging.exception(f"{effective_log_prefix} Unexpected error detail on attempt {attempt + 1}:")
                break

        final_error_msg = f"ERROR: Chunk {idx} ('{chunk_base_name}') failed after {max_retries} attempts. Last error: {last_error}"
        if progress_callback:
            progress_callback(final_error_msg, True)
        logging.error(f"{effective_log_prefix} Chunk {idx} failed after {max_retries} attempts. Last error: {last_error}")
        return None, None
