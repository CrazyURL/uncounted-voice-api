"""WhisperX STT service — thin wrapper over stt_processor module."""

from app import stt_processor
from app.stt_processor import load_models, transcribe


class WhisperXService:
    @staticmethod
    def load_models() -> None:
        load_models()

    @staticmethod
    def is_model_loaded() -> bool:
        return stt_processor._model is not None

    @staticmethod
    def transcribe(
        file_path: str,
        task_id: str,
        enable_diarize: bool = False,
        enable_name_masking: bool = False,
        mask_pii: bool = True,
        split_by_speaker: bool = False,
        split_by_utterance: bool = False,
        denoise_enabled: bool | None = None,
        mask_audio_pii: bool = False,
        mask_audio_names: bool = False,
        pii_intervals_only: bool = False,
        reference_embedding: list[float] | None = None,
    ) -> dict:
        return transcribe(
            file_path=file_path,
            task_id=task_id,
            enable_diarize=enable_diarize,
            enable_name_masking=enable_name_masking,
            mask_pii=mask_pii,
            split_by_speaker=split_by_speaker,
            split_by_utterance=split_by_utterance,
            denoise_enabled=denoise_enabled,
            mask_audio_pii=mask_audio_pii,
            mask_audio_names=mask_audio_names,
            pii_intervals_only=pii_intervals_only,
            reference_embedding=reference_embedding,
        )


whisperx_service = WhisperXService()
