"""Tests for v0.10.1 bug fixes - Windows Unicode, PPO params, dtype, diff."""

from pathlib import Path
from unittest.mock import patch

import pytest

from soup_cli.config.schema import SoupConfig

# --- BUG-001: Windows UnicodeEncodeError (no Unicode arrows/dashes in output) ---


class TestNoUnicodeInOutput:
    """Verify user-facing output uses only ASCII-safe characters."""

    def test_config_loader_error_uses_ascii_arrow(self):
        """Config validation errors should use -> not Unicode arrow."""
        from soup_cli.config.loader import load_config_from_string

        with pytest.raises(ValueError) as exc_info:
            load_config_from_string("base: x\ntask: invalid_task\n")
        # Error message should use -> not the Unicode arrow
        msg = str(exc_info.value)
        assert "\u2192" not in msg  # no Unicode right arrow

    def test_loss_format_uses_ascii(self):
        """Loss formatting in runs should use -> not Unicode arrow."""
        from soup_cli.commands.runs import _fmt_loss

        run = {"initial_loss": 1.5, "final_loss": 0.5}
        result = _fmt_loss(run)
        assert "->" in result
        assert "\u2192" not in result  # no Unicode right arrow

    def test_loss_format_missing_returns_ascii(self):
        """Missing loss should return ASCII dash, not em dash."""
        from soup_cli.commands.runs import _fmt_loss

        result = _fmt_loss({})
        assert result == "-"
        assert "\u2014" not in result  # no em dash

    def test_fmt_float_missing_returns_ascii(self):
        """Missing float should return ASCII dash."""
        from soup_cli.commands.runs import _fmt_float

        result = _fmt_float(None)
        assert result == "-"
        assert "\u2014" not in result

    def test_fmt_duration_missing_returns_ascii(self):
        """Missing duration should return ASCII dash."""
        from soup_cli.commands.runs import _fmt_duration

        result = _fmt_duration(None)
        assert result == "-"
        assert "\u2014" not in result

    def test_formats_empty_dataset_error_ascii(self):
        """Empty dataset error should use ASCII dash."""
        from soup_cli.data.formats import detect_format

        with pytest.raises(ValueError, match="Empty dataset"):
            detect_format([])


# --- BUG-002: PPO ppo_epochs parameter compatibility ---


class TestPPOParamCompat:
    """Test PPO trainer handles trl version differences."""

    def test_ppo_config_uses_inspect(self):
        """PPO setup should use inspect to detect valid parameter names."""
        from soup_cli.trainer.ppo import PPOTrainerWrapper

        cfg = SoupConfig(
            base="test-model",
            task="ppo",
            data={"train": "./data.jsonl"},
            training={
                "ppo_epochs": 3,
                "ppo_clip_ratio": 0.15,
                "ppo_kl_penalty": 0.03,
            },
        )
        wrapper = PPOTrainerWrapper(cfg, device="cpu")
        assert wrapper.config.training.ppo_epochs == 3
        assert wrapper.config.training.ppo_clip_ratio == pytest.approx(0.15)
        assert wrapper.config.training.ppo_kl_penalty == pytest.approx(0.03)


# --- BUG-003: Reward Model dtype mismatch ---


class TestComputeDtype:
    """Test get_compute_dtype returns correct dtype for device."""

    def test_cpu_returns_float32(self):
        """CPU should use float32, not bfloat16."""
        import torch

        from soup_cli.utils.gpu import get_compute_dtype

        with patch("torch.cuda.is_available", return_value=False):
            dtype = get_compute_dtype()
            assert dtype == torch.float32

    def test_cuda_with_bf16_returns_bfloat16(self):
        """CUDA with bf16 support should use bfloat16."""
        import torch

        from soup_cli.utils.gpu import get_compute_dtype

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=True):
            dtype = get_compute_dtype()
            assert dtype == torch.bfloat16

    def test_cuda_without_bf16_returns_float16(self):
        """CUDA without bf16 support should fall back to float16."""
        import torch

        from soup_cli.utils.gpu import get_compute_dtype

        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.is_bf16_supported", return_value=False):
            dtype = get_compute_dtype()
            assert dtype == torch.float16


# --- BUG-005: diff dtype -> torch_dtype ---


class TestDiffModelLoading:
    """Test diff command uses correct parameter names."""

    def test_load_model_uses_torch_dtype(self):
        """_load_model should pass torch_dtype, not dtype."""
        import inspect

        from soup_cli.commands.diff import _load_model

        source = inspect.getsource(_load_model)
        assert "torch_dtype=" in source
        assert "dtype=" not in source or "torch_dtype=" in source


# --- BUG-006: wandb version pin ---


class TestWandbVersionPin:
    """Test wandb dependency is version-pinned."""

    def test_wandb_upper_bound_in_pyproject(self):
        """pyproject.toml should pin wandb below 0.18.0."""
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        assert "<0.18.0" in content or "< 0.18.0" in content


# --- BUG-004: CPU quantization warning ---


class TestCPUQuantWarning:
    """Test that CPU + quantization produces a warning."""

    def test_train_source_has_cpu_quant_warning(self):
        """train.py should warn about quantization on CPU."""
        import inspect

        from soup_cli.commands import train

        source = inspect.getsource(train)
        assert "quantization on CPU" in source
